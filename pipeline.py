#!/usr/bin/env python3
"""Residual — real-estate mispricing scanner. Pipeline: hedonic model + permit momentum.
Data: Cambridge, MA open data (property database eey2-rv59, addition/alteration permits qu2z-8suj).
"""
import json, re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

RES_CLASSES = {"CONDOMINIUM","SNGL-FAM-RES","TWO-FAM-RES","CNDO LUX","THREE-FM-RES",
               "SINGLE FAM W/AUXILIARY APT","MULT-RES-2FAM","MULT-RES-3FAM","MULT-RES-4-8-APT",
               "4-8-UNIT-APT","MULTIUSE-RES"}

def norm_street(addr):
    if not isinstance(addr, str): return ""
    a = addr.upper().strip()
    a = re.sub(r"^[0-9]+[A-Z]?(-[0-9A-Z]+)?\s+", "", a)          # strip house number
    a = re.sub(r"\s+(UNIT|APT|#).*$", "", a)
    a = a.replace("AVENUE","AVE").replace("STREET","ST").replace("ROAD","RD").replace("PLACE","PL")
    a = a.replace("TERRACE","TER").replace("DRIVE","DR").replace("LANE","LN").replace("PARKWAY","PKWY")
    a = a.replace("SQUARE","SQ").replace("COURT","CT").replace("CIRCLE","CIR").replace("BOULEVARD","BLVD")
    return re.sub(r"\s+"," ",a).strip()

def house_num(addr):
    if not isinstance(addr, str): return None
    m = re.match(r"^([0-9]+)", addr.strip())
    return int(m.group(1)) if m else None

# ---------- load parcels ----------
df = pd.read_csv("data/propdb.csv", low_memory=False)
d = df[df.yearofassessment == 2026].copy()
d = d[d.propertyclass.isin(RES_CLASSES)].copy()
num_cols = ["landarea","buildingvalue","landvalue","assessedvalue","saleprice",
            "exterior_numstories","interior_livingarea","interior_numunits","interior_totalrooms",
            "interior_bedrooms","interior_kitchens","interior_fullbaths","interior_halfbaths",
            "interior_fireplaces","parking_open","parking_covered","condition_yearbuilt",
            "unfinishedbasementgross","finishedbasementgross"]
for c in num_cols: d[c] = pd.to_numeric(d[c], errors="coerce")
d["saledate"] = pd.to_datetime(d["saledate"], errors="coerce")
d["street"] = d["address"].apply(norm_street)
d["hnum"] = d["address"].apply(house_num)
d["map_sheet"] = d["gisid"].astype(str).str.split("-").str[0]
d["baths"] = d.interior_fullbaths.fillna(0) + 0.5*d.interior_halfbaths.fillna(0)
d["parking"] = d.parking_open.fillna(0) + d.parking_covered.fillna(0)
d["fin_bsmt"] = d.finishedbasementgross.fillna(0)
d["central_air"] = (d.systems_centralair.astype(str).str.upper()=="TRUE").astype(int)
d["yearbuilt"] = d.condition_yearbuilt.replace(0, np.nan)
d = d[(d.interior_livingarea > 200) & (d.assessedvalue > 50000)].copy()

CAT = ["propertyclass","zoning","taxdistrict","map_sheet","condition_overallcondition",
       "condition_overallgrade","exterior_style","systems_heattype"]
NUM = ["interior_livingarea","landarea","interior_bedrooms","baths","interior_totalrooms",
       "interior_numunits","interior_kitchens","interior_fireplaces","yearbuilt",
       "exterior_numstories","parking","fin_bsmt","central_air"]
for c in CAT: d[c] = d[c].astype(str).fillna("NA")

# ---------- training set: arm's-length sales 2018+ ----------
t = d[(d.saledate >= "2018-01-01") & (d.saleprice >= 150000) & (d.saleprice <= 12000000)].copy()
t = t[t.saleprice >= 0.35*t.assessedvalue]           # drop family/deed transfers
t["saleyear"] = t.saledate.dt.year + (t.saledate.dt.month-1)/12.0
FEATS = NUM + ["saleyear"] + CAT
X = t[FEATS].copy(); y = np.log(t.saleprice)
for c in CAT: X[c] = X[c].astype("category")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=7)
m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_depth=None,
        min_samples_leaf=8, l2_regularization=0.3, categorical_features="from_dtype", random_state=7)
m.fit(Xtr, ytr)
pred = np.exp(m.predict(Xte)); actual = np.exp(yte)
ape = np.abs(pred-actual)/actual
metrics = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "median_ape": round(float(np.median(ape))*100,1),
           "within_15pct": round(float((ape<=0.15).mean())*100,1)}
print("MODEL:", metrics)

# ---------- fair value 2026 for every residential parcel ----------
S = d[FEATS[:-len(CAT)-1] + CAT].copy() if False else d[NUM + CAT].copy()
S["saleyear"] = 2026.5
S = S[FEATS]
for c in CAT: S[c] = S[c].astype("category")
d["fair_value"] = np.exp(m.predict(S))

# ---------- permit momentum by street ----------
p = pd.read_csv("data/permits_addalt.csv", low_memory=False)
p["issue_date"] = pd.to_datetime(p["issue_date"], errors="coerce")
p["total_cost"] = pd.to_numeric(p["total_cost"], errors="coerce").fillna(0)
p = p.dropna(subset=["issue_date"])
p["street"] = p["full_address"].astype(str).str.replace(r",.*$","",regex=True).apply(norm_street)
p["hnum"] = p["full_address"].apply(house_num)
NOW = pd.Timestamp("2026-08-01"); H = pd.Timestamp("2024-08-01"); H0 = pd.Timestamp("2022-08-01")
rec  = p[(p.issue_date>=H)  & (p.issue_date<NOW)].groupby("street").agg(n_recent=("id","count"), cost_recent=("total_cost","sum"))
pri  = p[(p.issue_date>=H0) & (p.issue_date<H)].groupby("street").agg(n_prior=("id","count"))
mom = rec.join(pri, how="outer").fillna(0)
mom["ratio"] = (mom.n_recent+1)/(mom.n_prior+1)
mom["momentum"] = (np.clip(np.log(mom.ratio),-1.5,1.5)/1.5*0.65 + np.clip(mom.n_recent/mom.n_recent.quantile(0.95),0,1)*0.35)
mom["momentum"] = ((mom.momentum - mom.momentum.min())/(mom.momentum.max()-mom.momentum.min())*100).round(1)
d = d.merge(mom[["momentum","n_recent","n_prior"]], left_on="street", right_index=True, how="left")
d[["momentum","n_recent","n_prior"]] = d[["momentum","n_recent","n_prior"]].fillna(0)

# approx coords: nearest permit on same street by house number
pc = p.dropna(subset=["latitude","longitude","hnum"]).copy()
pc["latitude"]=pd.to_numeric(pc.latitude,errors="coerce"); pc["longitude"]=pd.to_numeric(pc.longitude,errors="coerce")
pc = pc.dropna(subset=["latitude","longitude"]).sort_values("hnum")
coords = {}
for st_name, g in pc.groupby("street"):
    coords[st_name] = (g.hnum.values, g.latitude.values, g.longitude.values)
def approx_ll(row):
    g = coords.get(row.street)
    if g is None or row.hnum is None or np.isnan(row.hnum if row.hnum is not None else np.nan): return (None,None)
    i = int(np.argmin(np.abs(g[0]-row.hnum)))
    return (round(float(g[1][i]),5), round(float(g[2][i]),5))
ll = d.apply(approx_ll, axis=1, result_type="expand")
d["lat"], d["lon"] = ll[0], ll[1]

# ---------- scored outputs ----------
d = d[d.address.notna() & (d.address.astype(str).str.strip() != "")].copy()
unit_s = d.unit.fillna("None").astype(str)
d["disp_addr"] = d.address.fillna("").astype(str).str.title() + np.where(unit_s.isin(["None","nan",""]), "", " #" + unit_s)
recent = d[(d.saledate>="2024-01-01") & (d.saleprice>=150000) & (d.saleprice>=0.35*d.assessedvalue)].copy()
recent["discount"] = ((recent.fair_value - recent.saleprice)/recent.fair_value*100).round(1)
recent = recent[recent.fair_value.between(2e5, 1.5e7)]
# Discounts beyond ~40% are almost never market inefficiency in Cambridge — they are
# deed-restricted (inclusionary-zoning) units or other non-arm's-length transfers.
recent["flag"] = recent.discount > 40
def mnorm(s):
    lo, hi = s.quantile(0.02), s.quantile(0.98); return np.clip((s-lo)/(hi-lo),0,1)
band = recent[(~recent.flag) & (recent.discount > 0)]
recent["score"] = 0.0
recent.loc[band.index, "score"] = (100*(0.65*mnorm(band.discount) + 0.35*band.momentum/100)).round(1)
recent = recent.sort_values(["score","discount"], ascending=False)

watch = d[(d.saledate < "2019-01-01") | (d.saleprice < 150000)].copy()   # long-held, no recent sale
watch["gap"] = ((watch.fair_value - watch.assessedvalue)/watch.fair_value*100).round(1)
watch = watch[(watch.fair_value.between(3e5, 8e6)) & (watch.interior_livingarea>400)]
watch = watch[watch.gap.between(10, 45)]   # >45% gap = restricted deed or model blind spot, not a target
watch["score"] = (100*(0.6*mnorm(watch.gap) + 0.4*watch.momentum/100)).round(1)
watch = watch.sort_values("score", ascending=False).head(400)

def rows(frame, kind):
    out=[]
    for _,r in frame.iterrows():
        o = {"a": r.disp_addr, "st": r.street.title(), "pc": r.propertyclass,
             "fv": int(r.fair_value), "av": int(r.assessedvalue),
             "sf": int(r.interior_livingarea),
             "bd": 0 if pd.isna(r.interior_bedrooms) else int(r.interior_bedrooms),
             "ba": 0.0 if pd.isna(r.baths) else float(r.baths),
             "yb": int(r.yearbuilt) if not pd.isna(r.yearbuilt) else None,
             "mom": float(r.momentum), "np24": int(r.n_recent), "sc": float(r.score),
             "lat": r.lat, "lon": r.lon}
        if kind=="sale":
            o["sp"] = int(r.saleprice); o["sd"] = r.saledate.strftime("%b %Y"); o["disc"] = float(r.discount)
            o["fl"] = bool(r.flag)
        else:
            o["gap"] = float(r.gap)
        out.append(o)
    return out

heat = (p[(p.issue_date>=H)].dropna(subset=["latitude","longitude"])
        .assign(lat=lambda x: pd.to_numeric(x.latitude,errors="coerce").round(4),
                lon=lambda x: pd.to_numeric(x.longitude,errors="coerce").round(4))
        .dropna(subset=["lat","lon"])[["lat","lon"]].values.tolist())

top_streets = (mom[mom.n_recent>=5].sort_values("momentum",ascending=False).head(15)
               .reset_index()[["street","momentum","n_recent","n_prior"]]
               .assign(street=lambda x: x.street.str.title()).values.tolist())

out = {"generated": "2026-08-02", "city": "Cambridge, MA",
       "metrics": metrics,
       "n_parcels": int(len(d)), "n_sales_scored": int(len(recent)),
       "n_permits_24m": int(len(p[p.issue_date>=H])),
       "sales": rows(recent, "sale"), "watchlist": rows(watch, "watch"),
       "streets": top_streets, "heat": heat[:6000]}
json.dump(out, open("scanner_data.json","w"))
import os; print("JSON KB:", os.path.getsize("scanner_data.json")//1024, "| sales:", len(recent), "| watch:", len(watch))

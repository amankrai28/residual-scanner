# Residual — a real-estate mispricing scanner

**[Live demo](https://amankrai28.github.io/residual-scanner/)** · Cambridge, MA edition · built on open data

Everyone sees the asking price. Nobody sees the **residual** — the gap between what a home sold for and what its physical fundamentals say it should be worth. Residual is a scanner that computes that gap for every residential parcel in Cambridge, then cross-references it with the one leading indicator retail buyers never check: **building-permit velocity** on the same street.

## How it works

**1 · Fair value.** A gradient-boosted hedonic model (scikit-learn `HistGradientBoostingRegressor`) is trained on 5,000+ arm's-length Cambridge sales since 2018 — living area, beds, baths, land, year built, condition, grade, style, zoning, assessor map sheet, and sale timing. On held-out sales it predicts within **8.6% of the actual price at the median** (70.7% of homes within 15%). Every residential parcel then gets a model fair value as of 2026.

**2 · Mispricing.** For every 2024–2026 sale: residual = (fair value − sale price) / fair value. A large positive residual means the home traded below what its fundamentals predict.

**3 · Momentum.** Renovation (addition/alteration) permits are aggregated per street, trailing 24 months vs. the prior 24. Rising permit velocity signals neighborhood reinvestment before prices reflect it. The composite score blends discount (65%) and street momentum (35%).

Two output lists: **recent sales priced below model** (what actually traded cheap vs. fundamentals) and an **under-assessed watchlist** — long-held homes whose model value runs far ahead of assessment, sitting on heating streets. The knock-on-the-door list.

## Honesty notes

Discounts beyond ~40% are flagged, not celebrated — in Cambridge they are almost always deed-restricted (inclusionary-zoning) units or non-arm's-length transfers, and a scanner that calls those "alpha" would be naive. This edition scores closed sales rather than active listings; a live-listings feed plugs into the same model unchanged. A residual is a screen, not an appraisal — it can't see interior condition on sale day, deal terms, or deed restrictions. Not investment advice.

## Reproduce it

```bash
# 1. pull the data (Cambridge Open Data, no auth)
curl "https://data.cambridgema.gov/resource/eey2-rv59.csv?\$where=yearofassessment%20%3E=%202024&\$limit=200000" -o data/propdb.csv
curl "https://data.cambridgema.gov/resource/qu2z-8suj.csv?\$limit=100000" -o data/permits_addalt.csv

# 2. train, score, and emit scanner_data.json
pip install pandas scikit-learn
python pipeline.py

# 3. open index.html — the site is a single self-contained file
```

## Data

[Cambridge Property Database FY2016–FY2026](https://data.cambridgema.gov/resource/eey2-rv59) · [Building Permits: Addition/Alteration](https://data.cambridgema.gov/resource/qu2z-8suj)

---

Built solo with Claude Code by [Aman Rai](https://github.com/amankrai28) — a non-engineer shipping production tools with AI in the loop.

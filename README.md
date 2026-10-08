# Credit Risk Scoring

A probability-of-default (PD) model on Home Credit's application-level
data: given an applicant's application-time attributes, estimate the
probability they default, with calibrated, confidence-interval-quantified
performance rather than a bare accuracy number. Full reasoning trail in
`docs/DECISIONS.md` (30 numbered decisions); plan and scope in `SPEC.md`.

## Data

Home Credit Default Risk (Kaggle), primary application table only —
307,511 rows, ~8% default rate, not the full multi-table schema
(`bureau.csv`, `previous_application.csv`, etc. are out of scope, SPEC.md
Non-goals). Not committed to this repo (competition terms); fetched via
`scripts/get_data.py` against the Kaggle API. 15% (46,127 rows) is held
out, stratified on `TARGET`, and touched exactly once, at the end, after
every modeling choice is already fixed (D6, D16).

Two processed tables are built from the same raw data: a naive **raw**
baseline (median imputation, label-encoded categoricals — a deliberately
unsophisticated reference point) and a **treated** table carrying the
EDA-informed fixes from D11-D15 (sentinel handling, binary-flag encoding,
document-flag aggregation, housing-feature collapse, missingness
indicators). Every statistic either table's cleaning step fits — a
median, a percentile cap — is fit on the train+val rows only, never on
the held-out test rows (D29).

## Models and evaluation protocol

Random Forest and LightGBM, each run through the same protocol: 5-fold
stratified CV over the train+val pool (85% of rows), tuned against PR-AUC
— the primary metric, given the ~8% base rate (D4) — then one fit on the
full train+val pool, evaluated once against the held-out test set (D16).
Calibration is checked for both: reliability curves (quantile-binned, D21)
plus Brier score and log loss, with `CalibratedClassifierCV` tried
(isotonic and sigmoid for RF, isotonic only for LightGBM — D21, D22).
Every held-out test metric reports a nonparametric 95% bootstrap CI (1000
resamples of the test set, D30) alongside its point estimate, so a single
number from 46,127 rows isn't read as more precise than it is.

## Results (held-out test set, touched once)

| Model | Dataset | Calibration | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | Log loss |
|---|---|---|---|---|---|---|
| Random Forest | treated | none | 0.7486 [0.7400, 0.7566] | 0.2364 [0.2236, 0.2497] | 0.0688 | 0.2509 |
| Random Forest | treated | isotonic | 0.7488 [0.7406, 0.7567] | 0.2355 [0.2229, 0.2489] | 0.0682 | 0.2485 |
| Random Forest | treated | sigmoid | 0.7490 [0.7408, 0.7568] | 0.2358 [0.2231, 0.2491] | 0.0686 | 0.2508 |
| LightGBM | treated | none | 0.7602 [0.7517, 0.7678] | 0.2480 [0.2342, 0.2608] | 0.0676 | 0.2455 |
| Random Forest | raw | none | 0.7446 | 0.2294 | — | — |

PR-AUC is the metric that matters here (D4) — ROC-AUC is invariant to
class prevalence and can look fine while the minority (default) class is
barely separated. On that metric, LightGBM's point estimate
(0.2480) is about 5% relative above RF's (0.2364), but the two models'
95% CIs overlap substantially ([0.2342, 0.2497]) — that gap is not
decisively established by two independently-computed CIs, and a proper
paired bootstrap on the difference hasn't been run (D30). Treat it as a
point-estimate observation, not a settled comparison.

The one real raw-vs-treated comparison available is RF: PR-AUC 0.2294
(raw) vs 0.2364 (treated) — a real but modest gain from the D11-D15
feature treatment. No raw-dataset LightGBM run exists to extend this
comparison to the other model.

By gain, LightGBM's top predictors are `EXT_SOURCE_3` (23.6% of total
gain), `EXT_SOURCE_2` (19.6%), `EXT_SOURCE_1` (8.0%), `ORGANIZATION_TYPE`
(6.4%), and `DAYS_EMPLOYED` (4.3%) — full ranking and figure in
`results/feature_importance_lightgbm_treated_*.csv` / `.png` (D24).

## Known limitations

- No operating threshold or cost-based accept/reject decision has been
  built — SPEC.md's Success criteria still calls for this, and D5/D8 are
  explicit that a probability score alone is not an accept/reject model.
- `CODE_GENDER`, `DAYS_BIRTH`, and `NAME_FAMILY_STATUS` are used like any
  other feature; `DAYS_BIRTH` is LightGBM's 6th-ranked predictor by gain.
  No fairness/disparate-impact audit has been done, and a real deployment
  would need one before using this model on real applicants — several
  jurisdictions bar age, sex, or marital status as a basis for a credit
  decision (D28).
- The RF-vs-LightGBM comparison above is a point-estimate observation, not
  a statistically tested one (D30).
- No deployment/serving layer (out of scope — see the separate
  classifier-service project).

## Reproducing

```
uv sync
uv run python scripts/get_data.py          # requires Kaggle API credentials
uv run python scripts/raw_baseline.py
uv run python scripts/clean_features.py
uv run python scripts/train_random_forest.py --dataset treated \
    --n-estimators 250 --max-depth 15 --min-samples-split 25 \
    --min-samples-leaf 8 --max-features sqrt --final
uv run python scripts/train_lightgbm.py --dataset treated \
    --n-estimators 2000 --learning-rate 0.05 --num-leaves 31 \
    --min-child-samples 20 --colsample-bytree 0.5 \
    --early-stopping-rounds 50 --calibration none --final
```

Every run (CV-only or `--final`) appends to `results/model_runs.csv`,
which is the primary source of truth for every number in this README —
see `docs/DECISIONS.md` for why each modeling choice was made.

## Stack

Python, pandas, scikit-learn, LightGBM, scipy (bootstrap/CV confidence
intervals).

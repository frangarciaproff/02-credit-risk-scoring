# Credit Risk Scoring

A probability-of-default (PD) model on Home Credit's application-level
data: given an applicant's application-time attributes, estimate the
probability they default, with calibrated, confidence-interval-quantified
performance rather than a bare accuracy number — plus a downstream
accept/decline and pricing layer built on top of those probabilities.
Full reasoning trail in `docs/DECISIONS.md` (33 numbered decisions); plan
and scope in `SPEC.md`.

## Key results

- **Using the model to decide who to approve beats simply approving
  everyone or declining everyone, at every risk tolerance tested.**
  Depending on how costly a missed default is treated relative to a
  missed good loan, the model-driven policy cuts the cost of wrong
  decisions by 7% to 35% versus the better of the two "blind"
  alternatives. At the most risk-averse setting tested, blindly declining
  every applicant actually becomes *cheaper* than blindly approving every
  applicant — and the model-driven policy still beats both, meaning it's
  finding genuine middle ground, not just picking the less-bad extreme.
- **The interest rate the model implies an applicant should be charged,
  based on their predicted risk, barely tracks what Home Credit actually
  charged them historically.** The two move in the same direction (higher
  predicted risk, somewhat higher historical rate), but weakly — a
  correlation of about 0.08 across 46,127 applicants. That's a real
  relationship, not noise, but far too small to say the model recovered
  how the bank actually priced risk; it may be that real-world pricing
  here barely varies by individual risk, or that the rate figure
  available in this dataset is too imprecise to show a relationship that
  genuinely exists.

Full numbers and derivation below (Credit decision & pricing) and in
`docs/DECISIONS.md` (D31-D32).

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

## Credit decision & pricing

SPEC.md's "Credit decision & pricing" step, implemented in
`scripts/decision_layer.py` and logged in `docs/DECISIONS.md` (D31
design, D32 real results). Reuses the frozen LightGBM test-set
predictions above — lowest Brier score and highest PR-AUC of the four
evaluated model/calibration variants (D31) — no new model is trained.

### Approve / decline

An applicant is approved iff their predicted default probability `p` is
below a threshold `t* = 1/(1+k)`, where `k = C_FN/C_FP` — how many times
worse a missed default is than a missed good loan — is a stated
assumption, never fit from the test data. `k` is swept, not chosen:

| k | t* | approval rate | realized cost/applicant [95% CI] | better trivial baseline | improvement |
|---|---|---|---|---|---|
| 3 | 0.250 | 95.4% | 0.224 [0.217, 0.231] | approve-all: 0.242 | 7.3% |
| 5 | 0.167 | 89.0% | 0.338 [0.328, 0.347] | approve-all: 0.404 | 16.3% |
| 10 | 0.091 | 71.6% | 0.526 [0.512, 0.542] | approve-all: 0.807 | 34.8% |
| 20 | 0.048 | 44.0% | 0.715 [0.696, 0.734] | decline-all: 0.919 | 22.2% |

(cost in units of C_FP; bootstrap CIs per the D30 methodology.) Base rate
in the test set: 8.07%. At `k=20` the two trivial baselines flip —
defaults are costly enough that declining everyone beats approving
everyone — and the model-driven threshold still beats both.

### Pricing

For an applicant with predicted default probability `p`, the rate that
breaks even at a target margin `m` (fixed at 0.05) given a loss-given-default
assumption `LGD` is `r* = [m + p(1+LGD)] / (1-p)`. LGD has no empirical
anchor in this dataset (no recovery/collections data), so it's swept over
0.5-1.0 rather than assumed:

| LGD | median r* | median historical rate | Pearson | Spearman |
|---|---|---|---|---|
| 0.5 | 0.139 | 0.050 | 0.056 | 0.082 |
| 0.6 | 0.145 | 0.050 | 0.056 | 0.082 |
| 0.7 | 0.150 | 0.050 | 0.056 | 0.082 |
| 0.8 | 0.156 | 0.050 | 0.056 | 0.082 |
| 0.9 | 0.162 | 0.050 | 0.056 | 0.082 |
| 1.0 | 0.168 | 0.050 | 0.056 | 0.082 |

(historical rate = `AMT_ANNUITY/AMT_CREDIT`, D7's proxy — not a clean
APR; n=46,127, none dropped.) The two correlation columns are identical
across every LGD row by construction, not by coincidence: at fixed
margin, r* is strictly increasing in `p` for any LGD, so its rank order
equals `p`'s rank order regardless of LGD, and Spearman is rank-based.
The only thing the LGD sweep actually moves is the median r* itself.

Every number above is reproducible from a single frozen model refit —
see `scripts/decision_layer.py` and D31/D32 for the full derivation and
the leakage guard (`k` and `t*` are computed before the test labels are
ever touched).

## Known limitations

- `k` (cost ratio) and `LGD` (loss given default) are stated assumptions
  swept over plausible ranges, not fit from data or collapsed into one
  final policy — a deliberate design choice (SPEC.md), not a gap, but no
  single `(k, LGD)` pair is endorsed here as a deployment-ready threshold.
- `CODE_GENDER`, `DAYS_BIRTH`, and `NAME_FAMILY_STATUS` are used as
  ordinary model features; no fairness or disparate-impact review has
  been done. If reusing these results, or this approach, to train or
  inform a real model, check local regulations on which applicant
  attributes are permitted in credit-risk decisions.
- The RF-vs-LightGBM comparison above is a point-estimate observation, not
  a statistically tested one (D30).
- No deployment/serving layer (out of scope).

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
uv run python -m scripts.decision_layer --dataset treated
```

Every run (CV-only or `--final`) appends to `results/model_runs.csv`,
which is the primary source of truth for every number in this README —
see `docs/DECISIONS.md` for why each modeling choice was made.

## Stack

Python, pandas, scikit-learn, LightGBM, scipy (bootstrap/CV confidence
intervals, correlation tests).

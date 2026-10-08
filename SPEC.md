# SPEC — Credit Risk Scoring

**CV line(s) (draft — finalize once real results exist):**

- Built a credit-default scoring model with cross-validated performance
  estimates and a cost-aware decision threshold, giving lenders a defensible
  probability-of-default signal for underwriting decisions.

## Problem

A lender wants a probability-of-default score to decide whether to extend
credit, and — separately — how much to extend and on what terms. The
decision that matters is the false-negative/false-positive trade-off
(missing a default vs. rejecting a good payer), not AUC in isolation, and
any "how much" decision needs a probability that is not just well-ranked
but well-calibrated.

## Data

- Source: Home Credit Default Risk (Kaggle competition), primary
  application-level table only (`application_train.csv` /
  `application_test.csv`).
- License / usage note: competition data with its own terms — do not commit
  raw files, and check Kaggle's competition terms before publishing detail
  about the raw data itself.
- Access: fetched by `scripts/get_data.py` via the official Kaggle API,
  never committed to the repo.
- **Constraint on this data — read carefully, this trips people up**:
  `application_train.csv` (~307k rows) is the only file with a `TARGET`
  column, so it supplies training *and* evaluation both — split it into
  train / validation / held-out test yourself (stratified on `TARGET`; CV
  over the train+val pool, one untouched final test split used exactly
  once). Kaggle's `application_test.csv` has no label at all; it plays no
  role in local training or evaluation, and was never part of anyone's
  training set either. Its only use is optional: submit predictions and
  read back a Kaggle leaderboard score (scored against labels Kaggle never
  releases) as an external sanity check, not a local metric — verify the
  competition still accepts late submissions before relying on this.
- **Scope boundary (unchanged)**: application-level table only, not the
  full multi-table schema (`bureau.csv`, `previous_application.csv`,
  `installments_payments.csv`, `POS_CASH_balance.csv`,
  `credit_card_balance.csv`). Revisit only if benchmark results motivate it
  — see `docs/DECISIONS.md`, D1.
- **Known data quality issue to correct**: `DAYS_EMPLOYED` contains a
  sentinel value of 365243 (≈1000 years), concentrated in a specific
  applicant segment rather than spread randomly — treat as
  missing-not-at-random. Replace with NaN *and* keep a binary indicator
  column before imputing, so the signal isn't discarded along with the bad
  value. Verify the fix with an explicit assertion after cleaning
  (`assert not (df["DAYS_EMPLOYED"] == 365243).any()`), not a print
  statement — see `docs/DECISIONS.md`, D2, for why a print wouldn't have
  caught this.

## Plan

1. **Data acquisition** — `scripts/get_data.py` pulls the official Kaggle
   files directly (not a pre-split mirror); document the exact fetch
   command and Kaggle auth setup in the README's "Running it" section.
2. **EDA & cleaning** (`scripts/eda.py` → `results/pre_process_results.md`
   + `results/figures/`) — class balance, missingness map, the
   `DAYS_EMPLOYED` fix above, and which features actually separate the
   classes before any model is trained. Document findings even when they
   change what steps 3-5 plan to do — see "Spec change policy" below.
3. **Random Forest benchmark** (`scripts/train_random_forest.py`) — reuse
   the hyperparameters already found via a prior `RandomizedSearchCV`
   rather than re-tuning (see `docs/DECISIONS.md`, D3) as CLI defaults; a
   proper stratified k-fold CV (not a single split) on the corrected
   data, reporting ROC-AUC *and* PR-AUC, mean +/- 95% CI across folds.
4. **Gradient boosting (LightGBM)** (`scripts/train_lightgbm.py`) — same
   CV protocol as step 3, reported against `average_precision` (PR-AUC)
   as the metric that actually matters for this target — see D4. A
   calibration check (reliability curve, Brier score) runs alongside the
   held-out test evaluation: step 5 consumes these probabilities
   directly, not just their rank order, so calibration is load-bearing
   here in a way it wasn't for steps 3-4 alone.
   - Both scripts share one CV/held-out-test protocol (D6, D16): a fixed
     15% test split (same rows on every run, any model, any dataset) is
     carved out up front and never touched by default — only CV metrics
     (over the remaining train+val pool) drive hyperparameter choice.
     Every hyperparameter is a CLI flag, meant to be run and compared
     repeatedly by hand, not searched; a `--final` flag gates the
     one-time exception that fits on the full train+val pool and
     evaluates the held-out test split exactly once. Every run (CV-only
     or `--final`) appends one row — hyperparameters plus results — to
     `results/model_runs.csv` for side-by-side comparison across runs,
     models, and the raw/treated datasets (D10's 2x2 grid).
5. **Decision layer — who gets credit, and how much** — see "Credit
   decision & pricing" below.

## Feature treatment plan (finalized after EDA — see `docs/DECISIONS.md`, D11-D16)

Decided during step 2 (EDA & cleaning), from the findings in
`results/pre_process_results.md` and `results/target_correlation_results.md`,
applied before any model is trained:

- `FLAG_OWN_CAR` / `FLAG_OWN_REALTY`: encode Y/N as 1/0.
- `FLAG_MOBIL` / `FLAG_CONT_MOBILE`: drop — neither shows a real
  TARGET-rate difference (two-proportion z-test, not eyeballing).
- `FLAG_DOCUMENT_*`: keep `FLAG_DOCUMENT_3` and `FLAG_DOCUMENT_21` as their
  own standalone columns; build one `DOCUMENTS_PROVIDED_COUNT` feature from
  the subset that survives Bonferroni correction across the 20 document
  flags — {6, 8, 9, 11, 13, 14, 15, 16, 18}; drop the rest from any
  aggregate (their apparent effect is a small-n artifact).
- Housing block (`_AVG`/`_MODE`/`_MEDI`, ~16 attributes x 3 columns each):
  keep `_MEDI` only, drop `_AVG`/`_MODE` — correlation between the three is
  0.96-0.998 for every attribute, so they're redundant, not complementary;
  PCA considered and rejected as unnecessary given that.
- High-missingness columns generally: resolved by D12, not a blanket
  drop — see below.
- `OWN_CAR_AGE`: impute missing as 0 — its 66% missingness matches
  `FLAG_OWN_CAR == 'N'` almost exactly (202,924 vs. 202,929 rows), so this
  is a structural "no car" gap, not an MNAR problem needing a general
  strategy.
- Remaining missing values, per column, split by whether an observed
  column explains the missingness (`docs/DECISIONS.md`, D12) —
  "partially explained" counts as explained:
  - `OCCUPATION_TYPE`: partially explained (100% missing for
    Pensioner/Unemployed, ~16% unexplained residual elsewhere) — filled
    with a `"Missing"` category for both models.
  - Housing block (14 `_MEDI` columns, `TOTALAREA_MODE`, and the 4
    remaining categorical `_MODE` columns): not explained by
    `NAME_HOUSING_TYPE`, but explains itself — missing together as one
    block (Jaccard 0.74-0.96). One shared `HOUSING_INFO_MISSING`
    indicator, numeric columns median-filled, categorical columns filled
    with `"Missing"`, both models.
  - `EXT_SOURCE_1`/`EXT_SOURCE_3` (and `EXT_SOURCE_2`, grouped with them):
    unexplained — no observed column gets anywhere close. A
    `_MISSING` indicator per column for both models; the value itself
    is median-imputed for Random Forest only. LightGBM keeps the real NaN
    and uses its native handling.
  - `AMT_REQ_CREDIT_BUREAU_*` (6 columns, identical missing rows):
    unexplained. One shared `CREDIT_BUREAU_REQUEST_MISSING` indicator for
    both models; median-imputed for Random Forest only, same as above.
  - Social-circle columns' residual missingness (after the D11 cap): no
    correlate found, negligible count — Random Forest only, same pattern.
  - `DAYS_EMPLOYED` (D13): imputed for both models — median of the
    genuine values (-1648.0). Fully explained by construction
    (`DAYS_EMPLOYED_ANOMALY` is 1 exactly where this is NaN); D2 added the
    indicator but never decided a fill value until now.
  - Five columns under 0.5% missing (D13), none investigated for a
    mechanism at that size — imputed for both models: `AMT_GOODS_PRICE`,
    `AMT_ANNUITY`, `CNT_FAM_MEMBERS`, `DAYS_LAST_PHONE_CHANGE` (median),
    `NAME_TYPE_SUITE` (`"Missing"` category).
  - Implementation: `scripts/clean_features.py`'s `build_clean_features()`
    produces the common treated table (81 columns, real NaN only in the
    "unexplained" group above, ~14 categorical columns still stored as
    strings); `finalize_for_random_forest()` is the additional step
    Random Forest (step 3) calls before training — median-imputes the
    unexplained group *and* label-encodes every remaining categorical
    column (D15), giving `train_treated_rf.csv` (81 columns, zero
    missing, fully numeric). `train_lightgbm.py` casts the same
    categorical columns to pandas `"category"` dtype at load time
    instead (D15) — a CSV can't preserve dtype, so this can't happen in
    `clean_features.py` itself.

**Raw baseline (D10's "raw" cell, realized in code — see D14):**
`scripts/raw_baseline.py` builds the shared raw-baseline table used for
both models: every categorical column label-encoded as-is
(`pandas.factorize`, NaN -> -1), any remaining numeric NaN generically
median-filled — nothing else touched (the `DAYS_EMPLOYED` sentinel, every
outlier, and the `FLAG_DOCUMENT_*`/housing-block redundancy all pass
through unchanged). 122 columns, same as the source data, zero missing.

**The three datasets D10's 2x2 grid needs, confirmed on the full
307,511-row `application_train.csv` (all gitignored, regenerated by
running the two scripts, not committed):**
`data/processed/train_raw.csv` (122 columns), `train_treated_lightgbm.csv`
(81 columns, real NaN in the unexplained group), `train_treated_rf.csv`
(81 columns, zero missing).
- `REGION_RATING_CLIENT` / `_W_CITY`: no encoding change — already ordinal
  integers; the `_NOT_*` variants are separate pre-existing binary columns.
- `AMT_INCOME_TOTAL`: cap at the 99.9th percentile (900,000), row kept, not
  dropped. The max value (117,000,000, `SK_ID_CURR=114967`) pairs an
  entirely ordinary loan/annuity/occupation/education profile with a
  9-figure income — a data-entry error, not a real outlier — and 5 more
  values above 5M look like the same systematic issue (clean multiples of
  ~4.5M). The row's `TARGET=1` label (an actual default) is kept; only the
  income value is capped.
- Social-circle columns (`OBS_30`/`DEF_30`/`OBS_60`/`DEF_60_CNT_SOCIAL_CIRCLE`):
  cap each at its own 99.9th percentile (17 / 4 / 16 / 3) for modeling
  robustness, but treated as genuine extreme values, not errors — the
  single row at both the `OBS_30` and `DEF_30` maxima is internally
  consistent (30-day and 60-day figures track each other), and
  `DEF_30 > OBS_30` / `DEF_60 > OBS_60` never occurs anywhere in the
  307,511-row dataset.

## Credit decision & pricing

**Constraint that shapes this section**: this dataset contains only
already-disbursed loans — every row is an approved applicant's good- or
bad-payer outcome. There is no rejected-applicant population here. This
isn't a scope gap to be worked around: whether a rejected applicant would
have been a good or bad payer is not observable on *any* dataset, ever, by
construction — a rejected applicant never receives a loan to repay or
default on. See `docs/DECISIONS.md`, D5/D7/D8, for the full reasoning,
including why the reject-inference literature can only validate its
correction methods against simulated rejections over fully-labeled data,
never against real, unobserved ones.

**Scope (final — see D8): a PD-driven decision-policy illustration, nothing
more.** Calibrated PD (step 4) plus an explicitly stated cost matrix (cost
of a missed bad payer vs. lost good-payer business) gives an
approve/decline threshold; a risk-tiered credit-amount/annuity check
against `AMT_CREDIT`/`AMT_ANNUITY` (confirmed present — see D7 for the full
column list and the interest-rate caveat) illustrates the "how much" side.
Reject inference (a LendingClub case study, or a simulated-rejection
experiment on this dataset) was considered and dropped — see D8.

**The model does not make the business decision.** A PD model outputs a
probability. The cost matrix, the threshold, and any pricing rule are
business decisions — risk appetite, cost of capital, competitive rates,
regulatory constraints — that this project treats as a stated, supplied
assumption, not something the data or the model determines. The point of
this section is to demonstrate the decision *framework* and be explicit
about that boundary, not to prescribe an actual lending policy.

**Protected attributes are not handled — a real limitation, not an
oversight.** `CODE_GENDER`, `DAYS_BIRTH` (age), and `NAME_FAMILY_STATUS`
are present, untouched, in every processed table (raw and both treated
variants) and are used by both models exactly like any other feature.
This isn't a marginal effect: in the logged LightGBM run, `DAYS_BIRTH`
ranks 6th of 79 features by gain, `CODE_GENDER` 17th, `NAME_FAMILY_STATUS`
39th (D28). No fairness metric (disparate-impact ratio, equalized odds,
or similar) is computed anywhere in this project, and no protected
attribute or proxy is dropped, reweighted, or audited. In the US, using
sex, marital status, or age this way in an actual credit decision is what
ECOA/Regulation B prohibits; other jurisdictions have their own
equivalents. This is a deliberate scope boundary for a portfolio project,
not a claim that the framework is fair as built — a real deployment would
need a fairness audit, and very likely feature changes, before this model
(or one like it) could decide on real applicants.

## Candidate directions (pick during the session — not a checklist to clear)

- Stratified k-fold CV instead of a single train/val split; report the
  distribution of AUC across folds, not one number.
- Class imbalance is real here (default is the minority class): don't rely
  on ROC-AUC alone to judge it — see Success criteria.
- SHAP or permutation importance on the top features, framed for a
  non-technical underwriting audience.
- WOE/IV logistic regression scorecard as an interpretable baseline (see
  "Credit decision & pricing").

## Success criteria

- Cross-validated ROC-AUC reported with a confidence interval, not a point
  estimate.
- PR-AUC (average precision) reported alongside ROC-AUC. ROC-AUC is
  invariant to class prevalence — TPR and FPR are each computed within one
  class, so the curve doesn't move just because the target is imbalanced.
  PR-AUC is what actually reflects performance on the minority (default)
  class, and is the number that should move if a technique is really
  helping with imbalance.
- Calibration reported (reliability curve + Brier score) — required before
  any threshold or pricing claim in the decision layer means anything.
- A stated operating threshold with its expected cost trade-off.
- A baseline-vs-final comparison table (ROC-AUC and PR-AUC both).

## Non-goals

- No deployment/serving here — that pattern lives in the classifier-service
  project.
- No full feature-engineering rebuild across every Home Credit table, just
  the primary application data.
- No claim of a validated accept/reject model from this dataset alone —
  see "Credit decision & pricing."
- No fairness/disparate-impact analysis, and no removal of protected
  attributes or their proxies (`CODE_GENDER`, `DAYS_BIRTH`,
  `NAME_FAMILY_STATUS`) — see "Credit decision & pricing" (D28).

## Stack

Python, pandas, scikit-learn, LightGBM, scipy (CV confidence intervals).
Candidate additions for step 5:
`category_encoders` or hand-rolled WOE binning (TBD), `shap` (optional, for
the interpretability direction).

## Spec change policy

This spec is not frozen for the duration of the project, and it isn't
rewritten wholesale every step either — it changes at whichever comes
first: a planned review point (finishing a step in the Plan above), or the
moment something discovered mid-step invalidates an assumption (the
`DAYS_EMPLOYED` bug, the reject-inference constraint above, a better
dataset found). What doesn't happen is a silent edit: every substantive
change to Data / Success criteria / Non-goals gets one entry in
`docs/DECISIONS.md` — date, what changed, what evidence triggered it, what
was true before. Git history is the record for everything else; this file
always reflects the current state, `docs/DECISIONS.md` is the append-only
record of why it moved.

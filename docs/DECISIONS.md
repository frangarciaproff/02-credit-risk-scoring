# Decisions log

Append-only. A past entry is never edited — if a decision changes, a new
entry supersedes it and says so. This is the reasoning trail behind
`SPEC.md`; `SPEC.md` always reflects the current state, this file records
how it got there.

## D0 — Spec change policy (2026-09-18)

Decided: `SPEC.md` changes either at a planned per-step review or
immediately when new evidence invalidates an assumption — not on a fixed
cadence, and never silently. See `SPEC.md`'s "Spec change policy" for the
rule; this log is where each individual change gets recorded.

## D1 — Dataset scope (2026-09-18)

Considered: (a) application-level Home Credit Default Risk table only
[current], (b) the fuller multi-table Home Credit schema
(bureau/previous_application/installments/POS/credit_card), (c) the 2024
"Home Credit - Credit Risk Model Stability" competition — a genuinely novel
temporal-drift framing, but its natural time axis doesn't exist in the 2018
application-level data, so adopting it here would mean switching datasets
entirely, not just adding a metric, (d) LendingClub accepted+rejected
data — has real reject labels, which (a)-(c) don't.

Decided: keep (a) for the core PD/benchmark work (Plan steps 2-4).
LendingClub reject data is considered only as an optional stretch for the
decision-layer step (step 5), not a full dataset switch.

Why: (a) keeps steps 2-4 scoped and comparable to prior benchmark work;
introducing reject data changes the type of question being asked (default
risk vs. approval policy), so it belongs only in step 5, if at all.

## D2 — `DAYS_EMPLOYED` anomaly handling (2026-09-18)

Found: prior preprocessing code called
`df["DAYS_EMPLOYED"].replace({365243: np.nan})` without reassigning the
result or passing `inplace=True` — the intended fix never actually applied,
so the 365243 sentinel (≈1000 years) survived into imputation and
`MinMaxScaler.fit`, which fits off the max value seen in training and would
compress every genuine `DAYS_EMPLOYED` value toward 0.

Decided: replace with NaN *and* keep a binary anomaly-indicator column
before imputing — the anomaly isn't random noise, it's concentrated in a
specific applicant segment and is itself informative, so flagging preserves
that rather than discarding it. Add a unit test asserting no 365243 values
remain post-cleaning, not just a print statement: this exact bug is
invisible to a print because `.replace()` executes without error, it just
doesn't mutate anything in place.

## D3 — Random Forest benchmark hyperparameters (2026-09-18)

Decided: reuse hyperparameters already found via a prior
`RandomizedSearchCV(scoring="roc_auc")` run (`n_estimators=200,
min_samples_split=25, min_samples_leaf=8, max_features="sqrt",
max_depth=15, class_weight=None`) rather than re-tuning.

Why: this step is meant to be a fast benchmark, not the project's main
model — re-tuning here would spend time step 4 needs more.

Caveat: these were selected optimizing ROC-AUC; PR-AUC for this exact
configuration is measured, not tuned-for. A weak PR-AUC here is a reason to
retune later, not a reason to distrust the PR-AUC measurement itself.

## D4 — Gradient boosting tuning metric (2026-09-18)

Decided: unlike the RF benchmark, tune LightGBM with
`scoring="average_precision"` (PR-AUC), not `"roc_auc"`.

Why: PR-AUC is this project's stated primary metric for the imbalanced
target (`SPEC.md`, Success criteria). Tuning against a different metric
than the one being reported was a mismatch in the prior version of this
pipeline, worth fixing on the model meant to be the main result.

## D5 — "Who gets credit and how much" scope (2026-09-18)

Found: this dataset contains only already-disbursed loans — every row was
approved. There is no rejected-applicant population, so an accept/reject
classifier can't be trained on this data without importing a second,
labeled-reject dataset. This is the reject-inference problem: training only
on approved applicants is a known source of bias in credit scoring, and
current literature (2025-2026) shows naive fixes for it can raise accuracy
while making the model worse at actually flagging risk — e.g. by inflating
the training default rate through a feedback loop when a model's own
predictions are used as ground truth for the rejected population. Sources:
"The Illusion of Improvement: Reject Inference Strategies in Credit
Scoring" (arXiv, 2026) and "Balancing Performance and Reject Inclusion: A
Novel Confident Inlier Extrapolation Framework for Credit Scoring" (arXiv,
2025). Not a solved problem — not one to route around with an
undocumented workaround.

Decided (core scope): a PD-driven decision policy — calibrated PD (step 4)
plus a stated cost matrix, giving an approve/decline threshold and a
risk-tiered amount/annuity rule, checked against actual
`AMT_CREDIT`/`AMT_ANNUITY` in the data. Explicitly not framed as a learned
approval model.

Decided (stretch, optional): a small reject-inference case study on
LendingClub's accepted+rejected data, only if Plan steps 1-5 (core) finish
with time left.

Also decided: add a WOE-binned logistic regression scorecard alongside the
GBM specifically for this step — the interpretable, auditable form the
credit industry still checks ML scores against, which is exactly what
"how do you argue this decision" requires.

## D6 — Train/val/test partitioning vs. Kaggle's real test file (2026-09-18)

Found: SPEC.md's Data section said evaluation "must come from
`application_train.csv` alone" but didn't spell out how train/val/test
actually partition, which is exactly what caused the "how do we test
without a target" question.

Clarified: `application_train.csv` alone supplies every local partition —
train, validation, and a held-out test split — via a stratified split
and/or k-fold CV over the train+val pool, with one untouched final test
split reported exactly once. Kaggle's `application_test.csv` is unlabeled
and plays no role in local training or evaluation; it never did. Its only possible use is an optional Kaggle
leaderboard submission (scored against labels Kaggle never releases) as an
external sanity check — separate from, and not a substitute for, the local
CV/test metrics this project reports. Training itself was never affected
by any of this: it only ever needed `application_train.csv`, which has a
`TARGET` value for every row.

## D7 — CI-EX mechanism (detail) and confirmed pricing columns (2026-09-18)

Extends D5, does not replace it.

Confirmed the application-level schema has `AMT_INCOME_TOTAL`, `AMT_CREDIT`,
`AMT_ANNUITY`, `AMT_GOODS_PRICE`, and
`AMT_REQ_CREDIT_BUREAU_{HOUR,DAY,WEEK,MON,QRT,YEAR}`. No interest-rate
column at this level — `RATE_INTEREST_PRIMARY`/`RATE_INTEREST_PRIVILEGED`
live in `previous_application.csv`, out of scope per D1. Candidate derived
ratios for the decision-layer step, standard in public Home Credit
solutions: `AMT_ANNUITY/AMT_CREDIT` (implied periodic rate),
`AMT_CREDIT/AMT_GOODS_PRICE` (financed premium over goods price),
`AMT_CREDIT/AMT_INCOME_TOTAL` and `AMT_ANNUITY/AMT_INCOME_TOTAL`
(debt-burden ratios) — legitimate stand-ins for a real APR given the scope
boundary, not a novel contribution.

CI-EX mechanism, for the D5 stretch option: two gates before any rejected
applicant gets a pseudo-label. (1) An Isolation Forest fit separately on
each accepted class (good/bad); a rejected applicant that's an outlier to
both is shelved, not labeled this round — this is what stops blind
extrapolation into feature regions with no real accepted-class support.
(2) Among the remaining inliers, only the top-confidence tail of a
supervised classifier's P(bad) ranking gets labeled (score >= 0.5 -> bad,
else good) — confidence-thresholded self-training, avoiding the ambiguous
mid-boundary cases where mislabeling risk concentrates. Repeated
iteratively (a small batch of samples added per round, ~10-20 rounds), so
the training pool expands cautiously instead of in one blind pass — the
"Illusion of Improvement" failure mode (a 22% inflated training default
rate) comes specifically from an ungated, one-shot version of this idea.
Evaluated with AUK (mean "kickout" across acceptance rates 1-100%; kickout
nets bad payers newly caught against good payers newly and wrongly
rejected, relative to a benchmark policy) rather than AUC alone, because
AUC can rise while kickout — the thing that actually matters here —
doesn't.

## D8 — Scope finalized: core PD-policy only, reject inference dropped (2026-09-18)

Extends D5 and D7; supersedes D5's "stretch, optional" line (D5 itself is
left as-is — this entry records the outcome of that open question).

Decided: drop the reject-inference stretch entirely — neither the
LendingClub case study nor a simulated-rejection experiment on this
dataset. Reasoning: the quantity such a correction would be estimating —
whether a rejected applicant would have been a good or bad payer — is not
merely hard to estimate, it is unmeasurable by construction, on any real
dataset, because a rejected applicant never receives a loan to repay or
default on. The literature's own validation of correction methods (D7)
only works because it is run over simulated rejections on fully-labeled
data, i.e. it validates a simulation of the problem, not the problem
itself. A correction applied to a genuinely unobserved rejected population
can never be checked against a real outcome — not on Home Credit, not on
LendingClub, not on any dataset. That is a limitation worth stating
precisely, not one worth spending build time routing around.

Scope for step 5, final: a PD-driven decision-policy illustration only —
calibrated PD (step 4) plus a stated cost matrix, giving an approve/decline
threshold and a risk-tiered amount/annuity check against
`AMT_CREDIT`/`AMT_ANNUITY`. Requires no inference about rejected applicants
at all.

Also clarified, and now stated directly in `SPEC.md`: the model does not
make the business decision. A PD model outputs a probability; the
threshold, the cost matrix, and any pricing rule are business decisions
(risk appetite, cost of capital, competitive rates, regulatory constraints)
that this project treats as a supplied, stated assumption — not something
derived from the data. Step 5 demonstrates that decision framework; it does
not prescribe an actual lending policy.

Terminology, kept precise from here on: `TARGET` (what the PD model
predicts) is "good payer" vs. "bad payer" (Home Credit's own definition —
1 = payment difficulties on early installments). "Accepted" vs. "rejected"
refers to the underwriting decision itself, a separate axis this dataset
has no data on at all (every row here is accepted). Not interchangeable
with good/bad payer.

## D9 — `application_test.csv` made optional in `validate()` (2026-09-18)

Found: manually downloading just `application_train.csv` from Kaggle's web
UI is a reasonable, simpler alternative to the API for a one-off fetch —
and it's all step 1 actually needs, per D6 (`application_test.csv` is
unlabeled and unused in any local split). `validate()` as first written
required both files and would have crashed on this exact workflow.

Decided: `application_train.csv` is required in `validate()`;
`application_test.csv` is optional and only checked (for the absence of
`TARGET`) if present.

## D10 — Baseline (untreated) vs. treated feature comparison, 2x2 grid (2026-09-24)

Decided: the "baseline-vs-final comparison table" in `SPEC.md`'s Success
criteria is a 2x2 grid, not a single row — {raw, treated} feature
preparation x {Random Forest, LightGBM} model, evaluated on identical CV
folds so the comparison isolates the effect of cleaning/feature work from
sampling variance, not sampling variance itself.

- **raw**: the minimum treatment needed for each model to run at all — no
  `DAYS_EMPLOYED` fix (D2), no missingness flags, categorical columns
  label-encoded as-is, numeric NaNs median-imputed. Nothing else touched.
- **treated**: the actual EDA-driven cleaning from step 2 — D2's fix, plus
  whatever findings from `results/pre_process_results.md` and
  `results/target_correlation_results.md` end up implemented (housing-block
  reduction, document-flag aggregation, outlier handling, etc.).

`data/raw/application_train.csv` itself is never mutated by either path —
there is no separate "untreated dataset" to create or store; both paths
read the same raw file and diverge only in the feature-preparation step
that follows.

## D11 — Feature treatment plan finalized from EDA findings (2026-09-25)

Extends D10's treated path with the specific decisions, made after
z-testing and row-level checks on `results/pre_process_results.md` and
`results/target_correlation_results.md` (see D3 in that discussion for the
significance-testing standard applied to `FLAG_DOCUMENT_*`: two-proportion
z-test + Bonferroni correction across 20 simultaneous tests, alpha =
0.05/20).

Decided:

- `FLAG_OWN_CAR` / `FLAG_OWN_REALTY`: encode Y/N as 1/0.
- `FLAG_MOBIL` / `FLAG_CONT_MOBILE`: drop. Neither shows a real TARGET-rate
  difference by value (z approx -0.21, p approx 0.84 for
  `FLAG_CONT_MOBILE`; `FLAG_MOBIL` has only 1 row at value 0, no usable
  signal either way).
- `FLAG_DOCUMENT_*` (20 columns, `_2` through `_21`): keep `FLAG_DOCUMENT_3`
  and `FLAG_DOCUMENT_21` as their own standalone columns (not aggregated).
  Build one new `DOCUMENTS_PROVIDED_COUNT` feature from the subset that
  survives Bonferroni correction: {6, 8, 9, 11, 13, 14, 15, 16, 18}. Drop
  the rest ({2, 4, 5, 7, 10, 12, 17, 19, 20}) from any aggregate — their
  apparent target-rate deviation is a small-n artifact, not a real effect
  (e.g. doc_2: n=13; doc_21 is kept anyway per below, not for its
  statistical strength but because there's no cost to keeping it).
  `FLAG_DOCUMENT_21` specifically: left as its own column rather than
  merged or dropped — no further information available on it (n=103,
  fails the corrected significance threshold) either way, and a standalone
  column costs nothing a tree-based model can't just assign near-zero
  importance to.
- Housing block (`_AVG` / `_MODE` / `_MEDI`, ~16 attributes x 3 columns
  each): keep `_MEDI` only, drop `_AVG` and `_MODE`. Pairwise correlation
  between the three variants is 0.96-0.998 for every attribute measured —
  they are the same underlying quantity through three aggregation methods,
  not complementary information. PCA was considered and rejected: with
  correlation already this uniform, a rotation buys nothing a single
  column selection doesn't already get for free.
- High-missingness columns generally: no blanket drop by missingness
  percentage alone. Sequence: apply the housing-block collapse above
  first (removes ~32 of the highest-missingness columns immediately by
  construction), then test a "was this value observed" indicator against
  TARGET for whatever high-missingness columns remain, before deciding
  column-by-column.
- `OWN_CAR_AGE`: impute missing as 0, relying on `FLAG_OWN_CAR` to
  disambiguate "no car" from "car age genuinely unknown". Justification:
  66% missingness lines up almost exactly with `FLAG_OWN_CAR == 'N'`
  (202,924 rows) vs. missing `OWN_CAR_AGE` (202,929 rows) — a structural
  gap, not missing-not-at-random noise needing a general imputation
  strategy.
- `REGION_RATING_CLIENT` / `REGION_RATING_CLIENT_W_CITY`: no encoding
  change — already stored as ordinal integers, kept as-is (relevant later
  if a WOE-binned logistic scorecard is added per D5/D8, a no-op for
  RF/LightGBM today). The `REGION_NOT_*` / `_NOT_WORK_*` variants are
  separate pre-existing binary columns, not touched here.
- `AMT_INCOME_TOTAL`: cap at the 99.9th percentile (900,000), row kept, not
  dropped. Row-level check on the max value (117,000,000,
  `SK_ID_CURR=114967`): `AMT_CREDIT=562,491`, `AMT_ANNUITY=26,194.5`,
  `OCCUPATION_TYPE="Laborers"`, `NAME_EDUCATION_TYPE="Secondary / secondary
  special"` — an entirely ordinary loan/annuity/occupation/education
  profile paired with a 9-figure income confirms a data-entry error, not a
  genuine outlier value. Five more rows sit above 5,000,000
  (18,000,090 / 13,500,000 / 9,000,000 / 6,750,000 / 4,500,000) and look
  like the same systematic issue (clean multiples of ~4.5M) rather than
  independent typos — a single percentile-based cap handles all six the
  same way rather than hand-adjudicating each. The row is not dropped:
  this applicant's `TARGET=1` (an actual default) is a real label that a
  row-drop would discard along with the bad income value.
- Social-circle columns (`OBS_30_CNT_SOCIAL_CIRCLE`,
  `DEF_30_CNT_SOCIAL_CIRCLE`, `OBS_60_CNT_SOCIAL_CIRCLE`,
  `DEF_60_CNT_SOCIAL_CIRCLE`): cap each at its own 99.9th percentile
  (17 / 4 / 16 / 3 respectively) for modeling robustness, but treated as
  genuine extreme values, not data errors, unlike the income case above.
  The row holding both the `OBS_30` max (348) and the `DEF_30` max (34) is
  a single applicant (`SK_ID_CURR=272071`, `TARGET=0`) whose 30-day and
  60-day figures track each other closely (348 vs. 344; 34 vs. 24) —
  internally consistent, not contradictory. Dataset-wide check:
  `DEF_30 > OBS_30` and `DEF_60 > OBS_60` each occur 0 times across all
  307,511 rows, confirming the field is logically well-behaved everywhere,
  not just for this row.

Not yet decided (deferred to when step 2 cleaning code is actually
written): the exact TARGET-vs-observed-indicator test for whichever
high-missingness columns remain after the housing-block collapse.

## D12 — Remaining missing-value handling after D11, split by explainability (2026-09-25)

Extends D11. Investigated whether an observed column explains the
missingness (crosstabs + Jaccard overlap between missing-row sets,
`data/raw/application_train.csv`) for every column still missing after
D11's treatment, to decide imputation per the same standard as D2/D11: use
the mechanism when there is one, don't invent one when there isn't.

Decided: "partially explained" is treated as explained — imputed for
*both* Random Forest and LightGBM. Fully unexplained columns are imputed
for Random Forest only (it cannot accept NaN input at all); LightGBM keeps
the real NaN and uses its native missing-value handling instead, since
papering over a signal it can already use natively would only hurt it.

**Partially explained (imputed for both models):**

- `OCCUPATION_TYPE` (96,391 missing): `NAME_INCOME_TYPE` = Pensioner or
  Unemployed is 100.0% missing here (n=55,362 and 22) — structurally "no
  occupation" because not working, the same shape of mechanism as D11's
  `OWN_CAR_AGE` fix. Doesn't explain all of it: Working/Commercial
  associate/State servant still show 15.7-17.4% missing with no
  explanation found. Filled with a `"Missing"` category regardless — a
  tree model can split on that category same as any other, so the
  unexplained residual costs nothing by being lumped in with the explained
  majority.
- Housing block (14 `_MEDI` columns + `TOTALAREA_MODE`, plus
  `FONDKAPREMONT_MODE`/`HOUSETYPE_MODE`/`WALLSMATERIAL_MODE`/
  `EMERGENCYSTATE_MODE`): `NAME_HOUSING_TYPE` does *not* explain this —
  missingness is 36-80% in every category, no category stands out. But the
  18 columns explain each other: missing-row-set Jaccard overlap with
  `APARTMENTS_MEDI` is 0.93-0.96 for `HOUSETYPE_MODE`/`WALLSMATERIAL_MODE`/
  `EMERGENCYSTATE_MODE`/`TOTALAREA_MODE`, 0.74 for `FONDKAPREMONT_MODE` —
  one shared "no building data on file" event, not 18 independent gaps.
  One shared `HOUSING_INFO_MISSING` indicator (from `APARTMENTS_MEDI`)
  replaces 18 individual ones. Numeric columns filled with their own
  median (e.g. `APARTMENTS_MEDI` 0.0864, `COMMONAREA_MEDI` 0.0208,
  `TOTALAREA_MODE` 0.0688, `YEARS_BUILD_MEDI` 0.7585); categorical columns
  filled with `"Missing"`.

**Unexplained (Random Forest only; LightGBM keeps the NaN):**

- `EXT_SOURCE_1` (173,378 missing) and `EXT_SOURCE_3` (60,965): income
  type shifts the rate (Pensioner 76.8% vs Working 53.2% for
  `EXT_SOURCE_1`), and the D2 `DAYS_EMPLOYED` anomaly flag shifts it a
  little more (76.8% vs 51.9%) — but even the best-explained group is
  still >50% missing. Nowhere close to a clean mechanism. `EXT_SOURCE_2`
  (660 missing, negligible) grouped with its two siblings for consistency
  rather than given separate treatment. `EXT_SOURCE_{1,2,3}_MISSING`
  indicators added regardless (safe for both models); the values
  themselves are median-imputed (0.505998 / 0.565961 / 0.535276
  respectively) only in `finalize_for_random_forest()`.
- `AMT_REQ_CREDIT_BUREAU_{HOUR,DAY,WEEK,MON,QRT,YEAR}` (41,519 missing,
  identical rows across all 6): Jaccard overlap against `OCCUPATION_TYPE`
  missingness (0.106), `EXT_SOURCE_1` missingness (0.132), and
  social-circle missingness (0.004) are all low — doesn't line up with
  anything else checked. One shared `CREDIT_BUREAU_REQUEST_MISSING`
  indicator (from `AMT_REQ_CREDIT_BUREAU_YEAR`) added for both models;
  median-imputed for Random Forest only (0 for HOUR/DAY/WEEK/MON/QRT, 1
  for YEAR).
- Social-circle columns' residual missingness (1,021 rows, after D11's
  capping): no correlate found either, negligible count. Median-imputed
  for Random Forest only, same as the two groups above.

**Not yet decided (deferred, found while implementing this entry, not
addressed by D2 or D11):** `DAYS_EMPLOYED` itself (55,374 NaN, from D2's
sentinel fix) was flagged with `DAYS_EMPLOYED_ANOMALY` but never actually
given a fill value — that's fully explained by its own indicator
(100% correlated by construction) and belongs in the "impute for both"
group above, just wasn't decided when D2 was written. Also undecided:
`NAME_TYPE_SUITE` (1,292), `AMT_GOODS_PRICE` (278), `AMT_ANNUITY` (12),
`CNT_FAM_MEMBERS` (2), `DAYS_LAST_PHONE_CHANGE` (1) — all under 0.5%
missing, not investigated for a mechanism, likely safe for a simple
median/mode fill for both models given the negligible size either way.

Implemented in `scripts/clean_features.py`:
`add_housing_info_missing_indicator`, `impute_housing_block`,
`impute_occupation_type`, and `add_unexplained_missing_indicators` are
folded into `build_clean_features()` (the common, LightGBM-ready table);
`finalize_for_random_forest()` is the additional step the Random Forest
benchmark script (step 3) must call before training, imputing exactly the
"unexplained" group above.

## D13 — Closing the two gaps D12 deferred: DAYS_EMPLOYED and five minor columns (2026-09-25)

Extends D12.

Decided: `DAYS_EMPLOYED` (the NaNs `fix_days_employed` created from D2's
365243 sentinel) is imputed for both models — median of the genuine,
non-anomalous values (-1648.0). This is fully explained by construction:
`DAYS_EMPLOYED_ANOMALY` is 1 exactly where this is NaN, a stricter
condition than anything else in the "partially explained" D12 group. D2
added the indicator but never decided a fill value; this closes that.

Also decided: five columns under 0.5% missing, none investigated for a
mechanism (not worth the effort at this size) — imputed for both models:
`AMT_GOODS_PRICE` (278 missing, median 450000.0), `AMT_ANNUITY` (12,
median 24903.0), `CNT_FAM_MEMBERS` (2, median 2.0),
`DAYS_LAST_PHONE_CHANGE` (1, median -757.0), `NAME_TYPE_SUITE` (1292,
`"Missing"` category — who accompanied the applicant to the loan
application; e.g. "Unaccompanied", "Family", "Spouse, partner").

Implemented in `scripts/clean_features.py`: `impute_days_employed` and
`impute_minor_missing`, both folded into `build_clean_features()`. After
this, the only columns still carrying real NaN in the common
(LightGBM-ready) treated table are the D12 "unexplained" group
(`EXT_SOURCE_1/2/3`, `AMT_REQ_CREDIT_BUREAU_*`, social-circle residual) —
every other missing-value question from D2/D11/D12/D13 is now closed.

## D14 — Raw-baseline implementation and the three persisted datasets (2026-09-25)

Realizes D10's "raw" path in code — D10 defined it conceptually
("the minimum treatment needed for each model to run at all... nothing
else touched") but didn't specify a label-encoding scheme or a
missing-value convention for it.

Decided: one shared raw-baseline table, `scripts/raw_baseline.py`, used
for both models' "raw" cell in the D10 2x2 grid — not a per-model raw
table. Every object-dtype column is label-encoded with `pandas.factorize`
(each distinct string gets its own integer code; NaN becomes -1 — this is
also the raw path's missing-value handling for categoricals, deliberately
naive, no "Missing" category or indicator the way the treated path has).
Any numeric column still missing after that is filled with its own
median, generically over every numeric column rather than the
hand-picked, column-by-column reasoning in D11/D12/D13 — the point of
"raw" is a deliberately unclever baseline, not a second treated path.
`DAYS_EMPLOYED`'s 365243 sentinel is explicitly left untouched (confirmed:
55,374 rows retain it), same for every outlier, the FLAG_DOCUMENT_* block,
and the housing `_AVG`/`_MODE`/`_MEDI` redundancy — nothing from D11/D12/D13
applies here.

This completes the three datasets D10's 2x2 grid needs, all confirmed by
running the actual scripts against `application_train.csv`
(307,511 rows throughout):

- `data/processed/train_raw.csv` — 122 columns (unchanged from source),
  zero missing values, from `scripts/raw_baseline.py`.
- `data/processed/train_treated_lightgbm.csv` — 81 columns, real NaN
  preserved in exactly the D12 "unexplained" group, from
  `scripts/clean_features.py`'s `build_clean_features()`.
- `data/processed/train_treated_rf.csv` — same 81 columns, zero missing
  values, from `finalize_for_random_forest()` applied to the table above.

All three are gitignored (`data/processed/`, same reasoning as
`data/raw/`) — they're regenerated by running the two scripts, not
committed.

## D15 — Categorical-encoding gap found in the treated tables, and the fix (2026-09-25)

Found (not decided — a bug, caught before any training script ran):
`clean_features.py`'s `build_clean_features()` never actually encoded most
categorical columns — only `FLAG_OWN_CAR`/`FLAG_OWN_REALTY` (D11). Its own
docstring claimed the LightGBM table was "ready as-is," which was false:
both `train_treated_rf.csv` and `train_treated_lightgbm.csv` still carried
14 raw string columns (`NAME_CONTRACT_TYPE`, `CODE_GENDER`,
`NAME_TYPE_SUITE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`,
`NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`,
`WEEKDAY_APPR_PROCESS_START`, `ORGANIZATION_TYPE`, `FONDKAPREMONT_MODE`,
`HOUSETYPE_MODE`, `WALLSMATERIAL_MODE`, `EMERGENCYSTATE_MODE`). This would
have crashed `RandomForestClassifier.fit()` immediately (string input
rejected outright) and wasn't valid LightGBM sklearn-API input either
without at least a dtype cast. Caught by checking `select_dtypes` on the
actual output file rather than trusting "zero missing values" meant
"model-ready" — the same verify-against-real-output discipline this
project already leans on.

Decided: the fix differs by model, and by where it happens, because a CSV
round-trip cannot preserve pandas dtype:

- Random Forest: `encode_categoricals_for_random_forest()`
  (`pandas.factorize`, same convention as `raw_baseline.py`) is now
  called inside `finalize_for_random_forest()`, baked into
  `train_treated_rf.csv` on disk. Ordinal/label codes are a defensible
  choice specifically because RF (and LightGBM) are tree ensembles — a
  split on an arbitrary integer threshold partitions categories however
  they're numbered, with no assumption of order or even spacing. That
  would NOT be a defensible choice for a linear or distance-based model.
- LightGBM: `train_lightgbm.py`'s `cast_categoricals()` casts the same
  remaining object columns to pandas `"category"` dtype at load time, so
  LightGBM's sklearn wrapper auto-detects and splits on them natively.
  This cast is not baked into `train_treated_lightgbm.csv` — CSV storage
  is dtype-blind, so it has to happen wherever the CSV is loaded for
  training, not in `clean_features.py`.

Verified on the real 307,511-row `train_treated_rf.csv` after the fix:
zero object-dtype columns, zero missing values. `train_treated_rf.csv`
also shrank from ~155MB to ~114MB (small integer codes vs. strings).
`train_treated_lightgbm.csv` is unchanged by this fix (categorical columns
still stored as strings on disk, by design — see above).

## D16 — Held-out test-set discipline for repeated hyperparameter runs (2026-09-25)

The user wants to run `train_random_forest.py`/`train_lightgbm.py`
themselves, repeatedly, with different hyperparameters, to decide values
by comparing results. Evaluating the D6 held-out test split on every one
of those iterations would relocate the exact problem D4 already flagged
for the LightGBM tuning metric: the "held-out" set stops being held out
the moment it's used, iteration after iteration, to pick a winner.

Decided: both training scripts default to reporting only stratified
k-fold CV metrics (mean, std, 95% CI via the t-distribution across folds)
over the train+val pool — the held-out test split is carved out
up front (so train+val excludes it) but never evaluated by default. A
`--final` flag gates the one-time exception: fit once on the full
train+val pool with the chosen hyperparameters, evaluate once on the
held-out test set, and report ROC-AUC, PR-AUC, Brier score, and a
reliability-curve figure. The expectation is: iterate freely on CV
metrics, then run once with `--final` after deciding.

Also decided: `RANDOM_STATE = 42` and `TEST_SIZE = 0.15` are fixed
constants in both scripts, not CLI flags — this is what guarantees the
same rows are held out regardless of model, dataset (raw/treated), or
hyperparameters, since every D10/D14 table preserves the original
`application_train.csv` row order (confirmed: `SK_ID_CURR` sequence is
identical across `train_raw.csv`, `train_treated_rf.csv`, and
`train_treated_lightgbm.csv`). Without this, RF-vs-LightGBM or
raw-vs-treated comparisons in `results/model_runs.csv` would be comparing
different held-out populations, not the same test.

Implementation: `scripts/train_random_forest.py` (RF, D3's tuned values
as defaults) and `scripts/train_lightgbm.py` (LightGBM's own library
defaults — no prior tuned run exists for this project, unlike RF's D3),
both appending one row per run (hyperparameters, CV summary, and, under
`--final`, test metrics) to `results/model_runs.csv` for cross-run
comparison. Verified end-to-end against the real 307,511-row data, all
four combinations (RF/LightGBM x raw/treated), both with and without
`--final`.

## D17 — Train-vs-val diagnostic added to both training scripts (2026-09-25)

Found: the per-fold line each script printed (`fold 1/5: ROC-AUC=... PR-AUC=...`)
was the validation fold only — the model is fit on the other folds and
scored on data it never saw during that fit. There was no training-fold
score computed anywhere, so there was no way to see an overfitting gap
from the script's own output.

Decided: `cross_validate()` in both scripts now scores each fold's own
training data in addition to its validation data, prints both per fold,
and reports both summaries (mean +/- 95% CI) plus the train-minus-val gap
for ROC-AUC and PR-AUC. Logged to `results/model_runs.csv` as
`train_roc_auc_mean`/`train_roc_auc_ci95`/`train_pr_auc_mean`/`train_pr_auc_ci95`,
alongside the existing `cv_*` (validation) columns.

Confirmed on real data (raw dataset, 3-fold, 50 estimators/trees each):
Random Forest shows a large train/val gap (train ROC-AUC 0.893 vs. val
0.733, gap 0.160; train PR-AUC 0.510 vs. val 0.217, gap 0.293) —
consistent with RF's known behavior (deep, largely unpruned trees fit
their own bootstrap sample closely; bagging reduces variance at the
ensemble level, not by preventing each tree from overfitting). LightGBM
shows a much smaller gap under the same fold count and estimator count
(train ROC-AUC 0.782 vs. val 0.751, gap 0.030; train PR-AUC 0.284 vs. val
0.234, gap 0.050). A large RF train/val gap is not on its own evidence of
a problem — weak val performance *combined with* a large gap is the
actual signal to watch; for LightGBM (which fits residuals stagewise) a
growing gap is more directly attributable to too many boosting rounds or
too little regularization for the data.

Also fixed in the same change: `append_run_log()` in both scripts now
reindexes the combined DataFrame to the canonical `RUN_LOG_COLUMNS` order
before writing, rather than trusting whatever order `pd.concat` produces
against an existing file's (possibly older) header — otherwise a schema
addition like this one causes new columns to get tacked onto wherever
`pd.concat` happens to put them instead of their intended position.
Existing rows in `results/model_runs.csv` written before this change
simply get `NaN` in the four new train-column fields.

## D18 — Fold-level parallelism in cross_validate (2026-09-26)

CV was slow: 5 folds fit one after another, each a full RandomForestClassifier/
LGBMClassifier fit with n_jobs=-1 (uses all cores per fit, but only one fit
running at a time).

Decided: parallelize across folds instead of within a single fit. Each
fold is fully independent (fresh model, disjoint train/val indices), so
there's no correctness cost — `cross_validate()` now uses
`joblib.Parallel`/`delayed` to fit all folds concurrently (one process
per fold via `_fit_and_score_fold`), with each fold's own model forced to
`n_jobs=1`. Running each fold's forest at n_jobs=-1 *and* multiple folds
at once would oversubscribe the same cores twice over — nesting
parallelism like that typically nets out slower, not faster, from
context-switching and cache contention, not just "no extra benefit". A
new `--n-jobs` CLI flag (default -1, all cores) controls how many folds
run at once; it is not logged as a hyperparameter in `model_runs.csv`
since it only affects wall-clock time, never the result.

Trade-off made explicit in the console output: per-fold lines now print
together once every fold finishes, not one at a time as each completes —
acceptable, since the run's total wall-clock time is what mattered here,
not live progress.

Verified for correctness (not speed — this development sandbox has only
2 cores) on the real data: identical metrics before/after the change at
matching hyperparameters and fold count, confirming parallelizing changed
nothing about the computation itself. Real speedup is bounded by the
user's own core count and --n-folds; expect close to Nx on an N-core
machine with n_folds <= N cores, diminishing beyond that.

## D19 — Fix: `--max-features` as an integer count crashed (2026-09-26)

Bug: `--max-features 9` raised `sklearn.utils._param_validation.
InvalidParameterError` inside every CV fold. Root cause was in
`_max_features_value()`, the argparse type-caster: it unconditionally
did `float(value)` on any non-"none" string, so `"9"` became `9.0`.
sklearn's `RandomForestClassifier.max_features` accepts an int (an
absolute feature count) or a float *only* in (0.0, 1.0] (a fraction) —
`9.0` satisfies neither an int check nor that float range, so every
fold's `model.fit()` failed validation before any work was done.

Fix: `_max_features_value()` now tries `int(value)` before `float(value)`,
so whole-number strings parse as int (matching sklearn's own "int = raw
count" semantics) and only decimal strings ("0.5") fall through to float.
Added a regression test (`test_max_features_value_parses_int_as_int_not_
float`) asserting the int branch specifically, since the existing test
only covered `"0.5"` and `"sqrt"` and would not have caught this.

Not a --final run's problem specifically — this broke CV regardless of
--final, and would have failed identically without D18's parallelism;
it surfaced now only because this was the first time an integer
--max-features value was tried.

## D20 — RF hyperparameter selection criterion; treated-dataset values; raw-dataset grid search (2026-09-28)

Decided: RF hyperparameters are selected by maximizing mean CV PR-AUC
(this project's stated primary metric — SPEC.md Success criteria, D4),
never by minimizing the train/val gap. Rationale: RF trees are grown
deep and largely unpruned by design (D17) — bagging controls variance at
the ensemble level, not by keeping each tree from fitting its own
bootstrap sample closely — so a large train/val gap is an expected
property of RF, not on its own evidence against a hyperparameter choice.
Minimizing the gap directly would push toward shallower/more-restricted
trees than the val metric itself calls for, actively trading away val
performance to chase a number (the gap) that was never the target.

Treated dataset — final RF hyperparameters: `n_estimators=250,
max_depth=15, min_samples_split=25, min_samples_leaf=8,
max_features='sqrt', class_weight=None`. Verified against all 9 logged
treated-RF runs in `results/model_runs.csv`: this combination gives the
best mean CV PR-AUC (0.2283) *and* the best mean CV ROC-AUC (0.7441) of
the nine, and reproduced identically across two separate runs (same
seed, same result) — not a lucky single run. `class_weight="balanced"`
underperforms `None` here too (0.2172-0.2271 vs. 0.2271-0.2283 at
matching depth/leaf settings) — the same direction already found on the
raw dataset three exchanges ago, now corroborated on treated as well.

Raw dataset — grid search (`scripts/grid_search_random_forest.py`,
D20): `{max_depth: [8, 10, 15, None], min_samples_leaf: [4, 8, 12],
max_features: ["sqrt", 0.2, 0.3], n_estimators: [250]}`, 36
combinations, `min_samples_split` fixed at 25 and `class_weight` fixed
at `None` (both already settled, not swept). This is a documented,
scoped exception to train_random_forest.py's own stated design
("every hyperparameter is a CLI flag... not searched"): covering 36
combinations by hand doesn't scale, so this automates the repetition,
not the decision — every combination is still scored by CV only (D16,
held-out test never touched), every combination is logged individually
to `results/model_runs.csv` exactly as a manual run would be, and the
"winner" is a plain readout of mean CV PR-AUC across the logged rows,
not a search that decides anything by itself. Deliberately not
parallelized across combinations (only within each combination's own
CV folds, via --n-jobs) — the same oversubscription reasoning D18
already applied to folds vs. per-fit threads.

## D21 — Calibration added to `--final`; quantile-binned reliability curves; log loss (2026-09-28)

SPEC.md's Success criteria already require calibration (reliability
curve + Brier score) "before any threshold or pricing claim... means
anything" — this was previously reported only for the raw, uncalibrated
RF. `--final` now also fits `CalibratedClassifierCV(estimator=<fresh,
unfit RF>, method=..., cv=<StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)>)` on the same 261k train+val pool, for `"isotonic"`,
`"sigmoid"`, or both (`--calibration`, default `both`) — a fixed
`StratifiedKFold` object rather than a bare `cv=5` int, so the internal
5-way split is reproducible like everything else in this pipeline, not
just "5 folds, whichever split sklearn happens to draw". The base
estimator passed in is unfit, not the model already fit above — an
already-fit model would let calibration see predictions the model made
on data it was trained on, biasing the reliability curve optimistically.

Every variant (uncalibrated + each calibration method) reports ROC-AUC,
PR-AUC, Brier score, and now log loss on the held-out test set, and gets
one row in `results/model_runs.csv` (new `calibration` column: `none` /
`isotonic` / `sigmoid`; new `test_log_loss` column) sharing one
reliability-curve figure for direct comparison.

Reliability curves now use quantile (equal-count) bins, not
`calibration_curve`'s default equal-width bins — the real `--final` run
three exchanges ago showed 5 of 10 equal-width bins empty (RF's proba
output compresses toward the low end). Equal-count bins guarantee every
bin gets roughly the same share of test points regardless of that skew;
per-bin counts are computed and printed alongside the curve, not
asserted. Own helper (`_quantile_reliability_curve`), since
`calibration_curve` has no `strategy="quantile"` per-bin-count output —
ties in predicted probability can still collapse requested bins into
fewer effective ones, which is why the count array's own length, not a
fixed 10, is what gets reported.

Caveat, stated plainly rather than glossed over: `CalibratedClassifierCV`
with an integer/splitter `cv` (not `cv="prefit"`) fits its own internal
clones per fold and averages them — the isotonic/sigmoid variants are
therefore small 5-model ensembles of forests each trained on 4/5 of the
pool, not the same single full-pool model with a monotonic rescaling on
top. A monotonic rescaling alone cannot change ROC-AUC/PR-AUC (both are
rank statistics), yet the very first correctness run showed ROC-AUC and
PR-AUC both improve under calibration (e.g. 0.7275->0.7391-0.7393
ROC-AUC on a reduced-settings run) — that improvement is the 5-model
averaging effect, not evidence that calibration itself improved ranking.
Brier score and log loss are what isolate calibration quality here;
ROC-AUC/PR-AUC differences between variants should not be read as "the
calibrated model discriminates better."

Left open, not yet done: `train_lightgbm.py` has no calibration step of
its own. Its `RUN_LOG_COLUMNS` picked up the same `calibration` /
`test_log_loss` columns in this change only to avoid a real bug —
both scripts append to the same `results/model_runs.csv` via a
reindex-to-own-schema `append_run_log`, so without this LightGBM's next
run would have silently dropped both new columns (RF's already-logged
calibration rows included) on its next append. D4 already calls
LightGBM "the model meant to be the main result" — calibrating it too
is needed before step 5's decision layer can use whichever model ends
up final, not yet decided which that is.

## D22 — LightGBM calibration decided by CV, not by comparing on the test set (2026-09-28)

RF's calibration design (D21) fits CalibratedClassifierCV(cv=5) once
under --final and reports uncalibrated/isotonic/sigmoid side by side on
the held-out test set. That is itself a form of test-set-driven model
selection if the three test-set rows are then used to decide which
variant "wins" — the exact thing D16 exists to prevent, just moved from
hyperparameters to calibration method. Flagged here, not fixed in D21's
code: RF's --final run for this was already in progress when this was
raised; revisiting D21's design is deferred to when RF is next
discussed, not decided unilaterally mid-run.

LightGBM does it differently: `--check-calibration` runs during ordinary
CV (never --final, never the test set) and fits an isotonic-calibrated
variant per fold via `CalibratedClassifierCV(estimator=FrozenEstimator(
model), method="isotonic")` — sklearn's current replacement for the
removed `cv="prefit"` (this project's installed scikit-learn is 1.9.1;
the pyproject.toml floor is bumped 1.5->1.6, the version that introduced
FrozenEstimator, so this is a real dependency requirement now, not just
documentation — already satisfied by the locked version, no action
needed). FrozenEstimator wraps the already-fit fold model without
refitting it, so calibration is fit once, on a holdout the boosted trees
never saw — never on the fold's own validation split, which would let
the tree count/leaf structure implicitly "know" the data used to judge
it. Isotonic Brier/log loss are compared out-of-fold against the raw
model across all folds; the verdict ("APPLY"/"DO NOT APPLY") uses the
same magnitude the user set for comparing RF vs. LightGBM CV means
(>0.005 in mean, or improvement in every fold) applied here to Brier
instead. Sigmoid is not offered for LightGBM — it already optimizes log
loss directly, so a flexible nonparametric check (isotonic) is the more
informative one to run; RF got both because its miscalibration pattern
(systematic, monotonic, tail-heavy) is exactly what a 2-parameter Platt
correction targets.

`--final`'s own `--calibration {none,isotonic}` flag does not decide
anything itself — it applies whichever the CV verdict already decided,
via the same FrozenEstimator + dedicated holdout mechanism (shared with
early stopping's eval set, D23, when both are on): ONE coherent model,
never CalibratedClassifierCV's default k-fold-ensemble behavior. Checked
empirically: isotonic's test ROC-AUC differed from the uncalibrated
model by <0.001 in a light correctness run (0.7559 vs. 0.7552) — the
small residual gap is isotonic's own tie-breaking on repeated values,
not model averaging, unlike RF's ~0.01-0.02 shift from its 5-model
ensemble (D21).

## D23 — Early stopping (2026-09-28)

`--early-stopping-rounds` (opt-in, default off — unset preserves every
prior run's exact fixed-n_estimators behavior). When set, `--n-estimators`
becomes a ceiling; the actual tree count (`best_iteration_`/`n_estimators_`)
is chosen by monitoring `average_precision` (D4's own primary metric,
not the LightGBM default of binary_logloss) on a holdout carved from
that fit's *own training portion only* — never the fold's validation
split during CV, never the held-out test split under --final. Early-
stopping on the data a fit is later scored on would pick the tree count
using the very data judging it, optimistically biasing that score — the
most common leakage bug in LightGBM CV pipelines, and the reason this
holdout is carved fresh each time rather than reusing an existing split.
Per-fold tree counts are printed (mean/min/max) since they can legitimately
differ fold to fold; not logged to model_runs.csv — an emergent training
diagnostic to eyeball, not a metric to compare runs by, and adding it
there would mean touching RF's schema too for a column RF has no
equivalent of (same reasoning that kept --n-jobs, D18, out of the log).

## D24 — Feature importance: gain, not split count (2026-09-28)

`--final` now saves `results/feature_importance_lightgbm_<dataset>_
<timestamp>.csv` (feature, gain, gain_pct, sorted descending) and a
top-20 bar chart, both from `booster_.feature_importance(importance_type=
"gain")` — total loss reduction from a feature's splits, not the
sklearn-API default (`split`, a raw split-frequency count that says
nothing about how much any split actually helped). Read from the
underlying fitted booster directly, so it's identical whether or not
`--calibration isotonic` is applied on top — calibration re-maps
probabilities, it never refits the trees.

## D25 — LightGBM hyperparameter-surface fixes and clarifications (2026-09-28)

Fixed: `--subsample-freq` was never exposed as a CLI flag. LightGBM's
own default (`subsample_freq=0`) disables bagging outright regardless of
`--subsample`'s value — any `--subsample < 1.0` was a silent no-op with
no warning. Confirmed via `results/model_runs.csv` this never actually
mattered yet: zero LightGBM runs exist in the log to date (RF/grid
search came first, per the user's own stated order). Now exposed, with
a runtime warning printed whenever `--subsample < 1.0` and
`--subsample-freq == 0` together, so the same mistake can't recur silently.

Confirmed already correct, no change: `--is-unbalance` defaults to
False/off (LightGBM's own default) — the calibration-neutral choice,
matching RF's `class_weight=None` default. No `scale_pos_weight` flag
exists at all, so that pitfall was never exposed in the first place.

Categorical encoding, corrected: neither model one-hot encodes.
RF's treated table label-encodes every categorical column to plain
integers (D15, `finalize_for_random_forest()`), which RF then treats as
ordinal/continuous — a real, already-accepted limitation of that table,
not a new finding. LightGBM's treated table instead casts the same
columns to pandas `category` dtype (D15) and lets LightGBM's sklearn
wrapper auto-detect and split on them natively — no `categorical_feature=`
passed explicitly, relying on that auto-detection. This asymmetry (label-
encoded-as-ordinal vs. true native categorical splits) was a deliberate
D15 decision, not an oversight, and stays as-is — "keep one-hot for a
fair comparison" doesn't apply since one-hot was never used on either
side.

## D26 — LightGBM tuning plateaued on treated; configuration frozen (2026-09-28)

Added: `val_roc_auc_folds`/`val_pr_auc_folds` columns to `results/
model_runs.csv` (both scripts, and `grid_search_random_forest.py`) — the
raw per-fold validation scores as a JSON list, not just their mean/CI.
The mean/CI pair already logged can't support a paired fold-by-fold
comparison (e.g. RF vs. LightGBM at matching `n_folds`, same fold
assignments per D22); the raw values now can. Existing rows get `NaN`
here, same reindex-on-append pattern as every prior schema addition.

Evidence — one-hyperparameter-at-a-time sweep, LightGBM/treated, 5-fold
CV, `n_estimators=2000` (ceiling), `learning_rate=0.05`,
`early_stopping_rounds=50`, `max_depth=-1`, `reg_alpha=reg_lambda=0.0`
(unswept — see caveat below), all from `results/model_runs.csv`:

| num_leaves | min_child_samples | colsample_bytree | CV ROC-AUC | CV PR-AUC |
|---|---|---|---|---|
| 31 (baseline) | 20 | 1.0 | 0.7550 ± 0.0059 | 0.2390 ± 0.0109 |
| 15 | 20 | 1.0 | 0.7549 ± 0.0055 | 0.2394 ± 0.0099 |
| 63 | 20 | 1.0 | 0.7536 ± 0.0057 | 0.2375 ± 0.0104 |
| 127 | 20 | 1.0 | 0.7522 ± 0.0063 | 0.2354 ± 0.0094 |
| 31 | 50 | 1.0 | 0.7550 ± 0.0061 | 0.2386 ± 0.0100 |
| 31 | 100 | 1.0 | 0.7550 ± 0.0058 | 0.2393 ± 0.0095 |
| 31 | 20 | 0.5 | 0.7558 ± 0.0064 | 0.2404 ± 0.0105 |
| 31 | 20 | 0.7 | 0.7554 ± 0.0063 | 0.2402 ± 0.0092 |

Every PR-AUC in this table sits within 0.0040 of the baseline (0.2390) —
below the >0.005 threshold already set for "a real difference" (D22),
and every pair of CIs overlaps heavily (each ±0.009 to ±0.011 on its
own). `num_leaves` beyond 31 clearly hurts (63, 127 both drop
monotonically); `num_leaves=15` and `min_child_samples` in {50, 100} are
statistically indistinguishable from the baseline, not improvements.
`colsample_bytree=0.5` has the single best point estimate on both
metrics (PR-AUC 0.2404, ROC-AUC 0.7558) but not by a margin that clears
the threshold either — this is a genuine plateau, not a hidden win.

Decided: freeze at `num_leaves=31, min_child_samples=20,
colsample_bytree=0.5` (all else as swept above) — `colsample_bytree=0.5`
over the library default 1.0 specifically because, with no statistically
real difference between them, training on half the columns per tree is
strictly cheaper (faster fits, same expected score) — a legitimate
tie-breaker between empirically-equivalent options, not a claim that
0.5 is a discovered improvement.

Caveat, not glossed over: `reg_alpha`/`reg_lambda` (tuning-order step 4,
explicitly "if at all") were never swept — every row above has both at
0.0. "Plateaued" describes `num_leaves`/`min_child_samples`/
`colsample_bytree` only; L1/L2 regularization remains untested, by
choice, not by having been checked and found flat too.

## D27 — Early-stopping eval-set audit; calibration-check sign convention labeled (2026-09-28)

**Early stopping, audited by reading the code path, not by reproducing a
run.** Reproducing `best_iteration=379` bit-for-bit only proves the run is
deterministic — it says nothing about what data the eval set actually was.
Checked directly: in `evaluate_holdout()`, `_fit_with_optional_early_stopping`
is called as

```python
X_train_r = X_train.reset_index(drop=True)
y_train_r = y_train.reset_index(drop=True)
all_idx = np.arange(len(X_train_r))
...
fit_idx, holdout_idx = _fit_with_optional_early_stopping(
    model, X_train_r, y_train_r, all_idx, args, ...
)
```

`X_train`/`y_train` here are the train+val pool (D6's `split_holdout`
output), never `X_test`. Inside the helper, `eval_X=X.iloc[holdout_idx]`,
`eval_y=y.iloc[holdout_idx]` come from `train_test_split(idx,
test_size=HOLDOUT_FRACTION, stratify=y.iloc[idx], random_state=RANDOM_STATE)`
applied to `idx=all_idx` — i.e. an internal, stratified 15% slice of the
pool itself. Grepping the file for `X_test` confirms it appears only at the
two `model.predict_proba(X_test)` call sites (evaluation, after fitting is
complete) — never assigned to `eval_X`/`eval_y`, never passed to `.fit()`.

Verified with the real numbers (`treated` dataset, `n=307,511`):
test split (D6, untouched by training) = 46,127 rows; pool = 261,384 rows;
internal early-stopping/calibration holdout = 39,208 rows (15.0% of the
pool), fit portion = 222,176 rows (85.0%); positive rate 0.0807 in the
fit slice, the holdout slice, and the pool overall (identical to 4
decimals) — the stratification is holding.

Decided: this is the "internal split of the pool" case, not "the test
set" — no leak, and the fallback (`n_estimators≈275` fixed, no early
stopping) does **not** apply. No code or config change needed here; this
entry is the record the earlier informal reproduction should have been.

**Calibration-check sign convention: the earlier "sign inconsistency" was
never a real inconsistency, and my "noise" explanation for it was wrong.**
0/5 folds improved and a negative mean delta cannot conflict — that's
arithmetic, not something noise produces or erases. `folds_improved_brier
= (brier_diff > 0).sum()`, so "0/5 improved" means every one of the 5
per-fold values is ≤ 0, which forces their mean to be ≤ 0 as well; a
positive mean paired with 0/5 improved would be the impossible
combination, not the reverse. Checked directly against the real per-fold
arrays from both `--check-calibration` runs already in the log:

- `colsample_bytree=1.0`: Brier deltas `[-0.00009451, -0.00007653,
  -0.00013910, -0.00013081, -0.00010536]` — all five negative.
- `colsample_bytree=0.5`: Brier deltas `[-0.00010585, -0.00011091,
  -0.00014799, -0.00010087, -0.00007633]` — all five negative.

Both fully consistent with "0/5 improved, negative mean" — there was
nothing to reconcile once `delta = raw − calibrated` (positive =
calibration helped) is stated plainly. Fixed by adding that sentence as a
printed line in `train_lightgbm.py`'s `cross_validate()` calibration-check
output itself (previously only a code comment, invisible at runtime), right
after the summary line, spelling out the "0/5 improved forces mean ≤ 0"
arithmetic so this doesn't need re-deriving from scratch next time.

## D28 — Protected attributes disclosed as unhandled; SPEC.md updated (2026-09-28)

`SPEC.md` said nothing, until now, about protected attributes. Checked
what's actually in the data rather than assuming: `CODE_GENDER`,
`DAYS_BIRTH` (age), and `NAME_FAMILY_STATUS` are present, untouched, in
`train_raw.csv` and both treated tables — nothing in the "Feature
treatment plan" (D11-D16) ever dropped them. Both RF and LightGBM use them
like any other feature. Checked the effect size, not just presence:
in the logged LightGBM `--final` run's gain-based feature importance
(D24), `DAYS_BIRTH` ranks 6th of 79 features (gain=10,424, 3.9% of total)
— a real, material predictor, not a marginal one. `CODE_GENDER` ranks
17th of 79 (gain=2,686); `NAME_FAMILY_STATUS` 39th of 79 (gain=1,077).

Decided: state this plainly as a scope boundary rather than leave it
implicit. Added to `SPEC.md`:

- A new "Non-goals" bullet: no fairness/disparate-impact analysis, and no
  removal of protected attributes or their proxies — they're used exactly
  like any other feature, evidenced by `DAYS_BIRTH`'s rank above.
- A new paragraph in "Credit decision & pricing" (the section that already
  draws the line between what the model outputs and what a real lending
  decision requires), naming the three columns, their gain ranks, and the
  US framing the project was built against: ECOA/Regulation B bars sex,
  marital status, and age as a basis for a credit decision, and other
  jurisdictions have their own equivalents — a real deployment would need
  a fairness audit and very likely feature changes before this model could
  decide on real applicants.

Why here and not a new section: "Credit decision & pricing" already
carries the "the model does not make the business decision" boundary;
this is the same boundary, extended to a limitation the project doesn't
claim to have solved. No code changes — this is a documentation-only
change, per the Spec change policy (substantive Non-goals change).

## D29 — Leak fix: median/percentile statistics fit on train rows only (2026-10-08)

A reviewer caught a real, if small, leak: `impute_days_employed` and every
other statistic `clean_features.py`/`raw_baseline.py` fit from the data —
every median fill, both 99.9th-percentile outlier caps — was computed on
the full 307,511-row table, before any train/test split existed. The 15%
of rows that later becomes the D6 held-out test split was baked into the
very medians and caps applied to it.

Fixed by computing the D6 train+val partition once, before any column is
dropped (`fit_train_index`, same `RANDOM_STATE=42`/`TEST_SIZE=0.15`
stratified split as `train_random_forest.py`/`train_lightgbm.py`'s own
`split_holdout`), and threading that index through every function that
fits a statistic: `impute_days_employed`, `impute_housing_block`,
`impute_minor_missing`, `cap_income_outliers`, `cap_social_circle_outliers`,
`finalize_for_random_forest`'s unexplained-group median fill, and
`raw_baseline.py`'s `median_impute_numeric`. Each now computes its
statistic from `df.loc[train_index]` only; passing no index (every
existing unit test, which uses tiny synthetic frames with no real
train/test concept) reproduces the old, pre-fix behavior exactly, so
`main()` is the only caller required to pass the real index.

The whole fix hinges on `fit_train_index`'s split landing on the *same*
rows `split_holdout` carves out later from the finished, column-dropped
table — checked directly against the real 307,511-row data rather than
assumed: `set`/`list` equality both `True`, 46,127 test rows either way.
Now a permanent regression test in both test files
(`test_fit_train_index_matches_downstream_split_holdout_partition`).

Also fixed the identical structural bug in `cap_income_outliers` /
`cap_social_circle_outliers` (the 99.9th-percentile caps) — not named by
the reviewer, but the same full-table-before-split pattern.
`cap_at_percentile` already took a `threshold` override for reuse, but
nothing had ever actually passed it a train-only value.

**Verified the fix changes real output, against the real data, not just
code structure.** `DAYS_EMPLOYED`'s imputed median moved from -1648.0
(full 307,511-row table) to -1647.0 (261,384-row train-only) — small, as
the reviewer predicted ("a median over about 250k values"), but real and
non-zero. On the held-out test set, comparing the last pre-fix `--final`
run (2026-09-28) against the first post-fix one (2026-10-08), same
hyperparameters, same `RANDOM_STATE`:

| Model | Metric | Before (leaky) | After (fixed) | Δ |
|---|---|---|---|---|
| RF, uncalibrated | ROC-AUC | 0.74856 | 0.74855 | -0.00001 |
| RF, uncalibrated | PR-AUC | 0.23612 | 0.23635 | +0.00023 |
| LightGBM, uncalibrated | ROC-AUC | 0.76114 | 0.76021 | -0.00093 |
| LightGBM, uncalibrated | PR-AUC | 0.25171 | 0.24798 | -0.00373 |

Every shift is inside the bootstrap noise floor established by D30 below.
LightGBM moved roughly 15x more than RF on PR-AUC — plausibly because
`DAYS_EMPLOYED` is LightGBM's 5th-ranked feature by gain (4.3% of total,
D24) while RF's bagging smooths over a one-unit median shift more; not
independently confirmed beyond this, and not worth chasing further given
the size of the effect.

## D30 — Bootstrap CI on held-out test-set metrics (2026-10-08)

Success criteria (SPEC.md) already requires cross-validated ROC-AUC with
a CI, not a point estimate. The one-time held-out test evaluation under
`--final` had no such thing — a single number from ~46,127 rows read as
more precise than it is. Extended the same discipline to it.

Added `_bootstrap_ci` (duplicated identically in both training scripts,
matching this project's self-contained-script convention): a
nonparametric (percentile) bootstrap, 1000 resamples of `(y_true, y_prob)`
with replacement at the test set's own size, recomputing ROC-AUC, PR-AUC,
Brier, and log loss on each resample, reporting the 2.5th/97.5th
percentile as the 95% interval. A resample landing on a single class
(undefined ROC-AUC/PR-AUC) is redrawn, not counted — astronomically
unlikely at an ~8% base rate and ~46k rows, but guarded rather than
assumed. The point estimate is still the one real D6/D16 evaluation
(unchanged); the bootstrap characterizes resampling variance around it,
never a second "real" estimate. `model_runs.csv` gets four new columns
(`test_{roc_auc,pr_auc,brier,log_loss}_ci95`, JSON `[lo, hi]`), always
`None` for `grid_search_random_forest.py` (D16: never touches the test
set).

Current intervals (95%, 1000 resamples, post-D29 data):

| Model | Calib. | ROC-AUC | PR-AUC |
|---|---|---|---|
| RF | none | 0.7486 [0.7400, 0.7566] | 0.2364 [0.2236, 0.2497] |
| RF | isotonic | 0.7488 [0.7406, 0.7567] | 0.2355 [0.2229, 0.2489] |
| RF | sigmoid | 0.7490 [0.7408, 0.7568] | 0.2358 [0.2231, 0.2491] |
| LightGBM | none | 0.7602 [0.7517, 0.7678] | 0.2480 [0.2342, 0.2608] |

RF's and LightGBM's PR-AUC intervals overlap substantially
([0.2342, 0.2497]). The ~5% relative gap between their point estimates
(the primary metric per D4) is not clearly decisive from two
independently-computed CIs alone — the statistically correct comparison
is a paired bootstrap on the per-resample difference (same resample
indices applied to both models' predictions, testing the sign of the
difference), which has not been run. Treat "LightGBM beats RF on this
data" as a point-estimate observation, not a statistically established
one, until that's done.

## D31 — Step 5 decision-layer design: cost-ratio threshold sweep + expected-loss pricing (2026-10-15)

Design-only entry — this records the decided logic for the step 5
decision layer (SPEC.md "Credit decision & pricing") before
`scripts/decision_layer.py` exists; the real numbers, once the script is
built and run, get their own follow-up entry (this project's convention
per D27: a later finding gets a new entry, not an edit to this one).

**Approve/decline — Bayes-cost threshold, not a 0.5 cutoff.** For an
applicant with predicted P(default)=p, the expected cost of approving is
p·C_FN (C_FN = cost of a missed bad payer — lost principal); the expected
cost of declining is (1-p)·C_FP (C_FP = opportunity cost of a lost good
payer — foregone margin). Approve iff p·C_FN < (1-p)·C_FP, which solves to

  p < C_FP / (C_FP + C_FN) =: t*

Only the ratio of the two costs matters, not their scale, so reparametrize
as k = C_FN/C_FP ("a missed default costs k times a missed good loan"):
t* = 1/(1+k). Since C_FN is typically >> C_FP in lending, t* sits well
below 0.5 — at this dataset's ~8% base rate, plausible k (3-20) puts t*
in roughly the 0.05-0.25 range.

Per SPEC.md ("the model does not make the business decision"), k is a
stated assumption, never fit from data — sweep k over {3, 5, 10, 20},
compute t*(k) in closed form for each, and report a cost curve: approval
rate, bad-rate-among-approved, and expected cost per applicant, at each
t*(k), against the frozen held-out test predictions.

**Leakage guard, stated explicitly because it's easy to get backwards:**
computing t* from a stated k and then applying it to test predictions is
not a new look at the test set — t* never depends on y_test. Scanning a
grid of raw thresholds and keeping whichever minimizes *empirical* test
cost would be a second, test-set-informed "fit," the same leakage D29
fixed in preprocessing, just relocated to the decision layer. The
implementation must compute every t* before ever touching y_test, not
search for one against it.

**Pricing ("how much") — expected-loss pricing, solved for rate, not just
stated as a value.** Expected portfolio value per unit lent at rate r is
(1-p)(1+r) - p·LGD (full repayment+interest if good, LGD-fraction loss if
bad, recoveries folded into LGD). Setting this to a target return 1+m and
solving for r:

  (1-p)(1+r) - p·LGD = 1 + m
  r* = [m + p·(1+LGD)] / (1-p)

Limits check out: p→0 gives r*→m (zero risk, charge just the target
margin); p→1 gives r*→∞ (no finite rate prices out a near-certain
default — decline, don't reprice). This is the same expected-value logic
as the threshold above, not a second mechanism: declining is the limiting
case where r* becomes infeasible (negative margin, or above any rate a
borrower would accept).

**LGD and m are stated, not fit** — there is no recovery/collections data
in this dataset (application-level table only, D1), so LGD cannot be
estimated, only assumed. Home Credit's loans are largely unsecured
consumer credit (no collateral to recover against), where a 50-85% LGD
range is typical in the literature; sweep LGD over {0.5, 0.6, 0.7, 0.8,
0.9, 1.0} (every 0.1, not just the three original anchor points) — this
sweep is pure arithmetic over an already-computed p (D31), not a
retrain, so there is no cost reason to coarsen it, and a 0.1 LGD step is
cheap insurance against a reader fixating on one value instead of the
shape of the dependence. m (target margin) is a pure
business choice, not an empirically uncertain quantity the way LGD is —
expose it as a CLI flag, default 0.05, not swept.

**Validation against real data, not just a formula in a vacuum**: this
dataset is already-disbursed loans, so `AMT_ANNUITY/AMT_CREDIT` is each
applicant's actual historical implied periodic rate (D7's proxy — not a
clean APR, term length isn't separable from rate in this ratio). Compare
r* against that real column for test-set applicants: does the bank's
existing pricing already move with predicted p the way r* implies? A null
result (little correlation) is a legitimate finding, not something to
suppress in favor of a cleaner-looking synthetic result.

**Model and predictions used — decided by the numbers already in hand, not
re-opened as a new choice.** LightGBM's uncalibrated test-set probabilities:
Brier=0.0676, the lowest of all four evaluated variants (RF
none/isotonic/sigmoid = 0.0688/0.0682/0.0686), and PR-AUC=0.2480, also the
highest. Calibration is load-bearing for this step (SPEC.md), so Brier is
the deciding criterion, not PR-AUC alone — both happen to agree here, so
no tension to adjudicate.

**No new model-persistence mechanism** — neither training script saves a
fitted model object or raw test-set predictions under `--final`.
`decision_layer.py` imports `load_dataset`/`split_holdout`/`build_model`
from `train_lightgbm.py` (same pattern `grid_search_random_forest.py`
already uses against `train_random_forest.py`) and refits directly —
deterministic given the fixed hyperparameters and `RANDOM_STATE`, so this
is the same model, not a second copy that can drift from it.
`split_holdout` doesn't reset the index, so `df.loc[X_test.index,
["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY"]]` recovers the real test-set
applicants' loan terms for the pricing-validation step above with no join
logic needed.

**Deferred, on purpose, per explicit instruction**: how the D28 caveat
(protected attributes feed p with no fairness audit, so every threshold/
price here inherits that) gets physically attached to whatever the
decision layer outputs — a footnote on a table, a printed line, wherever
— is a publication/presentation question, decided once real results
exist, not before.

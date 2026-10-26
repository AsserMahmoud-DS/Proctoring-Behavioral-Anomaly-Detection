# Behavioral Anomaly Detection: Final Experiment Plan

> Historical document: This is the scientific protocol, retained with its original development checkboxes and working rules. Earlier module paths and Git-ignored statements describe the state when written. The [final report](../../reports/experiment1/README.md) records actual execution and limitations. Engineering phases A–E are separate [integration history](../../plans/experiment_1_phases.md).

## Status and working rules

- [x] Study Phase 0: persist the agreed phased plan (documentation only).
- [x] Study Phase 1: original-training audit and specification freeze complete.
- [x] Study Phase 2: readiness and reusable integration complete and verified.
- [ ] Study Phase 3: model controls and isolated study runner.
- [ ] Study Phase 4: seed-42 validation study and selection lock.
- [ ] Study Phase 5: locked-finalist seed sensitivity.
- [ ] Study Phase 6: final test reporting and interpretation.

These are study-local phases, separate from the completed LSTM/ensemble phases
in `plans/issues.md`. Study files are isolated under the Git-ignored `study/`
folder. Phases 0 and 1 are complete; the user accepted the audited specification
with the angle-wrapping correction and documented remaining legacy definitions.
No model-selection experiments have been run under this protocol.

- Work on one phase at a time. Report changes, verification, and unresolved
  questions, then wait for the user's instruction before advancing.
- Never commit or push unless the user explicitly requests it. Suggested commit
  boundaries below organize future changes; they are not permission to commit.
- Later phases remain provisional until reviewed. Any specification correction
  must be documented before it is used; never silently expand a grid after
  seeing validation winners or revise an approach in response to test results.
- Local study implementation belongs in `study/`; production fixes remain in
  `src/`. Notebooks are thin research/report callers.
  Edit notebook `.py` twins only, then Jupytext-sync. The user executes notebooks.
- Programmatic implementation verification uses synthetic data. Original-training
  EDA is allowed in Phase 1; real validation metrics begin in Phase 4 and test
  metrics only in Phase 6.

## Study aim and evaluation contract

Determine whether IF, OCSVM, LSTM-AE, and their ensemble can learn useful
behavioral anomaly signals from this dataset. Describe how complete preprocessing
recipes, bounded configurations, augmentation, and a limited AE depth comparison
affect results. This is a proof of concept, not exhaustive optimization or a
production-readiness assessment.

### Fixed evaluation choices

- Retain the current session split: split seed 42, normal-validation fraction
  0.2, mixed-validation fraction 0.3. Freeze the actual ordered file assignments,
  not just the seed. Document student overlap; do not claim unseen-student
  generalization.
- Keep parent windows at 50 events with stride 25 and cheating-label fraction
  0.5. AE micro-windows remain 10 events with stride 5: nine timesteps per parent.
- Use combined normal + mixed validation for configuration selection and threshold
  calibration. AE early stopping uses unaugmented normal validation only.
- Select on validation ROC-AUC, retaining first-candidate ties deterministically.
  Report PR-AUC as average precision, with anomaly prevalence as its baseline.
- Tune each selected model's threshold on validation by maximizing recall subject
  to precision >= 0.5, with the existing F1 fallback. Record fallback use; the
  validation precision floor is not a test or deployment guarantee.
- Classify scores at the threshold as anomalous: `score >= threshold` everywhere.
- Keep augmentation training-only: two copies per original parent window,
  coordinate-noise sigma drawn from [2, 5] pixels, generation seed 42, and existing
  coordinate clipping bounds. The sigma range is not a bound on realized noise.
- Use model seed 42 throughout the main search. Seeds 123 and 2026 are only
  predefined repeats of locked IF/AE finalists; never choose the best seed.

### What comparisons mean

Run every configuration with augmentation off and on. Hold original data, split,
feature list, fitted preprocessing, gamma candidates, and model seed fixed across
the pair. Only the detector's training data gains synthetic copies.

Report two distinct comparisons:

1. **Matched configuration:** compare the same recipe and hyperparameters with
   augmentation off/on; this measures the observed augmentation intervention.
2. **Independently tuned condition winners:** compare each condition's best
   validation-selected configuration under the same candidate budget. These are
   complete tuned approaches, not an isolated augmentation effect.

Augmentation also increases AE optimizer updates under a fixed epoch budget.
Record epochs and update counts; do not attribute a gain exclusively to synthetic
sample diversity. Synthetic windows and overlapping micro-windows do not create
additional independent students or sessions.

## Phased execution

### Study Phase 0 — Persist this plan

Deliverables: replace the initial plan with this document and align the research
study checklist in `plans/issues.md`. Preserve completed issue history. Do not
implement models, run experiments, edit notebooks, or make commits in this phase.

### Study Phase 1 — Original-training audit and specification freeze

Audit findings: [Phase 1 report](phase_01_findings.md).
Outputs are saved under `study/artifacts/phase_01_audit/`; 662 parent rows,
5,958 micro rows, 15 training files, and 33 encoded columns. Both IF sample
candidates remain distinct. Numerical recipes and the bounded gamma range pass
the training-only audit. Review legacy extraction caveats before freezing.
The user-approved angle-wrapping fix now covers direction-change counts,
sum-of-angles, and sharp-angle counts with unchanged thresholds. Audit outputs
have been regenerated; singleton-click behavior is unchanged. Generated study
code, notebooks, tests, plans, and artifacts are consolidated under Git-ignored `study/`.

**Purpose:** verify feature meanings and the draft search before inspecting
selection results.

- Analyze original normal-training data once, covering parent and micro-feature
  representations within the same audit. Distinguish this from the current EDA
  path, which enables augmentation by default.
- Check definitions, signs, units, zero mass, constants, nonfinite values, and
  effects of the specified transforms. Resolve L/SL choices from extraction
  definitions, not labeled validation performance.
- Verify the 25 source columns below, direction codes, ordered output columns,
  and encoded dimension. Retain this fixed feature list across all conditions;
  do not redo feature selection after augmentation or for each recipe.
- Audit the actual normal-training row count so IF's 128/256 sample candidates
  remain meaningful rather than silently collapsing to the same effective size.
- Check the variance-aware OCSVM gamma candidates against transformed
  original-training geometry. Record the recipe-specific gamma centres and
  candidate distance/kernel summaries without tuning against validation.
- Record actual split assignments, parent-window identities, geometry, units,
  training constants, and configuration budgets. Review any corrections before
  freezing the specification.

**No duplicate augmented-data EDA:** there is no second histogram/feature-selection
study, no re-selection of transforms, and no preprocessing refit on augmented
data. Automated augmented-input checks belong to Phase 2.

**Done when:** the feature/recipe specification and grids are reviewed, numerical
questions are resolved, and original-training audit outputs are saved. Stop here
for user review before implementing preprocessing.

Suggested change boundary: feature audit/reporting and specification documentation.

### Study Phase 2 — Data and preprocessing readiness

Implementation details: [Phase 2 readiness](phase_02_readiness.md). Shared raw-noise
representations, frozen fitted recipes, persistence/manifest guards, and API
threshold equality are implemented and synthetically verified. The user successfully
executed the readiness notebook, and its real-data snapshot passes independent
manifest/alignment/preprocessing checks. Reusable components and regression tests
are now tracked; study-only orchestration stays ignored. Phase 2 is complete;
stop for user review before advancing to Phase 3.

**Purpose:** implement controlled paired representations and frozen transforms.

- Keep original training data available separately from synthetic rows. Prepare
  original, augmented, validation, and test representations with stable parent
  identities and ordering. For flat/AE comparability, verify that their synthetic
  representations come from the same realized noisy parent windows; equal seeds
  alone do not prove this if iteration order differs.
- Do not trust existing unversioned feature/sequence caches. Build study data
  fresh from raw inputs and reuse it within the study; any persisted reuse must
  match the saved input/split/geometry/augmentation manifest. Isolate study
  outputs from production caches and model artifacts.
- Implement the recipes and block scaling specified below. Fit imputation and
  learned transforms on original normal training only, then freeze them for
  augmentation, early stopping, validation, and test.
- Fit flat preprocessing on original parent rows and AE preprocessing on
  flattened original micro-features. Restore the AE sequence shape after
  encoding; do not reuse parent scalers or scale each timestep separately.
- Preserve constants and rare-event columns with the stated bypass/scaling
  conventions. Persist fitted state, gamma centres, and ordered output names.
- Check schemas, category/domain validity, finite transformed values, and
  sample/label alignment automatically. Do not perform a second augmented EDA.
- Align threshold equality handling between tuning, evaluation, detector
  prediction, and the API.

**Verification:** synthetic tests for exact feature operations, signed values,
constants, fixed categories, original-training-only fitting, unchanged fitted
statistics across augmentation conditions, cache isolation, sample identities,
serialization, and threshold equality.

**Done when:** both training conditions use identical frozen preprocessing and
the same held-out samples, with all relevant programmatic tests passing.

Suggested change boundaries: data/cache/alignment correctness; then preprocessing
recipes with their tests. Avoid unrelated API or infrastructure refactors.

### Study Phase 3 — Model controls and isolated study runner

**Purpose:** support the bounded grids without automatic test evaluation.

- Expose IF's planned max_features control and retain the selected fitted
  detector, rather than reconstructing parameters from a DataFrame or refitting
  the winner.
- Supply numeric recipe-specific gamma candidates to OCSVM and record convergence
  status. Failed fits are reported, not silently replaced by another recipe.
- Implement the fixed AE controls below: separate latent/built-in dropout,
  weight decay, gradient clipping, and relative early-stopping improvement
  tolerance. Restore the best normal-validation checkpoint and record epochs,
  best epoch, and optimizer updates.
- Provide separate validation-study and locked-final-reporting entrypoints.
  Do not repeatedly call the current train-and-test pipeline for recipe searches:
  it evaluates test data immediately and writes production artifacts.
- Save configuration identifiers, parameters, model/recipe/augmentation/seed,
  validation scores and metrics, fitted states, and failure/training diagnostics.
  Keep selected study artifacts separate from the deployable winner.
- Retain lazy torch loading and keep the AE research-only. Changing production
  model eligibility is outside this study.

**Verification:** synthetic grid, tie, convergence/failure, AE-control, artifact
reload, and end-to-end tests. Selection must not receive test scores/metrics.

**Done when:** a small synthetic study can select and persist paired recipe
winners without running final test reporting or overwriting production artifacts.

Suggested change boundaries: model controls; then study runner and its isolation
tests. Review these edits before real experiment execution.

### Study Phase 4 — Seed-42 validation study and selection lock

**Purpose:** run the predeclared paired search and lock approaches for reporting.

- Execute all configurations at seed 42 in both augmentation states: 204 main
  model fits. The user runs the thin Jupytext notebook.
- Record each candidate's validation ROC-AUC, PR-AUC, baseline, parameters,
  runtime, effective settings, failures, and training diagnostics. Report AE
  early-stopping loss/epoch-cap status to expose possible undertraining.
- Produce matched-config augmentation comparisons and per-condition tuned
  winners. Show grids/top configurations without claiming exhaustive search.
- Report the matched AE depth comparison at batch 32 and learning rate 1e-3,
  holding recipe, hidden size, augmentation state, and seed fixed.
- Lock one winner per recipe-condition cell: 4 IF + 8 OCSVM + 6 AE = 18.
  A cell with no valid candidate must be reported and resolved explicitly before
  a complete final reporting manifest can be approved.
- Within each augmentation condition, select the best IF and OCSVM across their
  recipe winners. Reuse those fitted detectors to build that condition's ensemble.
  Freeze score scaling on unaugmented normal-validation scores, then tune weights
  on combined validation ROC-AUC only.
- Tune thresholds for the 18 recipe winners and two ensembles. Save the selected
  artifacts, thresholds, weight/scaler state, and selection metadata.
- Freeze the final reporting manifest before reading test metrics. It also names
  the overall IF/AE finalists used in Phase 5; no re-selection from test results.

**Done when:** validation reports and the 20-approach reporting manifest are saved
and reviewed. Stop before seed repeats or final test evaluation.

### Study Phase 5 — Locked-finalist seed sensitivity

**Purpose:** check initialization sensitivity without enlarging the search.

- For each augmentation condition, repeat its overall selected IF and AE
  configuration at seeds 123 and 2026: four finalists x two repeats = eight fits.
- Keep recipe, hyperparameters, split, generated augmentation, preprocessing,
  and data order fixed. Each repeated detector may receive its own
  validation-calibrated threshold and AE early-stopping checkpoint.
- Report all three seeds and descriptive mean/std; never select the best seed,
  replace the seed-42 primary artifact, or retune configurations from these runs.
- Do not repeat deterministic OCSVM fits or add ensemble seed repeats. These
  checks address model-training sensitivity, not independent-session robustness
  or protection from validation-selection bias.
- Predefine the repeated artifacts for final reporting; do not inspect their test
  performance yet.

**Done when:** eight repeat fits, artifacts, and validation sensitivity summaries
are saved and reviewed. The planned total is now 212 model fits.

### Study Phase 6 — Final test reporting and interpretation

**Purpose:** evaluate locked approaches and produce the study report.

- Evaluate the 20 primary seed-42 approaches and eight predefined repeat artifacts
  in one final reporting stage. Do not evaluate every hyperparameter candidate on
  test or choose a new winner from test metrics.
- Report ROC-AUC, PR-AUC (average precision), prevalence baseline, precision,
  recall, F1, false-positive rate, threshold, fallback status, and TP/TN/FP/FN.
- Produce ROC/PR curves, score distributions, and seed-42 confusion matrices.
  Group panels by model family/recipe/augmentation for readability; do not average
  confusion-matrix counts across seeds.
- Show the seed-42 selected recipe/parameters for every cell, matched augmentation
  differences, independently tuned winners, the bounded AE depth comparison on
  validation, and a separate finalist seed-sensitivity table.
- Summarize original/augmented training sizes, original session counts, test/val
  prevalence, runtime, epochs, and optimizer updates. Augmented session_name
  suffixes must not be counted as new independent sessions.
- Use `study/artifacts/` for manifests, audit/validation/test tables,
  figures, and fitted study artifacts. Use unique study/configuration identifiers
  so separate approaches cannot overwrite each other's reports.
- Reuse existing metric/plot helpers, including confusion matrices. The user
  executes the reporting notebook; edit its `.py` twin and Jupytext-sync it.
- Explain limitations: few independent sessions, window overlap, student overlap,
  validation-selection optimism, unequal family search budgets, and augmentation's
  additional AE updates. If test results previously informed development, call
  the final evaluation exploratory rather than untouched.

**Done when:** locked-model tables, figures, artifacts, and conclusions agree.
No deployment promotion or subsequent test-driven experiment cycle is implied.

Suggested change boundary: final reporting/notebook integration and study narrative.

## Feature specification — exact operations

Notation:

- **I**: identity; no nonlinear transform.
- **L**: log1p(x), for verified nonnegative values.
- **SL**: sign(x) × log1p(abs(x)); preserves sign and zero.
- **YJ**: training-fitted Yeo–Johnson.
- **Q**: training-fitted quantile-to-normal.
- **E**: fixed categorical encoding.

| Feature | IF-log / SVM-log / AE-log | SVM-YJ / AE-YJ | SVM-quantile | Brief rationale |
|---|---|---|---|---|
| elapsed_time | L | YJ | Q | Compress long-duration tail |
| copy_events | I | I | I | Small rare counts; preserve occurrence |
| paste_events | I | I | I | Mostly binary; compression adds little |
| mouse_path_length | L | YJ | Q | Broad positive range and zero mass |
| mouse_straightness | I | I | I | Meaningful bounded fraction |
| mouse_velocity_mean | L | YJ | Q | Strong right tail |
| mouse_acceleration_mean | L | YJ | Q | Nonnegative magnitude; extreme time-denominator sensitivity |
| mouse_jerk_mean | SL | YJ | Q | Signed derivative of acceleration magnitude; extreme tail |
| mouse_direction_changes | I | I | I | Bounded, reasonably spread count |
| mouse_click_count | I | I | I | Modest count range |
| mouse_idle_time_ratio | I | I | I | Meaningful bounded fraction |
| mouse_angular_velocity_mean | SL | YJ | Q | Signed extremes |
| mouse_angular_velocity_std | L | YJ | Q | Nonnegative heavy-tailed variability |
| mouse_angular_velocity_min | SL | YJ | Q | Strong negative tail |
| mouse_curvature_mean | L | YJ | Q | Nonnegative absolute-curvature definition |
| mouse_curvature_std | L | YJ | Q | Heavy-tailed variability |
| mouse_curvature_min | L | YJ | Q | Nonnegative, zero-heavy curvature minimum |
| mouse_direction_class | E | E | E | Category codes are not ordinary magnitudes |
| mouse_sum_of_angles | I | I | I | Broad distribution; no strong initial need |
| mouse_largest_deviation | L | YJ | Q | Right-skewed spatial deviation |
| mouse_sharp_angles | I | I | I | Bounded count with meaningful endpoint masses |
| keyboard_typing_rate | L | YJ | Q | Strong positive tail; check denominator |
| keyboard_burst_count | L | L | L | Zero-heavy, skewed count |
| keyboard_pause_count | L | L | L | Zero-heavy, skewed count |
| window_switch_events | L | L | L | Preserve occurrence; compress repeated events |

Phase 1 resolves signs from extraction definitions and original training, not
validation performance: acceleration and curvature are nonnegative magnitudes;
jerk is signed and requires SL. See the Phase 1 report for units and legacy
feature-semantic caveats; extraction changes remain subject to review.

Raw/base recipes use I for every numerical feature, including burst, pause, and window-switch counts. Direction encoding is fixed across every recipe. These baselines compare complete preprocessing choices; they do not isolate individual feature effects.

Keep units fixed across experiments: log1p is unit dependent. Boundedness does not preclude useful transformations, but here bounded ratios and modest counts are deliberately left nonlinear-untransformed to limit the search.

### Scaling after the operations

| Feature block | IF | OCSVM / AE |
|---|---|---|
| Numerical features except copy/paste/window switches | None | StandardScaler fitted on original training representation |
| copy_events, paste_events, window_switch_events | None | None; retain recipe-specific raw/log units |
| Direction indicators | None | None |

Rare-event counts are excluded from variance scaling to avoid making a single occurrence enormous when training variance is tiny. This is a fixed weighting choice, not a claim of equal feature influence.

IF rationale: nonlinear compression changes random split geometry. Affine scaling is generally redundant, so RobustScaler is omitted. Raw IF remains necessary because compression can weaken informative extreme-value separation.

OCSVM rationale: scaling controls unequal units in RBF distances; nonlinear transforms control extreme spacing. Gaussian marginal distributions are not required.

AE rationale: scaling supports optimization and reduces unequal contributions to reconstruction MSE. Default nn.LSTM uses internal sigmoid/tanh operations, but inputs do not have to lie within [-1,1]. A linear reconstruction head can reconstruct values outside that range.

### Direction handling

Use fixed one-hot codes 0–8 across models. Code 0 means zero endpoint displacement
or insufficient mouse events, not necessarily no movement. Codes 1–8 are
45-degree sectors starting at +x in screen coordinates. Do not scale indicators.
This avoids ordinal-code distances but does not encode angular adjacency. Keep
this a fixed representation choice rather than a tuning axis.

Report 25 source features and the verified encoded dimension d=33. Output order
is the 24 numerical source columns in source order, then nine direction indicators
in code order 0–8. AE input and output dimensions must match d. Mean encoded-column
MSE means the encoding also affects reconstruction weighting; document that convention.

## Preprocessing specification — recipes

| Model | Recipe | Specification |
|---|---|---|
| IF | IF-raw | Numerical identity + fixed encoding; no scaler |
| IF | IF-log | Selective log/signed log + fixed encoding; no scaler |
| OCSVM | SVM-base | Numerical identity + encoding + block-specific scaling |
| OCSVM | SVM-log | Selective log/signed log + encoding + block-specific scaling |
| OCSVM | SVM-YJ | YJ on continuous block; fixed count treatment; encoding; block-specific scaling |
| OCSVM | SVM-quantile | Q on continuous block; fixed count treatment; encoding; block-specific scaling |
| AE | AE-base | Same operations as SVM-base, fitted on micro-chunks |
| AE | AE-log | Same operations as SVM-log, fitted on micro-chunks |
| AE | AE-YJ | Same operations as SVM-YJ, fitted on micro-chunks |

For YJ, use PowerTransformer(method="yeo-johnson", standardize=False), then the specified separate scaling. Fit each feature's parameter from original training only.

For Q, use output_distribution="normal", n_quantiles=min(100, n_original_training_rows), subsample=None, and a fixed random_state. Do not tune these parameters. Beyond-range observations map to fitted distribution boundaries, so extreme magnitude differences can be lost; tied zeros are not made informative by the transformation.

### Separate fitting for the AE

Fit flat preprocessing on original parent rows. Fit AE preprocessing on original normal-training sequences reshaped from (N,9,25) to (N×9,25), then restore (N,9,d) after encoding. Apply the same fitted feature transformations at every timestep. Do not scale per sequence/timestep or reuse flat aggregate scalers.

Repeated micro-chunks are correlated and do not multiply the number of independent sessions. If augmentation is used, keep its realized dataset and generation seed fixed across model-training seeds. Document whether augmentation acts on raw events or extracted features; do not imply equivalence unless implemented and verified.

### Numerical safeguards

- Constant training features bypass YJ/Q. Use a safe StandardScaler-equivalent convention: training mean and scale 1 for a constant numerical column.
- Retain the fixed source feature list and record constants; do not silently delete rare-event signals. IF cannot learn useful splits from a feature constant in training.
- Inspect near-constant transformed columns for excessive standardization amplification.
- Reject/report failed or nonfinite transformations; do not substitute a new recipe after examining labeled validation results.
- Save fitted means/scales, YJ lambdas, quantile landmarks, category order, units, and ordered output names.

## Model specification — bounded configurations

These are the agreed draft ranges for the Phase 1 audit. They are not claims of
universal adequacy. Freeze them before selection; more candidates increase the
opportunity to fit validation noise, even when training itself is regularized.

### Isolation Forest

| Parameter | Values |
|---|---|
| Preprocessing | IF-raw, IF-log |
| max_samples | 128, 256 |
| max_features | 0.75, 1.0 |
| n_estimators | 100, 300, 500 |
| bootstrap | Fixed false |
| contamination | Fixed "auto" |
| Main search seed | 42 |
| Finalist repeat seeds | 123, 2026; locked overall winner per augmentation state only |

24 configurations per augmentation condition: 2 recipes x 2 sample sizes x
2 feature fractions x 3 tree counts. Both conditions require 48 main fits.
More trees primarily stabilize the aggregated score rather than deepen each tree;
the range supports a bounded tree-count comparison, not guaranteed improvement.
Do not tune contamination when selecting score ranking and calibrating an external
threshold: it controls the default threshold rather than the score ordering.

### One-Class SVM

| Parameter | Values |
|---|---|
| Preprocessing | SVM-base, SVM-log, SVM-YJ, SVM-quantile |
| kernel | Fixed RBF |
| nu | 0.01, 0.05, 0.10 |
| gamma | 0.25 * gamma_0, gamma_0, 4 * gamma_0 |
| tol | Fixed 1e-3 |
| max_iter | -1; require successful convergence |

For each recipe, compute `gamma_0 = 1 / (d * X_original_transformed.var(ddof=0))`
on the finite, encoded, transformed original normal-training matrix. Here variance
is over all matrix entries, matching sklearn's gamma="scale" convention, and d
is the actual encoded dimension. Reject zero/nonfinite variance rather than
inventing a fallback. Freeze the three numeric candidates for augmentation off/on;
passing gamma="scale" directly on augmented training would recompute the centre.

36 configurations per augmentation condition; 72 main fits in total. OCSVM is
deterministic with these fixed inputs. nu affects the fitted normal boundary and
is not expected cheating prevalence. No nu/kernel/solver expansion is planned.

### LSTM-AE: main search plus depth comparison

| Block | Recipes | Hidden size | Batch size | Learning rate | Encoder/decoder layers |
|---|---|---|---|---|---|
| Main search | AE-base, AE-log, AE-YJ | 16, 32 | 16, 32, 64 | 3e-4, 1e-3 | 1 / 1 |
| Depth comparison | AE-base, AE-log, AE-YJ | 16, 32 | 32 | 1e-3 | 2 / 2 |

The main search has 36 configurations; depth adds 6, with matching one-layer
configurations present. Total: 42 configurations per augmentation condition and
84 main fits across both conditions at seed 42. This is a bounded depth check,
not exhaustive tuning of two-layer models. Do not add hidden-size, dropout, or
weight-decay grids after seeing validation winners.

| Fixed setting | Value |
|---|---|
| Architecture | Encoder final-layer hidden state as code; repeated code into decoder; linear reconstruction head |
| Optimizer | Adam |
| Weight decay | 1e-4 |
| Explicit latent dropout | 0.1 |
| Built-in nn.LSTM dropout | 0 for both one- and two-layer variants |
| Bidirectional / projections | False / none |
| batch_first | True |
| Batch handling | Shuffle training samples; keep final incomplete batch (drop_last=False) |
| Maximum epochs | 100 |
| Patience | 10 |
| Significant improvement | Relative normal-validation loss decrease of at least 0.001 |
| Gradient clipping | Global norm 1.0 |
| Loss | Mean reconstruction MSE over timesteps and encoded columns |
| Anomaly score | Same mean MSE; higher is more anomalous |
| Main search seed | 42 |
| Finalist repeat seeds | 123, 2026; locked overall winner per augmentation state only |

Batch 16 tests optimization with more frequent/noisier updates; it does not guarantee less overfitting. Two layers test added capacity and can increase overfitting risk. Keeping built-in dropout at zero changes depth alone while explicit latent dropout remains fixed. Built-in LSTM dropout does not operate between layers in a one-layer model.

### Ensemble specification

- One ensemble per augmentation condition, using that condition's overall
  validation-selected IF/OCSVM recipe/configuration winners at seed 42.
- IF weights: 0.0, 0.3, 0.5, 0.7, 1.0; OCSVM weight is 1 minus the IF weight.
- Reuse fitted detectors. Fit the persisted two-column RobustScaler on their
  unaugmented normal-validation scores; retain finite-score and nondegenerate-IQR
  guards. Apply the frozen scaler to combined validation and final test scores.
- Tune weight on validation ROC-AUC, then calibrate the selected ensemble's
  threshold with the shared validation policy. Test is reporting-only.
- Endpoint weights retain single-detector ranking baselines; no dense weight
  search, joint detector-pair search, or ensemble seed-repeat study is planned.

### Model-fit budget

| Model | Configurations per augmentation state | Main fits (off + on, seed 42) | Finalist repeats | Total fits |
|---|---:|---:|---:|---:|
| IF | 24 | 48 | 4 | 52 |
| OCSVM | 36 | 72 | 0 | 72 |
| AE | 42 | 84 | 4 | 88 |
| Total | 102 | 204 | 8 | 212 |

Ensemble calibration and weight scoring reuse existing fitted detectors and are
not additional base-model fits. Preprocessing fitting and failed attempts should
be logged separately. Different grid sizes reflect different tunable structures;
this is not an equal-compute comparison or exhaustive family optimization.

### Primary final-report count: why 18 plus 2

| Model | Recipes | Augmentation states | Validation-selected recipe winners |
|---|---:|---:|---:|
| IF | 2 | 2 | 4 |
| OCSVM | 4 | 2 | 8 |
| AE | 3 | 2 | 6 |
| Recipe winners | 9 | 2 | 18 |
| Ensemble | One selected detector pair per state | 2 | 2 |
| Primary reported approaches | | | 20 |

Each recipe-condition cell contributes one best configuration, not every grid
candidate. The eight finalist repeat artifacts appear separately in sensitivity
reporting; they are not new recipes or additional primary approaches. The planned
final test table therefore contains 20 primary rows plus eight sensitivity rows.

## Explicit non-goals and sources

No repeated split study, nested cross-validation, exhaustive optimization,
per-feature transformation ablation, duplicate augmented-data EDA, extra
early-stopping-source factor, or experiment-tracking platform. No automatic
deployment promotion. These can be future studies if the proof of concept warrants
them; they are not required to finish this protocol.

Reference checks:

- [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html): tree/sample controls, aggregation, and contamination offset.
- [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html): gamma="scale" convention, nu, and solver controls.
- [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) and [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html): fitted transformations and boundary behavior.
- [Cawley and Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html): model-selection overfitting on finite validation data; bounded grids and seed repeats do not remove this risk.

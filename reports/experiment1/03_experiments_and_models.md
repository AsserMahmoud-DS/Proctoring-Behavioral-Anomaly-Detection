# Experiment 1 — Protocol, preprocessing, models, and complete results

Proof-of-concept study that trains normal-only anomaly detectors on the frozen
paired dataset and selects configurations on validation only. This file holds
the method and the **complete result set**; the four final variants and the
conclusions are in `04_selected_models_and_conclusions.md`. Machine-readable
copies of every table live in `tables/`.

## 1. Objective and research questions

Can Isolation Forest (IF), One-Class SVM (OCSVM), an LSTM autoencoder (AE), and
an IF+OCSVM ensemble learn a useful behavioral anomaly signal from this small
session dataset, and how do complete preprocessing recipes, coordinate
augmentation, and a bounded AE depth comparison affect the outcome?

1. Which complete preprocessing recipe ranks anomalies best per model family?
2. Does training-only coordinate augmentation help (matched-configuration and
   independently tuned comparisons)?
3. Does a two-layer AE improve over its matched one-layer configuration?
4. How sensitive are the locked IF/AE finalists to the initialization seed?

Scope: proof of concept on annotated session files — not exhaustive
optimization, not deployment readiness, not misconduct verification.

## 2. Study phases

| Phase | Work |
|---|---|
| 0 | Persist the frozen protocol (splits, windows, recipes, grids, seeds, metrics, thresholds) |
| 1 | Original-training audit: signs, units, constants, tails, timing, transform/kernel geometry, angle-wrapping fix |
| 2 | Readiness: fresh paired parent/micro extraction, original-only fitted preprocessing, augmentation identity checks |
| 3 | Model controls + isolated validation-only runner (frozen-preprocessor injection, solver/training diagnostics) |
| 4 | Seed-42 main search: 204 fits, 18 recipe winners + 2 ensembles, frozen reporting manifest |
| 5 | Seed sensitivity: 8 repeats of the per-condition IF/AE finalists (seeds 123/2026) |
| 6 | Single test evaluation of the frozen 20 primary + 8 sensitivity artifacts, tables and figures |

## 3. Evaluation contract

- **Split:** frozen audit split (`02_data_sessions_and_windows.md` §3); no
  re-derivation during the study.
- **Preprocessing fit:** original normal training only (662 parent rows for flat
  recipes; 5,958 flattened micro rows for AE recipes); frozen for augmented and
  held-out inputs.
- **Seeds:** 42 for the main search; 123 and 2026 only as predefined repeats of
  locked IF/AE finalists (no best-seed choice).
- **Selection:** validation ROC-AUC within each family/recipe/augmentation cell;
  ties keep the first declared candidate. Average precision (reported as
  `pr_auc`) is displayed but is not the selection metric.
- **Threshold:** sweep ascending unique validation scores; predict anomalous at
  `score >= threshold`; maximize recall subject to validation precision ≥ 0.5;
  if the floor is never met, fall back to the F1-maximizing threshold.
- **Test:** evaluated once in Phase 6 by the gated reporting stage, at the
  frozen thresholds; test results never change configurations, weights,
  thresholds, or winners.
- **Ensemble:** per augmentation condition, reuse the fitted best IF and OCSVM
  recipe winners; freeze a RobustScaler on their scores on the 197 unaugmented
  pure-normal validation windows; choose the IF weight on combined-validation
  ROC-AUC.

## 4. Feature preprocessing

Pipeline order per recipe: schema validation and canonical ordering → ±∞→NaN
with original-training **median** imputation → nonlinear operations → (OCSVM/AE
only) block StandardScaler → direction one-hot → finite-output check. All fitted
state comes from original training and is reused unchanged for synthetic,
validation, and test rows.

| Feature block | Base | Selective log | Yeo–Johnson | Quantile-to-normal | SVM/AE scaling |
|---|---|---|---|---|---|
| 10 nonnegative continuous | identity | `log(1+x)` | fitted YJ | fitted empirical quantile | StandardScaler |
| 3 signed continuous | identity | `sign(x)·log(1+abs(x))` | fitted YJ | fitted empirical quantile | StandardScaler |
| keyboard burst/pause, window_switch | identity | `log(1+x)` | `log(1+x)` | `log(1+x)` | burst/pause yes; window_switch no |
| copy/paste counts | identity | identity | identity | identity | no (rare-count weighting) |
| direction_changes, click, sharp_angles | identity | identity | identity | identity | yes |
| straightness, idle_ratio | identity | identity | identity | identity | yes |
| sum_of_angles | identity | identity | identity | identity | yes |
| direction_class | 9 indicators | 9 indicators | 9 indicators | 9 indicators | no |

Why these choices:

- **Base** is the control: source magnitudes with only imputation, encoding, and
  the model-required scaling.
- **Selective log** compresses the large finite tails seen in the audit (parent
  acceleration up to ≈ 8.27e6, jerk ≈ 8.42e9) while preserving zeros; the signed
  variant keeps the sign of jerk/angular features instead of invalidating them.
  Ordinary counts and bounded ratios are deliberately not transformed.
- **Yeo–Johnson** fits one power parameter per feature on original training
  (`PowerTransformer(method="yeo-johnson", standardize=False)`), accepting both
  signs; scaling is applied separately. Mapping:
  `x ≥ 0, λ ≠ 0: ((x+1)^λ − 1)/λ`; `x ≥ 0, λ = 0: log(x+1)`;
  `x < 0, λ ≠ 2: −((1−x)^(2−λ) − 1)/(2−λ)`; `x < 0, λ = 2: −log(1−x)`.
- **Quantile-to-normal** maps through the training empirical distribution
  (`n_quantiles=min(100, n_rows)`, `output_distribution="normal"`,
  `subsample=None`, seed 42). Beyond-range values map to fitted boundaries.
- **Scaling:** StandardScaler is applied to 21 numerical columns for OCSVM/AE
  (RBF distances and reconstruction MSE are magnitude-sensitive); IF recipes are
  intentionally unscaled (random split trees do not need variance
  normalization; compression still changes split geometry). `copy_events`,
  `paste_events`, and `window_switch_events` are excluded from variance scaling
  to avoid amplifying a single rare occurrence.
- **Constants:** retained, never silently dropped; YJ/quantile bypass a constant
  continuous column and a scaled constant uses scale 1. No source column is
  constant in this run.
- **Direction encoding:** the integer code becomes nine indicator columns (no
  ordinal distances), producing 24 numerical + 9 indicator = 33 outputs.

Nine compared recipes:

| Family | Recipe | Nonlinear operations | Scaling | Representation |
|---|---|---|---|---|
| IF | IF-raw | base | none | parent |
| IF | IF-log | selective log | none | parent |
| OCSVM | SVM-base | base | selected-block StandardScaler | parent |
| OCSVM | SVM-log | selective log | standard scaler | parent |
| OCSVM | SVM-YJ | YJ continuous + log-count block | standard scaler | parent |
| OCSVM | SVM-quantile | quantile continuous + log-count block | standard scaler | parent |
| AE | AE-base | base | standard scaler | micro sequence |
| AE | AE-log | selective log | standard scaler | micro sequence |
| AE | AE-YJ | YJ continuous + log-count block | standard scaler | micro sequence |

## 5. Models, grids, and fixed controls

**Isolation Forest** — 2 recipes × `max_samples ∈ {128, 256}` ×
`max_features ∈ {0.75, 1.0}` × `n_estimators ∈ {100, 300, 500}` = 24
fits/condition; `contamination="auto"`; seed 42. Scores are the negated
`decision_function` (higher = more anomalous). Contamination affects the
built-in offset, not the ROC ranking used for selection.

**One-Class SVM** — 4 recipes × `nu ∈ {0.01, 0.05, 0.10}` ×
`gamma ∈ {0.25γ₀, γ₀, 4γ₀}` = 36 fits/condition; RBF kernel; `tol=1e-3`;
`max_iter=-1`. The frozen center is derived from the encoded
original-training matrix:

`γ₀ = 1 / (d · Var(X_original_transformed, ddof=0))`, `d = 33`, variance over
all matrix entries.

| Recipe | γ₀ | numeric candidates (0.25×, 1×, 4×) |
|---|---|---|
| SVM-base | 0.04359031420070035 | 0.010897578550175088, 0.04359031420070035, 0.1743612568028014 |
| SVM-log | 0.0452880618000325 | 0.011322015450008126, 0.0452880618000325, 0.18115224720013 |
| SVM-YJ | 0.0452880618000325 | 0.011322015450008126, 0.0452880618000325, 0.18115224720013 |
| SVM-quantile | 0.0452880618000325 | 0.011322015450008126, 0.0452880618000325, 0.18115224720013 |

Convergence is recorded per candidate: all 72 OCSVM fits converged
(`fit_status_ == 0`; solver iterations 39–1,238). Gamma is frozen numerically so
augmentation cannot silently recompute `gamma="scale"`.

**LSTM autoencoder** — encoder final hidden state is the latent code (latent
dropout 0.1), repeated over 9 timesteps, decoded by an LSTM and a linear head.
Training loss and anomaly score are the mean squared reconstruction error over
all 9 timesteps × 33 encoded columns. Main grid: 3 recipes × hidden ∈ {16, 32} ×
batch ∈ {16, 32, 64} × lr ∈ {3e-4, 1e-3} = 36 (1/1 layers); bounded depth: same
3 recipes × hidden ∈ {16, 32} at batch 32 / lr 1e-3 = 6 (2/2 layers); 42
fits/condition. Fixed controls:

| Control | Setting |
|---|---|
| Optimizer | Adam, weight decay 1e-4 |
| Dropout | explicit latent 0.1; built-in LSTM dropout 0.0 at both depths |
| Epochs / patience | 100 / 10 |
| Significant improvement | validation loss < best × (1 − 1e-3) |
| Gradient clipping | global norm 1.0 |
| Batching | shuffle; final incomplete batch kept |
| Early-stopping input | original, unaugmented pure-normal validation sequences |
| Main seed | 42 (repeats 123/2026 for finalists only) |

**Ensemble** — per condition, reuse the fitted best IF and OCSVM winners
(no refit). Normalize each base score on the pure-normal validation windows:
`z_j(x) = (score_j(x) − median_j) / IQR_j`, then
`score(x) = w·z_IF(x) + (1−w)·z_OCSVM(x)` with
`w ∈ {0, 0.3, 0.5, 0.7, 1.0}` selected by combined-validation ROC-AUC. Both
conditions select `w = 0.7`.

**Budget** — 102 fits/condition → 204 main fits; 8 finalist repeats; 212 total
model fits. Ensemble weight scoring and threshold tuning reuse existing fits.

## 6. Search mechanics

- One shared ordered candidate sweep (declaration order) implements the stable
  first-candidate tie-break, records ROC-AUC/AP/baseline/runtime per candidate,
  and records failures instead of substituting results.
- Cells with no valid candidate are recorded in the manifest
  (`missing_cells`); this run has none (204 successful fits, 0 failures).
- Candidates receive the frozen original-fitted preprocessor (deep copy), so
  augmentation changes only the rows the estimator sees; an invariance test
  confirms the fitted preprocessing state is unchanged across a full sweep.
- The search layer never reads test partitions (AST-guarded); test scoring lives
  in the reporting stage, gated by the frozen manifest.

## 7. Complete results

### 7.1 Primary approaches (20)

All rows are seed-42 validation winners or validation-selected ensembles.
Validation columns are from the search; test columns are evaluated once at the
frozen thresholds. `TP/FP/TN/FN` are test counts (test: 205 positives, 161
negatives).

| Family | Recipe / aug | Val ROC | Val AP | Test ROC | Test AP | Prec | Rec | F1 | FPR | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|---|---|
| if | IF-log / off | 0.7618 | 0.5595 | 0.6881 | 0.7390 | 0.7791 | 0.6537 | 0.7109 | 0.2360 | 134/38/123/71 |
| if | IF-raw / off | 0.7519 | 0.5721 | 0.7051 | 0.7566 | 0.7765 | 0.6439 | 0.7040 | 0.2360 | 132/38/123/73 |
| if | IF-log / on | 0.7818 | 0.5728 | 0.7004 | 0.7370 | 0.7711 | 0.6244 | 0.6900 | 0.2360 | 128/38/123/77 |
| if | IF-raw / on | 0.7847 | 0.6273 | 0.7138 | 0.7500 | 0.7391 | 0.6634 | 0.6992 | 0.2981 | 136/48/113/69 |
| ocsvm | SVM-base / off | 0.7361 | 0.4511 | 0.7205 | 0.7403 | 0.8583 | 0.5024 | 0.6338 | 0.1056 | 103/17/144/102 |
| ocsvm | SVM-log / off | 0.6549 | 0.3358 | 0.6653 | 0.7001 | 0.8293 | 0.1659 | 0.2764 | 0.0435 | 34/7/154/171 |
| ocsvm | SVM-quantile / off | 0.6433 | 0.3452 | 0.6468 | 0.7181 | 0.8269 | 0.2098 | 0.3346 | 0.0559 | 43/9/152/162 |
| ocsvm | SVM-YJ / off | 0.6379 | 0.2855 | 0.6811 | 0.7173 | 0.6784 | 0.6585 | 0.6683 | 0.3975 | 135/64/97/70 |
| ocsvm | SVM-base / on | 0.7664 | 0.5134 | 0.7323 | 0.7461 | 0.7798 | 0.6390 | 0.7024 | 0.2298 | 131/37/124/74 |
| ocsvm | SVM-log / on | 0.6919 | 0.4083 | 0.6873 | 0.7164 | 0.8028 | 0.2780 | 0.4130 | 0.0870 | 57/14/147/148 |
| ocsvm | SVM-quantile / on | 0.6822 | 0.3956 | 0.6781 | 0.7475 | 0.8630 | 0.3073 | 0.4532 | 0.0621 | 63/10/151/142 |
| ocsvm | SVM-YJ / on | 0.6870 | 0.3365 | 0.7026 | 0.7313 | 0.6976 | 0.6976 | 0.6976 | 0.3851 | 143/62/99/62 |
| ae | AE-base / off | 0.7420 | 0.5408 | 0.6368 | 0.6909 | 0.7313 | 0.4780 | 0.5782 | 0.2236 | 98/36/125/107 |
| ae | AE-log / off | 0.7317 | 0.5472 | 0.6211 | 0.6456 | 0.6846 | 0.4341 | 0.5313 | 0.2547 | 89/41/120/116 |
| ae | AE-YJ / off | 0.6554 | 0.3025 | 0.5804 | 0.6123 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/161/205 |
| ae | AE-base / on | 0.7419 | 0.5598 | 0.6414 | 0.6956 | 0.7279 | 0.4829 | 0.5806 | 0.2298 | 99/37/124/106 |
| ae | AE-log / on | 0.7445 | 0.5717 | 0.6314 | 0.6607 | 0.7099 | 0.4537 | 0.5536 | 0.2360 | 93/38/123/112 |
| ae | AE-YJ / on | 0.6546 | 0.3036 | 0.5919 | 0.6196 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/161/205 |
| ensemble | IF-log+SVM-base / off | 0.7682 | 0.5524 | 0.7015 | 0.7413 | 0.7952 | 0.6439 | 0.7116 | 0.2112 | 132/34/127/73 |
| ensemble | IF-raw+SVM-base / on | 0.7946 | 0.6158 | 0.7191 | 0.7587 | 0.7670 | 0.6585 | 0.7087 | 0.2547 | 135/41/120/70 |

### 7.2 Matched-configuration augmentation (102 pairs)

Identical family/recipe/hyperparameters, augmentation on minus off, validation
deltas. `positive` counts pairs with ΔROC > 0.

| Family | Recipe | Pairs | Positive | Median ΔROC | ΔROC range | Median ΔAP |
|---|---|---|---|---|---|---|
| if | IF-log | 12 | 9 | +0.0133 | −0.0049 … +0.0335 | +0.0013 |
| if | IF-raw | 12 | 12 | +0.0229 | +0.0061 … +0.0393 | +0.0494 |
| ocsvm | SVM-base | 9 | 6 | +0.0242 | −0.0172 … +0.0303 | +0.0406 |
| ocsvm | SVM-log | 9 | 6 | +0.0174 | −0.0224 … +0.0391 | +0.0439 |
| ocsvm | SVM-quantile | 9 | 8 | +0.0392 | −0.0135 … +0.0592 | +0.0503 |
| ocsvm | SVM-YJ | 9 | 6 | +0.0247 | −0.0208 … +0.0494 | +0.0381 |
| ae | AE-base | 14 | 12 | +0.0039 | −0.0027 … +0.0093 | +0.0148 |
| ae | AE-log | 14 | 14 | +0.0113 | +0.0058 … +0.0271 | +0.0164 |
| ae | AE-YJ | 14 | 13 | +0.0115 | −0.0031 … +0.0310 | +0.0115 |

**86 of 102 pairs improve validation ROC-AUC**; positive medians across every
recipe indicate a broad modest ranking benefit rather than one winner-specific
effect. Pairs are correlated evaluations on the same validation sessions.

### 7.3 Independently tuned recipe winners (off → on)

| Family | Recipe | Off ROC | On ROC | ΔROC | ΔAP | Threshold off → on |
|---|---|---|---|---|---|---|
| if | IF-log | 0.7618 | 0.7818 | +0.0200 | +0.0133 | −0.0196 → −0.0235 |
| if | IF-raw | 0.7519 | 0.7847 | +0.0328 | +0.0552 | −0.0395 → −0.0622 |
| ocsvm | SVM-base | 0.7361 | 0.7664 | +0.0303 | +0.0623 | +0.0331 → −0.0191 |
| ocsvm | SVM-log | 0.6549 | 0.6919 | +0.0370 | +0.0724 | +0.2937 → +0.0667 |
| ocsvm | SVM-quantile | 0.6433 | 0.6822 | +0.0389 | +0.0504 | +0.0579 → +0.8243 |
| ocsvm | SVM-YJ | 0.6379 | 0.6870 | +0.0491 | +0.0510 | +0.0160 → +0.1003 |
| ae | AE-base | 0.7420 | 0.7419 | −0.0001 | +0.0189 | +0.4333 → +0.3978 |
| ae | AE-log | 0.7317 | 0.7445 | +0.0127 | +0.0244 | +0.4178 → +0.3605 |
| ae | AE-YJ | 0.6554 | 0.6546 | −0.0007 | +0.0011 | +19.0973 → +17.1668 |

SVM-YJ uses the F1-threshold fallback in both conditions (validation precision
below 0.5); the other primary thresholds satisfy the precision floor, and
AE-YJ's floor compliance comes with very low recall (see §7.6).

### 7.4 Bounded AE depth (2/2 vs 1/1 at batch 32, lr 1e-3)

| Recipe | Pairs | Positive | Median ΔROC | ΔROC range |
|---|---|---|---|---|
| AE-base | 4 | 4 | +0.0082 | +0.0045 … +0.0130 |
| AE-log | 4 | 3 | +0.0036 | −0.0042 … +0.0107 |
| AE-YJ | 4 | 2 | +0.0014 | −0.0021 … +0.0042 |

Nine of 12 pairs improve; gains are small. Base/log recipe winners use two
layers, YJ selects one layer. No evidence that two layers universally improve
temporal anomaly learning.

### 7.5 Epoch cap, runtime, and seed sensitivity

| Recipe | Aug | Fits | At 100 epochs | Epoch range |
|---|---|---|---|---|
| AE-YJ | off | 14 | 14 | 100–100 |
| AE-YJ | on | 14 | 13 | 93–100 |
| AE-base | off | 14 | 13 | 97–100 |
| AE-base | on | 14 | 10 | 80–100 |
| AE-log | off | 14 | 13 | 78–100 |
| AE-log | on | 14 | 14 | 100–100 |

**77 of 84 main AE fits reach 100 epochs.** At batch 32, augmentation off uses
2,100 updates and on 6,300, so extra optimization is part of the on-condition
approach. Recorded candidate fit+score time totals **2,189.89 s (36.50 min)**;
AE accounts for 2,140.54 s (97.7%), IF 41.5 s, OCSVM 7.9 s. These are sums of
per-candidate recorded times, not wall-clock orchestration time.

Seed repeats (finalists: IF-log/off, IF-raw/on, AE-base/off, AE-log/on;
fixed data/augmentation/preprocessing, own thresholds and checkpoints):

| Family | Recipe | Aug | Seed | Val ROC | Val AP | Test ROC | Test AP | Prec | Rec | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| if | IF-log | off | 123 | 0.7462 | 0.5427 | 0.6916 | 0.7397 | 0.7778 | 0.5463 | 0.6418 |
| if | IF-log | off | 2026 | 0.7487 | 0.4979 | 0.7005 | 0.7478 | 0.7853 | 0.6244 | 0.6957 |
| if | IF-raw | on | 123 | 0.7557 | 0.5936 | 0.6997 | 0.7509 | 0.7945 | 0.5659 | 0.6610 |
| if | IF-raw | on | 2026 | 0.7739 | 0.5880 | 0.7034 | 0.7432 | 0.7590 | 0.6146 | 0.6792 |
| ae | AE-base | off | 123 | 0.7395 | 0.5469 | 0.6385 | 0.6930 | 0.7259 | 0.4780 | 0.5765 |
| ae | AE-base | off | 2026 | 0.7411 | 0.5404 | 0.6364 | 0.6920 | 0.7218 | 0.4683 | 0.5680 |
| ae | AE-log | on | 123 | 0.7401 | 0.5631 | 0.6321 | 0.6717 | 0.7059 | 0.4683 | 0.5630 |
| ae | AE-log | on | 2026 | 0.7387 | 0.5623 | 0.6328 | 0.6716 | 0.7132 | 0.4488 | 0.5509 |

Including the seed-42 rows: IF-log/off test ROC 0.6881–0.7005; IF-raw/on
0.6997–0.7138; AE-base/off 0.6364–0.6385; AE-log/on 0.6314–0.6328. AE's weaker
ranking is consistent across seeds; IF ranking varies modestly while
thresholded recall moves by roughly ten points. This measures
model-initialization sensitivity, not split independence.

### 7.6 AE-Yeo–Johnson threshold-transfer diagnostic

Both AE-YJ approaches predict every test window normal at the transferred
threshold, while their test ranking remains weakly nonconstant (ROC ≈ 0.58 /
0.59, AP ≈ 0.61 / 0.62).

| Condition | Stored threshold | Val TP | Val FP | Val recall | Max test score | Test TP/FP/FN |
|---|---|---|---|---|---|---|
| off | 19.097293853759766 | 1 | 1 | 1/82 = 0.0122 | 8.469143867492676 | 0 / 0 / 205 |
| on | 17.166751861572266 | 1 | 1 | 1/82 = 0.0122 | 8.301196098327637 | 0 / 0 / 205 |

The precision-floor rule accepts two validation predictions (one anomalous, one
normal), attains precision 0.5, avoids the F1 fallback, and keeps an extreme
threshold with almost no recall. This is an operating-threshold transfer
failure at a two-point validation decision, not evidence of test leakage or
invalid scores; the failing threshold stays in the frozen results rather than
being lowered post-test.

## 8. Metric definitions

- **ROC-AUC**: ranking quality across thresholds; 0.5 is chance under the usual
  binary interpretation.
- **`pr_auc`**: scikit-learn **average precision**, not a trapezoidal PR
  integral. Reported no-skill baseline = anomaly prevalence:
  validation 82/393 = 0.208651, test 205/366 = 0.560109.
- **Precision** = TP/(TP+FP), **recall** = TP/(TP+FN), **F1** = harmonic mean,
  **FPR** = FP/(FP+TN); precision is reported as 0 when there are no positive
  predictions (a zero-division convention).
- Scores and thresholds are not probabilities and need not lie in [0, 1];
  classification is `score >= threshold` everywhere (search, tuning, API).
- Higher test AP than validation does not by itself imply better generalization:
  test prevalence is much higher and the partitions are not exchangeable
  populations.

## 9. Reproducible artifacts

Machine-readable copies: `tables/candidates.csv` (204 candidate rows with
validation metrics, runtime, diagnostics, failure status),
`tables/reporting_manifest.json` (frozen primary/repeat entries, parameters,
thresholds, artifacts), `tables/augmentation_matched.csv`,
`tables/augmentation_winners.csv`, `tables/depth_comparison.csv`,
`tables/test_results.csv` (28 test rows), `tables/dataset_summary.csv`,
`tables/final_metrics.csv`. Source artifacts and figures remain under
`study/artifacts/phase_04_study/` (locally generated). Reproduction commands are
in `04_selected_models_and_conclusions.md` §6.

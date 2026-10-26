# Study Phase 1 — Original-training audit findings

> Historical document: These are historical audit findings. Earlier module paths, tracking statements, and development instructions are retained as context. See the [study inventory](../README.md) for preserved files and the [final report](../../reports/experiment1/README.md) for the completed study.

Status: Phase 1 complete; user accepted the refreshed audit and frozen specification
with the approved angle-wrapping correction. No Phase 2
implementation, detector fits, labeled validation metrics, or test results.

## Reproducible boundary and artifacts

- Entry point from the repository root: `python -m study.audit`.
- Reporting notebook: `study/notebooks/original_training_audit.py`, paired with Jupytext.
  It reads saved diagnostics; the user executes it.
- Output: `study/artifacts/phase_01_audit/` (entire `study/` folder is Git-ignored):
  `manifest.json`, `sessions.csv`, `parent_identities.csv`,
  `parent_features.csv`, `micro_features.csv`, `transformed_features.csv`,
  `kernel_geometry.csv`.
- Recomputed the sorted-file seed-42 split and verified all four assignments
  against the existing `dataset/processed/split_info.json`. Only original normal
  training CSV contents were loaded. Held-out filenames were listed, not read.
- Re-extracted from raw training files; no legacy feature/sequence cache,
  augmentation, feature selection, model fitting, or production writes.
- Recorded training-file hashes, ordered parent identities `(session_file,
  parent_start_event)`, event offsets, start times, source/output column order,
  geometry, transform diagnostics, and the planned model-fit budgets.
- Diagnostic transformation parameters are inspection outputs, not deployable
  preprocessors. Phase 2 still implements fitted preprocessing for reuse,
  train-derived imputation, serialization, and controlled augmented transforms.

## 1. Data size and fixed representations

| Quantity | Observed |
|---|---:|
| Original normal-training session files | 15 |
| Parent windows, 50 events / stride 25 | 662 |
| AE micro-feature rows, 10 events / stride 5 | 5,958 |
| AE timesteps per parent | 9 |
| Fixed source features | 25 |
| Encoded columns | 33 |
| Parent-window range per training file | 17–111 |

Parent and sequence counts agree per session. Both source representations contain
all specified columns and only normal labels. There are **no nonfinite extracted
values and no constant source columns** in either representation. This is checked
before any feature imputation; extraction itself skips invalid derivative values.

All direction codes 0–8 occur. Freeze output order as the 24 numerical source
columns in source order, followed by direction indicators 0–8. This is a study
representation; the old deployment pipeline still uses its own feature list.

The 5,958 micro rows are not independent observations: parent windows overlap,
and adjacent parents reuse micro-windows. Fifteen files also do not establish
15 distinct students; filename variants suggest repeat recordings. Keep session
and student generalization limitations explicit.

## 2. Transform signs and units

**Required specification correction: jerk uses signed log, not ordinary log.**

| Feature group | Extraction meaning / units | Log decision |
|---|---|---|
| elapsed_time | Last minus first event timestamp, seconds | L |
| path_length, largest_deviation | Path sum / perpendicular deviation, pixels | L |
| velocity_mean | Mean speed magnitude, pixels/second | L |
| acceleration_mean | Mean vector acceleration magnitude, pixels/second² | L |
| jerk_mean | Mean derivative of acceleration magnitude, pixels/second³; signed | **SL** |
| angular_velocity_mean/min | Wrapped signed angle differences / dt, radians/second | SL |
| angular_velocity_std | Nonnegative angular variability, radians/second | L |
| curvature_mean/std/min | Absolute cross product divided by speed³, inverse pixels | L |
| keyboard_typing_rate | Keyboard-event count / keyboard timestamp span, events/second | L |
| copy/paste, clicks, burst/pause, switches | Event or gap counts, dimensionless | Plan's fixed count treatment |
| straightness, idle_time_ratio | Dimensionless ratios | I |
| direction_changes, sharp_angles | Thresholded successive-angle counts | I |
| sum_of_angles | Sum of absolute successive-angle differences, radians | I |
| direction_class | Zero sentinel or 45-degree sector code | Fixed one-hot |

Negative jerk occurs in **15.86% of parent rows** and **22.99% of micro rows**.
Negative acceleration/curvature does not occur, consistent with their absolute
magnitude definitions. Therefore resolve acceleration/curvature L* to L and
jerk L* to SL. Do not use `log1p(max(jerk, 0))`: that would erase deceleration.

Source units remain seconds and pixels, including the inverse-pixel curvature
input to log1p. These are unit-dependent recipes, not unit-invariant physical
normalizations.

## 3. Feature names do not always describe their actual semantics

These are extraction-definition caveats, not issues discovered from validation.
Do not silently change them during the preprocessing implementation.

1. **Direction class:** the code starts at +x, not North despite its comment.
   Categories 1–8 are `[0°,45°)`, `[45°,90°)`, …, `[315°,360°)` using
   `atan2(end_y-start_y, end_x-start_x)`. With screen y increasing downward,
   angles increase clockwise. Category 0 means zero **endpoint displacement**
   or fewer than two mouse events; a closed nonzero-length path also gets 0.
   Call it the zero-displacement/insufficient-events sentinel, not no movement.
   One-hot codes are still valid and avoid imposing artificial ordered distances.
2. **Sharp angles:** threshold is 0.0005 radians (about 0.029°), so this counts
   almost any nonzero angle change, not only visibly sharp turns. Preserve the
   current feature as a fine-threshold turn count if retaining the extraction.
3. **Angle differences (corrected with user approval):** direction_changes,
   sum_of_angles, and sharp_angles now reuse the wrapped signed difference already
   used by angular_velocity, taking its absolute value for counts/sums. A +179° to
   -179° transition contributes 2°, not 358°. Thresholds remain π/4 and 0.0005
   radians. Regression tests cover both crossing directions, tiny crossings,
   genuine large turns, and insufficient/unchanged directions. Angular velocity
   itself and all other extraction formulas are unchanged.
4. **Typing rate:** keyboard events include copy/paste, not just keydown. There
   are 12 such raw events across three original training files. Interpret this
   feature as keyboard-event rate unless extraction is explicitly revised.
5. **Idle ratio:** time associated with the within-window lowest-decile speeds,
   not an absolute inactivity detector; a mostly constant speed can count as idle.
6. **Single mouse event:** the short-circuit returns every mouse feature as zero,
   including click_count. A sole click is therefore not counted by this feature.

**Recommendation for this bounded PoC:** document the remaining legacy definitions
and keep extraction fixed across all approaches. Signed-log and direction-label
clarifications are mandatory. The approved wrapped-turn fix is included in the
refreshed audit. Singleton-click counting remains unchanged and outside this
approved fix; any later correction requires explicit approval and another audit.
Do not redefine jerk, idle detection, or keyboard-event rate simply to improve
model results. They can remain operational features with honest names.

## 4. Timing, zeros, and tails

- Raw event order has 12 timestamp decreases across three training files, and
  two duplicate mouse timestamp gaps across two files. Mouse/keyboard extraction
  sorts its subsets; parent windows still follow the CSV's event order. Do not
  silently globally reorder events: that changes parent membership and labels.
- Minimum positive mouse dt is approximately 0.001 seconds. Derivatives amplify
  tiny dt: parent acceleration reaches 8.27e6 and jerk 8.42e9; micro jerk reaches
  5.41e10. Finite does not imply physically plausible. Log/YJ comparisons are
  useful here, but do not claim preprocessing removes logging artifacts.
- Keyboard-event rate reaches 2,000 events/second in both representations, due
  to the count/span definition and short timestamp spans. No epsilon/clipping
  choice should be added after observing validation outcomes.
- Parent copy/paste/switch zero fractions are 98.34% / 98.19% / 97.58%; micro
  fractions are 99.60% / 99.66% / 99.36%. Retain these columns and keep the agreed
  exclusion from variance scaling: they are rare, not absent.
- Micro features have more zeros because ten-event windows often have too few
  relevant events. E.g., zero mouse_path_length is 12.54% for parents versus
  30.90% for micro rows. Separate AE-fitted preprocessing is justified.
- All observed x/y maxima are within the current 1920×1080 clipping bounds.
  This does not establish identical screen resolutions across students.

Keep timing caveats in final interpretation. A timestamp-cleaning study or a new
velocity floor would add a new feature-engineering intervention, beyond the
agreed preprocessing comparison.

## 5. Recipe effects and numerical readiness

Every diagnostic recipe produces finite 33-column output without transformer
warnings. No source feature is dropped; fixed one-hot order is shared. Constants
are exercised by synthetic tests even though none occur in this training audit.

| Representation / recipe | Largest absolute transformed value |
|---|---:|
| Parent / base + scaling | 25.71 |
| Parent / log + scaling | 15.94 |
| Parent / YJ + scaling | 17.66 |
| Parent / quantile + scaling | 6.19 |
| Micro / base + scaling | 77.17 |
| Micro / log + scaling | 31.54 |
| Micro / YJ + scaling | 48.09 |

Raw AE jerk is especially concentrated; squared reconstruction loss can be
dominated by such tails. The log recipe reduces this risk without guaranteeing
balanced loss. In the micro log recipe, sparse curvature_min still reaches
31.54 standard deviations. YJ is not guaranteed to remove both signed tails:
micro jerk remains roughly -34.28 to +48.09 after YJ and scaling.

Parent curvature_min has 83.38% zeros and a maximum only 0.0683 inverse pixels.
Its YJ lambda is about -390.25 (micro: -42.44). The fit is finite, with no warning,
but the large lambda shows strong unit/zero-mass sensitivity. Do not arbitrarily
bound lambdas or reweight loss now; record this and validate transforms on other
inputs in Phase 2. Quantile transforms also saturate outside training range and
cannot guarantee preserving anomalous magnitude.

These observations support comparing the already agreed complete recipes;
they do not justify another scaler/feature-weight grid or duplicate augmented EDA.

## 6. Configuration coverage

### IF

662 original rows exceed both 128 and 256; effective max_samples are distinct
before augmentation. Keep max_samples=(128,256), max_features=(0.75,1.0),
n_estimators=(100,300,500). With d=33, the feature subsamples are 24 versus 33
columns (integer truncation); they sample encoded columns, not source groups.
No need to expand the grid merely because the training count is larger.

### OCSVM

Compute `gamma_0 = 1 / (33 * X_original_transformed.var(ddof=0))` per parent
recipe; the variance is over all entries. Estimated kernels below use the same
512-row seed-42 sample, 130,816 off-diagonal pairs, for every recipe.

| Recipe | gamma_0 | Median kernel at 0.25× | At 1× | At 4× |
|---|---:|---:|---:|---:|
| base | 0.043590 | 0.7681 | 0.3480 | 0.01467 |
| log | 0.045288 | 0.6806 | 0.2146 | 0.00212 |
| YJ | 0.045288 | 0.6948 | 0.2331 | 0.00295 |
| quantile | 0.045288 | 0.6948 | 0.2330 | 0.00295 |

The three multipliers span broad to local similarity rather than three nearly
identical kernels. At 4×, 45.71%–65.07% of sampled kernels are below 0.01: this
is intentionally the local end, not evidence it will generalize best. None is
uniformly all zero or all one. Keep the bounded gamma and nu grids.

Log/YJ/quantile centres coincide because scaled numerical columns have unit
variance and the unscaled counts/indicators have the same treatments; their
pairwise geometry still differs. Freeze these numeric centres from originals
for both augmentation states in Phase 2 rather than refitting `gamma="scale"`.

### AE and budget

No training-only evidence warrants expanding the hidden/batch/lr grid. Preserve
the matched one/two-layer comparison and regularization/update accounting.
Budget remains 48 IF + 72 OCSVM + 84 AE = **204 main model fits**, plus eight
locked-finalist repeats = **212**; **20 primary final reports**.

## Review gate

### Approved wrapping correction: refreshed training-only observations

| Statistic | Before wrapping | After wrapping |
|---|---:|---:|
| Parent direction_changes median / maximum | 9 / 29 | 8 / 25 |
| Parent sum_of_angles median / maximum, radians | 31.69 / 102.26 | 21.30 / 62.38 |
| Micro sum_of_angles median / maximum, radians | 3.74 / 35.20 | 3.20 / 20.64 |

The sharp-angle median/max stay unchanged under the intentionally tiny threshold;
the boundary tests establish correctness for smaller-than-threshold crossings.
Training counts, finite-value checks, and configuration budgets remain unchanged.
Gamma centres retain the same rounded values; kernel summaries above are refreshed.
Superseded pre-fix audit files were removed after regeneration, leaving one current
artifact set in `study/artifacts/phase_01_audit/`. All study code, tests, notebooks,
plans, and generated artifacts are consolidated under the ignored `study/` root,
separate from the production package. No audit data was regenerated merely for
this relocation.

The audit and its numerical questions are resolved. The study specification is
**frozen with user approval**. The angle-wrapping correction is approved and implemented;
the remaining legacy definitions are documented, not silently changed. No extra
hyperparameter candidates, augmentation EDA, or validation/test inspection is
proposed. Stop before Phase 2.

Existing unversioned feature/sequence caches and saved models were not overwritten.
They predate this extraction correction and must not be mixed with fresh features.
Phase 2's fresh-raw study path remains necessary; any later use of the legacy
training pipeline must explicitly regenerate its affected caches and refit models.

> **SUPERSEDED.** Archived for provenance only. The canonical Experiment 1
> report is `../01_features.md`, `../02_data_sessions_and_windows.md`,
> `../03_experiments_and_models.md`, and
> `../04_selected_models_and_conclusions.md`.

> Historical reference: original tracking/status statements predate snapshot preservation. Start with the [final report index](../README.md) and [reproduction guide](../REPRODUCING.md) for current navigation and prerequisites.

# Experiment 1 — Behavioral anomaly detection: complete study report

**Report date:** 2026-10-09  
**Purpose:** A self-contained account of the implemented experiment, its feature definitions, preprocessing, model selection, saved results, and limitations.  
**Scope:** Proof of concept on the available annotated session files; not a production-readiness or misconduct-verification assessment.  
**Snapshot:** Repository HEAD `776f629000d21abefc28c36b46fb7ad33394f86f`, with additional working-tree changes present when this report was written. Numerical results come from the saved artifacts identified in Section 15, not from newly trained models.

## Contents

1. Executive conclusion
2. Research questions and experimental phases
3. Dataset, cleaning, splits, windows, and labels
4. Feature dictionary and extraction mathematics
5. Original-training audit and feature-schema decisions
6. Preprocessing recipes: what changes, where, and why
7. Paired augmentation and comparison fairness
8. Models, tuning grids, and fixed training controls
9. Selection, score calibration, thresholds, and metrics
10. Complete primary results
11. Selected configurations and operating thresholds
12. Augmentation, depth, runtime, and seed observations
13. AE-Yeo–Johnson threshold diagnostic
14. Interpretation, limitations, and engineering caveats
15. Artifact inventory, provenance, and reproduction
16. Final study statement

## 1. Executive conclusion

The study finds a useful but imperfect behavioral anomaly signal in this dataset. Isolation Forest (IF), One-Class SVM (OCSVM), and their ensemble generalize better than the tested LSTM autoencoder (AE). Small coordinate-noise augmentation usually improves validation ranking, but does not uniformly improve precision, recall, F1, or test average precision. The bounded two-layer AE comparison yields mostly small validation gains rather than a decisive improvement.

**Best overall under the agreed validation ROC-AUC rule:** augmentation-on ensemble of **IF-raw + SVM-base**, with **70% IF and 30% OCSVM after separate robust score normalization**.

| Measure | Selected ensemble |
| --- | --- |
| Validation ROC-AUC | 0.7946 |
| Validation average precision | 0.6158 |
| Validation-selected threshold | 0.6587707622593663 |
| Test ROC-AUC | 0.7191 |
| Test average precision | 0.7587 |
| Test precision | 0.7670 |
| Test recall | 0.6585 |
| Test F1 | 0.7087 |
| Test false-positive rate | 0.2547 |
| Test TP / FP / TN / FN | 135 / 41 / 120 / 70 |

The selected detector catches 135 of 205 anomalous test windows, misses 70, and flags 41 of 161 normal windows. These are **window-level** results, not counts of correctly identified students or sessions.

Augmentation-on SVM-base has the highest primary **test** ROC-AUC, 0.7323, but this does not retrospectively replace the validation-selected ensemble. Doing so would use test results for model selection.

Execution completed with **204 successful main model fits, eight finalist repeats, 20 primary reporting rows, and eight sensitivity rows**. All 72 OCSVM candidates reported convergence, and the saved reporting manifest contains no missing cells.

## 2. Research questions and experimental phases

### 2.1 Questions

- Can normal-only anomaly detectors learn signals associated with the annotated anomalous behavior?
- Which complete, model-appropriate preprocessing recipe works best within the declared search?
- Does training-only coordinate augmentation help when comparing the same configuration, and when independently tuning each augmentation condition?
- Does a bounded two-layer AE variant improve over its matched one-layer counterpart?
- Are locked IF/AE finalist results sensitive to initialization seed?

Earlier development involved ad-hoc preprocessing and validation ablations. The experiment brief records an earlier AE comparison around ROC-AUC 0.46 versus 0.77 as motivation. Those values are **historical development observations**, not Experiment 1 result rows. This experiment isolates whole recipes and augmentation more clearly; it is not a claim that the study was designed without any previous validation exposure.

### 2.2 Work completed

| Stage | Implemented work | Evidence in this snapshot |
| --- | --- | --- |
| Protocol definition | Fixed windows, feature schema, recipe catalog, grids, seeds, selection metric, threshold rule, and reporting stages | Experiment-specific configuration and saved run manifest |
| Phase 1: original-training audit | Feature-domain, sign, sparsity, timing, transform, and kernel-geometry diagnostics; consistent angle wrapping | Original-training audit artifacts |
| Phase 2: readiness | Fresh paired parent/micro extraction; original-only fitted preprocessing; augmentation identity checks; explicit unusable-file reporting | Readiness and prepared-bundle artifacts |
| Phase 3: controls and isolation | Separate AE dropout controls, weight decay, clipping, early-stopping tolerance, solver/training diagnostics, validation-only runner | Model implementations and experiment runner |
| Phase 4: main search | Both augmentation conditions; recipe/configuration comparisons; saved fitted recipe winners and ensembles | 204 candidate rows and 20 primary approaches |
| Phase 5: sensitivity | Repeat locked IF/AE finalists at seeds 123 and 2026 without reselecting a seed | Eight additional fitted artifacts |
| Phase 6: reporting | Locked-threshold test metrics, confusion counts, dataset summary, ROC/PR/confusion/score figures | 28 test rows and grouped figures |

Reusable extraction/preprocessing lives in production Python modules under `src/cheatdetect/data/`. The production preparation path delegates to the same paired dataset builder. Shared model selection and final reporting are separated under `src/cheatdetect/pipeline/`; experiment-specific orchestration, configuration, analysis, and notebooks live under `src/cheatdetect/experiments/experiment_1/`.

Notebook execution is research orchestration, not a second implementation of core extraction or inference. Study artifacts remain separate from deployable model outputs. Completing the study does **not** automatically replace the API's saved model.

## 3. Dataset, cleaning, splits, windows, and labels

### 3.1 Actual data scale

This run assigns **36 session files**: 19 pure-normal files and 17 mixed files. Of these, **34 contribute usable windows**. File counts are not necessarily unique-student counts: the same participant can have normal and mixed recordings or multiple recordings.

The assigned files contain **38,039 raw rows** and **36,784 cleaned events** in total. Raw row counts include discarded metadata/summary content, so they should not be interpreted as event counts before feature extraction without qualification.

| Partition | Assigned sessions | Contributing sessions | Parent windows | Anomalous windows | Prevalence |
| --- | --- | --- | --- | --- | --- |
| Original normal training | 15 | 15 | 662 | 0 | 0% |
| Synthetic normal training | Same 15 originals | Same 15 originals | 1,324 | 0 | 0% |
| Pure-normal validation | 4 | 4 | 197 | 0 | 0% |
| Mixed validation | 5 | 5 | 196 | 82 | 41.84% |
| Combined validation | 9 | 9 | 393 | 82 | 20.87% |
| Mixed test | 12 | 10 | 366 | 205 | 56.01% |

Augmentation off uses 662 training parents/sequences. Augmentation on uses **1,986 = 662 originals + 1,324 synthetic copies**, still derived from the same 15 training sessions.

Two assigned mixed-test files contain 32 raw rows each but no cleaned events: `mixed - El Farouk Omar.csv` and `youssef ahmed- mixed.csv`. They remain in the frozen file assignment and are explicitly excluded from window evaluation. No replacement files are drawn to compensate.

### 3.2 Split policy

The seed-42 split is made at the **session-file level before window extraction**:

- Pure-normal files: 80% training, 20% normal validation, yielding 15 and four files.
- Mixed files: 30% validation, 70% test, yielding five and 12 assigned files.
- No training-file windows are randomly scattered into validation/test.
- Ordered file assignments, not only the seed, are preserved in the audit/prepared manifests.
- No pure-normal session partition is added to final test; test normal examples are the normal-labeled windows within contributing mixed files.

This is stronger than a random overlapping-window split, but it is **not an unseen-student split**. Student overlap remains a limitation. Validation includes pure-normal sessions, whereas test contains mixed sessions only; their normal-window populations may differ.

### 3.3 Raw-event cleaning

The implemented cleaning path:

1. Keeps time, event type, X/Y coordinates, action, and the event annotation.
2. Removes the footer starting at the `SESSION SUMMARY` marker.
3. Coerces time to numeric and removes invalid-time rows.
4. Tags mouse events (`mousemove`, `click`), keyboard events (`keydown`, `copy`, `paste`), and action events.
5. Coerces coordinates, forward-fills X/Y within the session, and drops remaining missing coordinates.
6. Creates copy/paste/blur/focus/tab-switch flags.
7. Parses the annotation by comparing the stripped uppercase string to `TRUE`; a missing annotation defaults to false.

The cleaner preserves session event order. Mouse and keyboard feature calculations separately sort their selected events by time; this is not equivalent to globally reordering parent-window membership.

Labels and session identities are kept separately from model inputs. No event annotation, filename, or student/session identifier is a predictor.

### 3.4 Parent and sequence geometry

A parent contains **50 consecutive cleaned events**, advanced by **25 events**. For a session with E cleaned events, the number of usable parents is:

`max(0, floor((E - 50) / 25) + 1)`.

Inside each parent, AE micro-windows contain **10 events**, advanced by **five events**. The starting offsets are 0, 5, ..., 40, giving:

`floor((50 - 10) / 5) + 1 = 9 timesteps`.

IF/OCSVM receive one 25-source-feature aggregate per parent, transformed to 33 columns. AE receives a `(N, 9, 25)` source tensor, transformed to `(N, 9, 33)`. Flat and sequence samples refer to exactly the same realized parent windows, in the same order, with the same parent labels. They are aligned evaluations, but not identical representations: AE retains within-parent temporal structure.

Parent windows overlap; adjacent micro-windows also overlap. Repeated micro-windows across adjacent parents are not additional independent observations.

### 3.5 Ground truth

A parent is labeled anomalous when:

`mean(event_is_cheating within the 50-event parent) >= 0.5`.

Thus a parent with 25 annotated anomalous events qualifies. This label is attached to both flat and AE representations. Micro-window labels are not supplied to AE as targets; AE reconstructs features and is evaluated against the parent label.

Original training and its synthetic copies are normal-only. The loader rejects annotated anomalous events in a normal partition. Session labels are supplied annotations, not independently verified misconduct findings established by the model.

## 4. Feature dictionary and extraction mathematics

### 4.1 Definitions shared by mouse features

Let consecutive **time-sorted mouse events** have coordinates (x_i, y_i) and times t_i:

- `dx_i = x_i - x_(i-1)`, `dy_i = y_i - y_(i-1)`, `dt_i = t_i - t_(i-1)`.
- `vx_i = dx_i / dt_i`, `vy_i = dy_i / dt_i`.
- `speed_i = sqrt(vx_i² + vy_i²)`.
- `ax_i = (vx_i - vx_(i-1)) / dt_i`, likewise for ay.
- `accel_i = sqrt(ax_i² + ay_i²)`.
- `jerk_i = (accel_i - accel_(i-1)) / dt_i`.
- `theta_i = atan2(dy_i, dx_i)`.
- `turn_i = ((theta_i - theta_(i-1) + pi) mod (2*pi)) - pi`.
- `omega_i = turn_i / dt_i`.
- `curvature_i = abs(vx_i*ay_i - vy_i*ax_i) / speed_i³`.

Angles are in radians, time is in seconds, and distances are in pixels. The wrap is into **[-pi, pi)**. Crossing the angular boundary should represent a small turn rather than an artificial approximately 2*pi jump.

Undefined derivatives and infinities from zero dt or zero speed are converted to missing values and skipped during aggregation. Empty valid aggregates return zero. Standard-deviation aggregates use the population convention, ddof=0. Very small positive dt is **not** clipped, so large finite derivative values remain possible.

For fewer than two mouse events, the extractor returns zero for every mouse feature. This includes click count; consequently, a lone click is currently undercounted. Zero can therefore mean no applicable data or a genuine zero value, rather than one unique behavioral condition.

### 4.2 All 25 model-source features

“Group” maps each feature to the exact transformation rules in Section 6.

| Source feature | Actual calculation and units/domain | Intended information | Group |
| --- | --- | --- | --- |
| `elapsed_time` | Last minus first timestamp of the parent/micro event slice; seconds, nonnegative in accepted feature rows | Time needed to generate a fixed number of events | Continuous, nonnegative |
| `copy_events` | Sum of copy flags over all events; when Action is present, either Action or Event Type equal to copy sets the flag | Clipboard activity; not proof of cheating | Rare count, unscaled |
| `paste_events` | Sum of paste flags, analogously to copy | Clipboard activity | Rare count, unscaled |
| `mouse_path_length` | Sum of sqrt(dx² + dy²); pixels | Amount of cursor travel | Continuous, nonnegative |
| `mouse_straightness` | Start-to-end distance divided by path length; zero for zero path length; [0,1] | Direct versus wandering paths | Bounded ratio |
| `mouse_velocity_mean` | Mean valid speed; pixels/second | Average movement speed | Continuous, nonnegative |
| `mouse_acceleration_mean` | Mean valid accel magnitude; pixels/second² | Changes in movement speed/direction | Continuous, nonnegative |
| `mouse_jerk_mean` | Mean valid derivative of scalar accel magnitude; pixels/second³, signed | Increasing versus decreasing acceleration magnitude | Continuous, signed |
| `mouse_direction_changes` | Count of abs(turn) > pi/4; strict 45-degree threshold | Substantial changes of heading | Ordinary count |
| `mouse_click_count` | Number of mouse events with Event Type equal to click; zero under the fewer-than-two-mouse-events fallback | Interaction frequency | Ordinary count |
| `mouse_idle_time_ratio` | Sum of dt where speed is at/below that window's 10th-percentile valid speed, divided by total mouse dt; [0,1] | Relative slow-motion time | Bounded ratio |
| `mouse_angular_velocity_mean` | Mean valid omega; radians/second, signed | Average signed turning rate | Continuous, signed |
| `mouse_angular_velocity_std` | Population std of valid omega; radians/second, nonnegative | Variability of turning rate | Continuous, nonnegative |
| `mouse_angular_velocity_min` | Minimum valid omega; radians/second, signed | Most negative signed turning rate | Continuous, signed |
| `mouse_curvature_mean` | Mean valid curvature; inverse pixels | Typical geometric bending | Continuous, nonnegative |
| `mouse_curvature_std` | Population std of valid curvature; inverse pixels | Variation in bending | Continuous, nonnegative |
| `mouse_curvature_min` | Minimum valid curvature; inverse pixels | Least-curved valid segment | Continuous, nonnegative |
| `mouse_direction_class` | Net-displacement atan2 mapped into eight 45-degree sectors; 0 for zero net displacement or insufficient mouse events | Categorical net movement direction | Categorical |
| `mouse_sum_of_angles` | Sum of abs(wrapped turn); radians | Total amount of turning | Angle aggregate |
| `mouse_largest_deviation` | Maximum perpendicular distance to the infinite start-to-end line; pixels; zero for fewer than three mouse events or zero net displacement | Departure from a direct path | Continuous, nonnegative |
| `mouse_sharp_angles` | Count of abs(turn) > 0.0005 radians | Near-nonzero heading changes under the existing threshold | Ordinary count |
| `keyboard_typing_rate` | Number of keyboard-category events divided by their first-to-last time span; events/second; zero for fewer than two events or nonpositive span | Keyboard-event tempo | Continuous, nonnegative |
| `keyboard_burst_count` | Count of consecutive keyboard-category time gaps < 0.2 seconds | Rapid event bursts | Log-count |
| `keyboard_pause_count` | Count of consecutive keyboard-category time gaps > 1.0 second | Long gaps | Log-count |
| `window_switch_events` | blur_events + focus_events + tab_switch_events; sums event flags, not distinct switch episodes | Browser/window transitions | Log-count, unscaled |

Interpretation details that matter:

- Keyboard-category events include copy and paste, not only keydown. “Typing rate” is therefore an event-rate proxy, not text characters per second.
- Jerk is the derivative of **scalar acceleration magnitude**, not the magnitude of a vector jerk. Negative jerk is valid.
- Angular mean/minimum are signed. Angular minimum can also be positive if all valid turns are positive; it is not necessarily negative.
- Idle ratio is relative to each window's own speed distribution; it is not time with a stationary cursor or an absolute idle threshold.
- The existing “sharp angles” threshold is only 0.0005 radian, about 0.029 degrees. The name should not be read as a conventional large/sharp-turn definition.
- Largest deviation is distance from an infinite line, not distance from the finite line segment.
- Direction sector 1 is 0–45 degrees from +X, sector 2 is 45–90 degrees, ..., sector 8 is 315–360 degrees. With screen Y increasing downward, positive angular rotation is clockwise. The implementation is not the compass mapping suggested by an old code comment.
- Direction category 0 also covers a path returning to its starting point; it does not always mean no movement.
- Raw action counts depend on the cleaned flags. If Action is absent, the cleaner initializes copy/paste/blur/focus flags to zero; tab-switch still comes from Event Type.

These features are behavioral descriptors. Their proposed interpretations explain why they were considered, but do not establish that a particular feature is a causal or reliable indicator of misconduct.

## 5. Original-training audit and feature-schema decisions

### 5.1 Why a frozen schema

The extractor produces 34 raw predictors: 25 mouse, three keyboard, and six action features. Merging blur/focus/tab-switch into one window-switch count gives 32 predictors. The fixed model schema retains 25, excluding seven mouse aggregates:

- `mouse_velocity_std`, `mouse_velocity_max`.
- `mouse_acceleration_std`, `mouse_acceleration_max`.
- `mouse_jerk_std`.
- `mouse_angular_velocity_max`.
- `mouse_curvature_max`.

The retained list follows the earlier tiered-correlation-filtered training feature set. Experiment 1 **does not rerun selection separately** by recipe, augmentation state, model, or held-out performance. The original-training audit checks the fixed list's distributions and semantics; it is not a new test-informed feature selector.

This prevents a moving feature set from confounding recipe/augmentation comparisons. It does not prove that these 25 features are optimal, that correlation is eliminated, or that every retained feature contributes useful signal.

### 5.2 What the audit inspected

Only original normal-training session contents are read for the Phase 1 audit. Diagnostics include parent and micro-feature signs, nonfinite counts, zeros, constants, skew, min/median/p99/max, timing irregularities, transformation parameters, and sampled OCSVM kernel geometry. Held-out filenames may be verified against split metadata without inspecting their feature distributions.

Key training findings:

- All 25 original parent and micro source columns have zero nonfinite aggregate counts and are nonconstant in this snapshot.
- Original parent acceleration has a maximum around 8.27 million pixels/second²; mean jerk reaches about 8.42 billion pixels/second³.
- Parent keyboard-event rate reaches approximately 2,000 events/second. These finite extremes motivate investigating selective compression rather than assuming StandardScaler removes skew.
- Signed values occur in about 15.9% of parent jerk means, 40.5% of angular means, and 85.8% of angular minima.
- Copy, paste, and window-switch counts are zero in approximately 98.3%, 98.2%, and 97.6% of training parents; sparsity is greater in the micro representation.
- The 15 training files contain two duplicate mouse timestamps, 12 backward timestamp differences in original event order, and a minimum positive mouse gap of about 0.001 second. These are recorded rather than interpreted as extra signal automatically.
- Angle wrapping was made consistent for direction-change counts, sharp-angle counts, sum of angles, and angular velocity. This fixes a circular-arithmetic error while retaining existing count thresholds.

Micro and parent distributions differ even though they use the same feature names. For example, micro-windows contain fewer events, more no-mouse/no-keyboard fallbacks, and different derivative aggregates. AE preprocessing must therefore be fitted on its own micro-feature representation.

The audit is conducted once on original training. Augmented representations receive automatic validity/alignment checks, not a second feature-selection or EDA cycle.

## 6. Preprocessing recipes: what changes, where, and why

### 6.1 Common fitted pipeline and order

Every study candidate uses:

1. Fixed 25-column schema validation and canonical ordering.
2. Convert numerical +/- infinity to missing, then impute using **original normal-training medians**. Fail if a column has no finite training values.
3. Apply the selected nonlinear operations to the fixed blocks below.
4. For OCSVM/AE only, apply block-specific StandardScaler using **original-training** means and population scales.
5. Replace the direction code with nine indicator columns, in a fixed order.
6. Verify finite output and reuse the frozen fitted state for synthetic, validation, and test inputs.

Numerical validation rejects invalid nonnegative-feature values, ratios outside [0,1], fractional count values, invalid direction categories, duplicate columns, or a missing/extra source feature. These are consistency checks, not validation/test distribution fitting.

For a constant numerical training feature, StandardScaler uses scale 1. Constant continuous features bypass YJ/quantile fitting. Columns are retained rather than silently dropped. This rule is general; no original source column is constant in this run.

### 6.2 Exact transformation blocks

The continuous block has **13 features**: elapsed_time, path_length, velocity_mean, acceleration_mean, jerk_mean, angular_velocity_mean/std/min, curvature_mean/std/min, largest_deviation, and keyboard_typing_rate.

The signed subset is jerk_mean, angular_velocity_mean, and angular_velocity_min.

| Feature block | Base | Selective log | Yeo–Johnson | Quantile-to-normal |
| --- | --- | --- | --- | --- |
| Ten nonnegative continuous features | Identity | log(1+x) | Per-feature training-fitted YJ | Per-feature training-fitted empirical quantile mapping |
| Three signed continuous features | Identity | sign(x)*log(1+abs(x)) | Per-feature training-fitted YJ | Per-feature training-fitted empirical quantile mapping |
| keyboard_burst_count, keyboard_pause_count, window_switch_events | Identity | log(1+x) | log(1+x), not YJ | log(1+x), not quantile |
| copy_events, paste_events | Identity | Identity | Identity | Identity |
| mouse_direction_changes, mouse_click_count, mouse_sharp_angles | Identity | Identity | Identity | Identity |
| mouse_straightness, mouse_idle_time_ratio | Identity | Identity | Identity | Identity |
| mouse_sum_of_angles | Identity | Identity | Identity | Identity |
| mouse_direction_class | Nine indicators | Nine indicators | Nine indicators | Nine indicators |

IF leaves all numerical columns unscaled. OCSVM/AE standardize **21 of the 24 numerical columns** after the operations above; copy_events, paste_events, and window_switch_events remain unscaled. Direction indicators are also unscaled. Thus window_switch_events is logarithmically compressed in non-base recipes but is still excluded from variance scaling.

These are complete weighting/geometry recipes, not a full factorial test of every transformation/scaler combination.

### 6.3 Why these choices

**Base:** A simple reference that retains source magnitudes, with scaling where required by the model. “Base” does not mean no preprocessing: imputation and direction encoding remain, and SVM/AE also apply scaling.

**Selective log:** Compresses large finite tails while preserving zeros. The signed version retains sign instead of invalidating negative jerk/angular values. Ordinary direction/click/angle-change counts and bounded ratios are not automatically transformed merely because some distributions are skewed. The selection of transform blocks is fixed from semantics and training inspection, not repeatedly selected on validation.

**Yeo–Johnson:** Fits a per-feature power parameter lambda using original training and accepts both signs. `PowerTransformer(method="yeo-johnson", standardize=False)` is followed by the separately specified scaler. For completeness, its mathematical mapping is:

- x >= 0, lambda != 0: `((x+1)^lambda - 1) / lambda`.
- x >= 0, lambda == 0: `log(x+1)`.
- x < 0, lambda != 2: `-((1-x)^(2-lambda) - 1) / (2-lambda)`.
- x < 0, lambda == 2: `-log(1-x)`.

This can reduce skew but does not guarantee improved anomaly separation, reliable tail behavior, or a better AE reconstruction metric.

**Quantile-to-normal:** Approximately maps x through `Phi_inverse(F_training(x))`, using the original-training empirical distribution. Settings are `n_quantiles=min(100, n_original_rows)`, `output_distribution="normal"`, `subsample=None`, and seed 42. It can regularize skewed marginals, but beyond-range values map to fitted boundaries and extreme-magnitude differences may be lost. It is included for OCSVM only.

**Scaling:** OCSVM's distances/kernel width and AE's optimization/MSE are sensitive to numerical magnitude. StandardScaler puts the selected numerical block on a training-relative scale. IF's random-split trees do not require variance normalization, so its recipes are intentionally unscaled. This does not make nonlinear log transformation irrelevant to IF: it changes the random split geometry.

**Rare-event exclusions:** Variance scaling a mostly-zero count can give a single event a very large standardized magnitude. Copy/paste and window switching are therefore not scaled. This is an explicit weighting choice, not evidence of equal feature influence or an empirical finding that these counts should dominate.

**Direction encoding:** An integer code would imply arbitrary numerical distances/order between categories. Replacing one source column with nine indicators yields **24 numerical + nine categorical = 33 outputs**, rather than 25. It adds representational dimensions, not eight newly measured behaviors. The nine timesteps and nine direction categories are unrelated quantities. One-hot avoids ordinal distances but does not encode adjacent-sector circular similarity. Its influence on SVM distances and AE MSE remains part of the fixed recipe.

### 6.4 Where fitting occurs

- IF/OCSVM processors are fitted once per recipe on **662 original parent rows**.
- AE processors are fitted once per recipe on **662*9 = 5,958 original micro rows**, obtained by flattening the original sequence tensor.
- AE restores the sequence structure after transformation and applies the same fitted per-feature mapping at every timestep.
- AE does not fit a different scaler per timestep or per sequence, and does not reuse parent-level fitted statistics.
- Neither 1,324 synthetic parents nor their micro rows contribute to any preprocessing fit.
- Every candidate receives its recipe's frozen original-fitted processor; estimator training alone sees the augmentation-on/off row difference.

This is why the preprocessing comparison is not accidentally a comparison of synthetic-refitted scalers. It also means 5,958 micro rows must not be described as 5,958 independent training sessions.

### 6.5 Nine compared recipes

| Family | Study recipe | Nonlinear operations | Numerical scaling | Representation |
| --- | --- | --- | --- | --- |
| IF | IF-raw | Base | None | Parent |
| IF | IF-log | Selective log | None | Parent |
| OCSVM | SVM-base | Base | Selected-block StandardScaler | Parent |
| OCSVM | SVM-log | Selective log | Selected-block StandardScaler | Parent |
| OCSVM | SVM-YJ | YJ continuous + log-count block | Selected-block StandardScaler | Parent |
| OCSVM | SVM-quantile | Quantile continuous + log-count block | Selected-block StandardScaler | Parent |
| AE | AE-base | Base | Selected-block StandardScaler | Micro sequence |
| AE | AE-log | Selective log | Selected-block StandardScaler | Micro sequence |
| AE | AE-YJ | YJ continuous + log-count block | Selected-block StandardScaler | Micro sequence |

## 7. Paired augmentation and comparison fairness

Each original **normal-training parent** gets two synthetic copies. For each copy:

- Draw one sigma uniformly from [2,5] pixels.
- Add independent zero-mean Gaussian noise of that sigma to X and Y of mouse/click rows only.
- Clip synthetic coordinates to [0,1920] x [0,1080].
- Keep timestamps, event types, action counts, order, and annotations unchanged.
- Re-extract both parent features and all nine micro-feature vectors from that **same noisy 50-event realization**.
- Record session file, parent event offsets, copy index, and realized sigma.

The RNG seed is 42 and the realized augmentation is shared across all recipes/model seeds. There is no augmentation of normal validation, mixed validation, or test. Keyboard/action forward-filled coordinate rows are not jittered; mouse features use the actual mouse-category rows.

The intention is local coordinate perturbation rather than creating new behavioral classes or new sessions. Nevertheless, derivative/curvature features are nonlinear and can change substantially when dt is small. “Only 2–5 pixels” is not proof that every engineered-feature distribution is unchanged.

Two complementary comparisons are reported:

1. **Matched configuration:** same family, recipe, hyperparameters, main seed, and held-out samples; augmentation changes training rows. This is the cleaner intervention comparison.
2. **Independently tuned winners:** choose the best candidate within each recipe/condition separately. This measures the outcome of the complete tuned approach, not augmentation alone.

For AE, augmentation also changes the number of optimizer updates under a fixed epoch budget. At batch size 32, 662 rows give 21 updates per epoch, while 1,986 give 63. More diverse/noisy inputs and more optimization are not causally separated.

## 8. Models, tuning grids, and fixed training controls

### 8.1 Isolation Forest

IF isolates observations through random feature/split trees. More easily isolated behavior receives a larger anomaly score after negating sklearn's decision_function. The study tunes:

| Parameter | Candidates |
| --- | --- |
| Recipes | IF-raw, IF-log |
| max_samples | 128, 256 |
| max_features | 0.75, 1.0 |
| n_estimators | 100, 300, 500 |
| contamination | Fixed "auto" |
| random_state | 42 for main search |

This is 2*2*2*3 = **24 fits per augmentation condition**, 48 total. Both max_samples choices are below the 662 original parents. max_features controls the feature subset available to each tree.

Contamination controls sklearn's built-in score offset, not the external labeled-validation threshold or the ROC ranking of a fixed fitted forest. The study therefore does not pretend that contamination is the dataset's cheating prevalence.

All four IF recipe/condition winners use **256 samples, all features, and 100 trees**. This says 100 trees wins within this declared grid/split; it does not establish universal optimality or that more trees always harm performance.

### 8.2 One-Class SVM

The RBF OCSVM learns a boundary around normal training inputs. It is fitted without anomaly labels. Its score is negated so larger means more anomalous.

| Parameter | Candidates |
| --- | --- |
| Recipes | SVM-base, SVM-log, SVM-YJ, SVM-quantile |
| nu | 0.01, 0.05, 0.10 |
| gamma | 0.25, 1, or 4 times the recipe-specific original-training center |
| kernel | Fixed RBF |
| tol | Fixed 0.001 |
| max_iter | Fixed -1, no explicit iteration cap |
| Main seed metadata | 42; OCSVM itself is deterministic for this input/solver |

For encoded original-training matrix X with d=33:

`gamma_center = 1 / (d * variance(X, ddof=0))`.

Here variance is over all matrix entries, including unscaled counts and direction indicators; it is not the mean of per-column variances. The center is fixed before augmentation so gamma="scale" cannot silently recompute a different width on augmented rows.

| Recipe | Original-training gamma center | Numeric gamma candidates |
| --- | --- | --- |
| SVM-base | 0.04359031420070035 | 0.010897578550175088, 0.04359031420070035, 0.1743612568028014 |
| SVM-log | 0.0452880618000325 | 0.011322015450008126, 0.0452880618000325, 0.18115224720013 |
| SVM-YJ | 0.0452880618000325 | 0.011322015450008126, 0.0452880618000325, 0.18115224720013 |
| SVM-quantile | 0.0452880618000325 | 0.011322015450008126, 0.0452880618000325, 0.18115224720013 |

Equal centers for the three transformed/scaled recipes do not imply equal pairwise distances or equal model behavior. Training standardization can produce the same overall variance while nonlinear geometry still differs.

This is 4*3*3 = **36 fits per condition**, 72 total. Solver iteration counts and convergence flags are recorded; all converge in this run. nu is a boundary/training-error/support-vector control, not a tuned estimate of test anomaly prevalence.

### 8.3 LSTM autoencoder

Input is a nine-step sequence with 33 encoded features per step. The encoder's final-layer hidden state is the latent code; latent dropout is applied, and the code is repeated nine times as decoder input. A decoder LSTM followed by a linear output layer reconstructs the original feature sequence.

Training loss and anomaly score are the **mean squared reconstruction error across all nine timesteps and all 33 encoded columns**:

`score(parent) = sum((reconstruction - input)²) / (9*33)`.

There is no anomaly-label reconstruction target, no per-timestep anomaly classifier, and no conversion of MSE into an anomaly probability. MSE magnitudes from different preprocessing recipes are not directly comparable.

| Search block | Recipes | Hidden size | Batch | Learning rate | Encoder/decoder layers | Fits per condition |
| --- | --- | --- | --- | --- | --- | --- |
| Main | AE-base, AE-log, AE-YJ | 16, 32 | 16, 32, 64 | 0.0003, 0.001 | 1 / 1 | 36 |
| Bounded depth | Same three | 16, 32 | 32 | 0.001 | 2 / 2 | 6 |

There are **42 AE fits per condition**, 84 total. Each depth candidate has a matching one-layer configuration in the main block. Two-layer tuning is deliberately restricted rather than exhaustively expanded.

| Fixed AE control | Actual setting |
| --- | --- |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Latent dropout | 0.1 |
| Built-in encoder/decoder LSTM dropout | 0.0 at either depth |
| Max epochs | 100 |
| Patience | 10 |
| Significant normal-validation improvement | loss < accepted_best_loss * (1 - 0.001) |
| Gradient clipping | Global norm 1.0 |
| Batch handling | Shuffle training rows; keep final incomplete batch |
| Early-stopping input | Original, unaugmented pure-normal validation sequences |
| Main initialization seed | 42 |
| Finalist repeat seeds | 123, 2026 |
| Numerical tensor type | float32 |

The accepted checkpoint and patience reset use the significant-improvement rule together; the saved checkpoint is the best **accepted** normal-validation state under that tolerance, not necessarily the absolute lowest tiny loss fluctuation. This reflects the implemented early-stopping behavior.

Using pure-normal early stopping avoids rewarding reconstruction of anomalous patterns, but it still uses validation to select a checkpoint. Subsequently choosing configurations on combined validation reuses these normal sessions and introduces selection optimism; it is not an independent validation of generalization.

### 8.4 Budget and finalists

| Family/work | Fits per condition | Both conditions | Additional repeats |
| --- | --- | --- | --- |
| IF | 24 | 48 | Four: one finalist per condition, two repeat seeds |
| OCSVM | 36 | 72 | None |
| AE | 42 | 84 | Four: one finalist per condition, two repeat seeds |
| Total fitted base models | 102 | 204 | Eight |

Total base-model training runs = **212**. Ensemble weight evaluation reuses selected fitted base models and does not add base-model retraining.

Reporting retains four IF winners, eight OCSVM winners, six AE winners, and two ensembles = **20 primary approaches**. Eight repeats give **28 reported approaches**, not 28 independent experiments.

## 9. Selection, score calibration, thresholds, and metrics

### 9.1 Validation-only selection order

1. Fit each candidate on its normal training condition with the frozen original-fitted processor.
2. Score combined validation: 197 pure-normal windows followed by 196 mixed windows.
3. Select the fitted candidate with the highest **validation ROC-AUC** within each family/recipe/condition cell.
4. Keep that exact fitted detector; do not reconstruct/refit a winner from DataFrame parameter values.
5. Select the best IF and best OCSVM recipe winner within each augmentation condition for the ensemble.
6. Calibrate their scores on pure-normal validation, then select ensemble weight on combined-validation ROC-AUC.
7. Tune each selected approach's operating threshold using combined validation labels.
8. Lock finalists and run the predefined seed sensitivity without picking the best seed.
9. Report the locked approaches on test without changing configurations, weights, thresholds, or the overall validation-selected winner.

ROC-AUC ties keep the first declared candidate. AP is reported but is not the selection criterion in this experiment. Threshold metrics do not override the ROC-AUC winner.

### 9.2 Why ensemble scores need separate normalization

Raw IF and OCSVM scores have unrelated offsets and magnitudes. Averaging them directly can let one model dominate simply because it has a wider score range.

For each already-selected base model j, fit a RobustScaler to its anomaly scores on the **197 unaugmented pure-normal validation windows**:

`z_j(x) = (score_j(x) - median_normal_val_j) / IQR_normal_val_j`.

Then:

`ensemble_score(x) = weight_IF*z_IF(x) + (1-weight_IF)*z_OCSVM(x)`.

Weight candidates are **0, 0.3, 0.5, 0.7, 1.0**. Base models are not refitted per weight. Normalization is fitted once and persisted; zero/negligible IQR or nonfinite base scores is rejected. Calibration on normal validation is intended to establish a normal reference without allowing the anomaly mixture to determine score-scale statistics.

This calibration is **validation-dependent**, unlike the feature preprocessing. That is intentional model calibration, not test leakage. It relies on only four independent normal-validation sessions, despite 197 overlapping windows, and is retained as a small-data baseline rather than claimed to be statistically robust.

Both ensemble conditions select IF weight 0.7. The numerical robust scales are model-specific; “70% IF” refers to normalized scores, not raw probabilities or a 70% chance of cheating.

### 9.3 Operating threshold

For each selected score vector:

- Sweep ascending unique validation score values.
- Predict anomalous using **score >= threshold**.
- Maximize recall subject to validation precision >= 0.5.
- If no candidate satisfies the precision floor, maximize F1 instead.
- For equal recall or equal fallback F1, the first ascending threshold is retained; there is no additional precision-based tie-break.

The threshold is selected **after** ROC-based configuration/weight selection. ROC/AP are threshold-free ranking metrics; tuning a threshold does not change them. No normal-validation percentile alone determines the binary boundary.

The precision floor is not a guaranteed test precision, nor a minimum recall guarantee. SVM-YJ uses the F1 fallback in both conditions. AE-YJ satisfies the floor with extremely low recall and consequently does not invoke fallback.

Scores/thresholds need not be in [0,1]. Negative IF/OCSVM thresholds are valid because scores are not probabilities.

### 9.4 Metric definitions

- ROC-AUC measures ranking across thresholds; 0.5 is chance-level ranking under the usual binary interpretation.
- The field named `pr_auc` is sklearn **average precision (AP)**, not a trapezoidal numerical integral of a PR curve.
- The reported no-skill PR reference is anomaly prevalence: **82/393 = 0.2086514 for validation**, **205/366 = 0.5601093 for test**.
- Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 is their harmonic mean.
- FPR = FP/(FP+TN).
- Precision is reported as zero when there are no positive predictions; this is a zero-division convention, not evidence of observed false positives.
- Confusion matrices, score distributions, ROC curves, and PR curves accompany the primary results.

Higher test AP/precision than validation does not alone imply better generalization because test prevalence is much higher. ROC comparisons are also affected by actual conditional behavior/split differences, even though ROC's definition is not directly prevalence-weighted.

## 10. Complete primary results

All rows below are seed-42 recipe/condition winners or ensembles chosen using validation. **Precision, recall, F1, FPR, and confusion counts are test values at the validation-selected threshold.** Values are rounded for display; source artifacts retain full precision.

| Family | Recipe / augmentation | Val ROC | Val AP | Test ROC | Test AP | Precision | Recall | F1 | FPR | TP/FP/TN/FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| if | IF-raw / off | 0.7519 | 0.5721 | 0.7051 | 0.7566 | 0.7765 | 0.6439 | 0.7040 | 0.2360 | 132/38/123/73 |
| if | IF-log / off | 0.7618 | 0.5595 | 0.6881 | 0.7390 | 0.7791 | 0.6537 | 0.7109 | 0.2360 | 134/38/123/71 |
| ocsvm | SVM-base / off | 0.7361 | 0.4511 | 0.7205 | 0.7403 | 0.8583 | 0.5024 | 0.6338 | 0.1056 | 103/17/144/102 |
| ocsvm | SVM-log / off | 0.6549 | 0.3358 | 0.6653 | 0.7001 | 0.8293 | 0.1659 | 0.2764 | 0.0435 | 34/7/154/171 |
| ocsvm | SVM-YJ / off | 0.6379 | 0.2855 | 0.6811 | 0.7173 | 0.6784 | 0.6585 | 0.6683 | 0.3975 | 135/64/97/70 |
| ocsvm | SVM-quantile / off | 0.6433 | 0.3452 | 0.6468 | 0.7181 | 0.8269 | 0.2098 | 0.3346 | 0.0559 | 43/9/152/162 |
| ae | AE-base / off | 0.7420 | 0.5408 | 0.6368 | 0.6909 | 0.7313 | 0.4780 | 0.5782 | 0.2236 | 98/36/125/107 |
| ae | AE-log / off | 0.7317 | 0.5472 | 0.6211 | 0.6456 | 0.6846 | 0.4341 | 0.5313 | 0.2547 | 89/41/120/116 |
| ae | AE-YJ / off | 0.6554 | 0.3025 | 0.5804 | 0.6123 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/161/205 |
| if | IF-raw / on | 0.7847 | 0.6273 | 0.7138 | 0.7500 | 0.7391 | 0.6634 | 0.6992 | 0.2981 | 136/48/113/69 |
| if | IF-log / on | 0.7818 | 0.5728 | 0.7004 | 0.7370 | 0.7711 | 0.6244 | 0.6900 | 0.2360 | 128/38/123/77 |
| ocsvm | SVM-base / on | 0.7664 | 0.5134 | 0.7323 | 0.7461 | 0.7798 | 0.6390 | 0.7024 | 0.2298 | 131/37/124/74 |
| ocsvm | SVM-log / on | 0.6919 | 0.4083 | 0.6873 | 0.7164 | 0.8028 | 0.2780 | 0.4130 | 0.0870 | 57/14/147/148 |
| ocsvm | SVM-YJ / on | 0.6870 | 0.3365 | 0.7026 | 0.7313 | 0.6976 | 0.6976 | 0.6976 | 0.3851 | 143/62/99/62 |
| ocsvm | SVM-quantile / on | 0.6822 | 0.3956 | 0.6781 | 0.7475 | 0.8630 | 0.3073 | 0.4532 | 0.0621 | 63/10/151/142 |
| ae | AE-base / on | 0.7419 | 0.5598 | 0.6414 | 0.6956 | 0.7279 | 0.4829 | 0.5806 | 0.2298 | 99/37/124/106 |
| ae | AE-log / on | 0.7445 | 0.5717 | 0.6314 | 0.6607 | 0.7099 | 0.4537 | 0.5536 | 0.2360 | 93/38/123/112 |
| ae | AE-YJ / on | 0.6546 | 0.3036 | 0.5919 | 0.6196 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0/161/205 |
| ensemble | IF-log+SVM-base / off | 0.7682 | 0.5524 | 0.7015 | 0.7413 | 0.7952 | 0.6439 | 0.7116 | 0.2112 | 132/34/127/73 |
| ensemble | IF-raw+SVM-base / on | 0.7946 | 0.6158 | 0.7191 | 0.7587 | 0.7670 | 0.6585 | 0.7087 | 0.2547 | 135/41/120/70 |

The test population has 205 positives and 161 negatives for every row. TP/FP/TN/FN is explicitly listed in that order; graphical confusion matrices use the conventional [[TN,FP],[FN,TP]] layout.

### 10.1 Model-family conclusions

- **IF:** Log is the family winner without augmentation (validation 0.7618), while raw wins with augmentation (0.7847). Both unscaled recipes remain viable; log is not universally preferable.
- **OCSVM:** Base is the family winner in both conditions. Compression/YJ/quantile do not improve its validation ranking here. This does not mean transformations are universally harmful.
- **AE:** Base wins without augmentation (0.7420); log wins with augmentation (0.7445). Their test ROC-AUCs are about 0.6368 and 0.6314, respectively. The larger validation-to-test gap makes AE less convincing than the classical alternatives in this setup.
- **Ensemble:** Augmentation-on is the overall validation winner. Its test ranking is not superior to augmentation-on SVM-base, and its test F1 is slightly below the augmentation-off ensemble. Combining scores helps validation ranking but does not guarantee improvement in every held-out metric.
- **Threshold trade-offs:** SVM-base augmentation increases recall from 0.5024 to 0.6390 while FPR rises from 0.1056 to 0.2298. IF-raw improves test ROC modestly but F1 moves from 0.7040 to 0.6992. Augmentation is not a free improvement at every operating point.

### 10.2 Selected-winner figures

The selected augmentation-on ensemble has saved [ROC](../../../study/artifacts/phase_04_study/phase_06_reporting/figures/roc_ensemble_aug_on.png), [PR](../../../study/artifacts/phase_04_study/phase_06_reporting/figures/pr_ensemble_aug_on.png), [score-distribution](../../../study/artifacts/phase_04_study/phase_06_reporting/figures/scores_ensemble_aug_on.png), and [confusion-matrix](../../../study/artifacts/phase_04_study/phase_06_reporting/figures/confusion_ensemble_aug_on.png) figures. These links require the local ignored study artifacts; the numerical tables in this report remain readable without them.

## 11. Selected configurations and operating thresholds

### 11.1 Every primary selected configuration

The common family controls from Section 8 still apply. Recipes in an ensemble entry identify its already-selected IF and OCSVM components; their individual fitted parameters appear in the corresponding rows.

| Family | Recipe / augmentation | Selected parameters |
| --- | --- | --- |
| if | IF-raw / off | `{"contamination": "auto", "max_features": 1.0, "max_samples": 256, "n_estimators": 100}` |
| if | IF-log / off | `{"contamination": "auto", "max_features": 1.0, "max_samples": 256, "n_estimators": 100}` |
| ocsvm | SVM-base / off | `{"gamma": 0.04359031420070035, "kernel": "rbf", "max_iter": -1, "nu": 0.01, "tol": 0.001}` |
| ocsvm | SVM-log / off | `{"gamma": 0.18115224720013, "kernel": "rbf", "max_iter": -1, "nu": 0.05, "tol": 0.001}` |
| ocsvm | SVM-YJ / off | `{"gamma": 0.18115224720013, "kernel": "rbf", "max_iter": -1, "nu": 0.01, "tol": 0.001}` |
| ocsvm | SVM-quantile / off | `{"gamma": 0.18115224720013, "kernel": "rbf", "max_iter": -1, "nu": 0.01, "tol": 0.001}` |
| ae | AE-base / off | `{"batch_size": 32, "hidden_dim": 32, "lr": 0.001, "num_layers": 2}` |
| ae | AE-log / off | `{"batch_size": 32, "hidden_dim": 32, "lr": 0.001, "num_layers": 2}` |
| ae | AE-YJ / off | `{"batch_size": 16, "hidden_dim": 32, "lr": 0.001, "num_layers": 1}` |
| if | IF-raw / on | `{"contamination": "auto", "max_features": 1.0, "max_samples": 256, "n_estimators": 100}` |
| if | IF-log / on | `{"contamination": "auto", "max_features": 1.0, "max_samples": 256, "n_estimators": 100}` |
| ocsvm | SVM-base / on | `{"gamma": 0.04359031420070035, "kernel": "rbf", "max_iter": -1, "nu": 0.01, "tol": 0.001}` |
| ocsvm | SVM-log / on | `{"gamma": 0.18115224720013, "kernel": "rbf", "max_iter": -1, "nu": 0.01, "tol": 0.001}` |
| ocsvm | SVM-YJ / on | `{"gamma": 0.18115224720013, "kernel": "rbf", "max_iter": -1, "nu": 0.1, "tol": 0.001}` |
| ocsvm | SVM-quantile / on | `{"gamma": 0.18115224720013, "kernel": "rbf", "max_iter": -1, "nu": 0.1, "tol": 0.001}` |
| ae | AE-base / on | `{"batch_size": 32, "hidden_dim": 16, "lr": 0.001, "num_layers": 2}` |
| ae | AE-log / on | `{"batch_size": 32, "hidden_dim": 32, "lr": 0.001, "num_layers": 2}` |
| ae | AE-YJ / on | `{"batch_size": 32, "hidden_dim": 32, "lr": 0.001, "num_layers": 1}` |
| ensemble | IF-log+SVM-base / off | `{"if_weight": 0.7, "ocsvm_weight": 0.30000000000000004}` |
| ensemble | IF-raw+SVM-base / on | `{"if_weight": 0.7, "ocsvm_weight": 0.30000000000000004}` |

The serialized value 0.30000000000000004 is ordinary floating-point representation of 1-0.7, not a different intended weight.

### 11.2 Every primary operating threshold and its validation behavior

Use the full-precision manifest threshold for inference. The six-decimal display here is not an instruction to round thresholds before comparison.

| Family | Recipe / augmentation | Threshold | Val precision | Val recall | F1 fallback |
| --- | --- | --- | --- | --- | --- |
| if | IF-raw / off | -0.039540 | 0.5000 | 0.6098 | false |
| if | IF-log / off | -0.019620 | 0.5000 | 0.6098 | false |
| ocsvm | SVM-base / off | 0.033057 | 0.5000 | 0.4268 | false |
| ocsvm | SVM-log / off | 0.293685 | 0.5000 | 0.1463 | false |
| ocsvm | SVM-YJ / off | 0.016016 | 0.3289 | 0.6098 | true |
| ocsvm | SVM-quantile / off | 0.057937 | 0.5000 | 0.1585 | false |
| ae | AE-base / off | 0.433311 | 0.5000 | 0.5366 | false |
| ae | AE-log / off | 0.417765 | 0.5000 | 0.5122 | false |
| ae | AE-YJ / off | 19.097294 | 0.5000 | 0.0122 | false |
| if | IF-raw / on | -0.062151 | 0.5000 | 0.6707 | false |
| if | IF-log / on | -0.023537 | 0.5000 | 0.6220 | false |
| ocsvm | SVM-base / on | -0.019055 | 0.5000 | 0.6098 | false |
| ocsvm | SVM-log / on | 0.066659 | 0.5000 | 0.2683 | false |
| ocsvm | SVM-YJ / on | 0.100325 | 0.3642 | 0.6707 | true |
| ocsvm | SVM-quantile / on | 0.824314 | 0.5000 | 0.1951 | false |
| ae | AE-base / on | 0.397841 | 0.5000 | 0.5732 | false |
| ae | AE-log / on | 0.360546 | 0.5000 | 0.5732 | false |
| ae | AE-YJ / on | 17.166752 | 0.5000 | 0.0122 | false |
| ensemble | IF-log+SVM-base / off | 0.712850 | 0.5000 | 0.5854 | false |
| ensemble | IF-raw+SVM-base / on | 0.658771 | 0.5000 | 0.6585 | false |

SVM-YJ fallback precision is below 0.5 because the floor could not be met; the fallback is explicitly reported rather than presented as floor-compliant. AE-YJ's 0.0122 recall is approximately one of 82 validation anomalies, despite meeting the precision floor.

## 12. Augmentation, depth, runtime, and seed observations

### 12.1 Matched augmentation

The following deltas are **augmentation on minus off on validation**, at identical hyperparameters and recipe. They summarize the saved 102 matched pairs, not independently tuned winners.

| Family | Recipe | Matched pairs | Positive ROC deltas | Median ROC delta | ROC delta range | Median AP delta |
| --- | --- | --- | --- | --- | --- | --- |
| ae | AE-YJ | 14 | 13 | +0.0115 | -0.0031 to +0.0310 | +0.0115 |
| ae | AE-base | 14 | 12 | +0.0039 | -0.0027 to +0.0093 | +0.0148 |
| ae | AE-log | 14 | 14 | +0.0113 | +0.0058 to +0.0271 | +0.0164 |
| if | IF-log | 12 | 9 | +0.0133 | -0.0049 to +0.0335 | +0.0013 |
| if | IF-raw | 12 | 12 | +0.0229 | +0.0061 to +0.0393 | +0.0494 |
| ocsvm | SVM-YJ | 9 | 6 | +0.0247 | -0.0208 to +0.0494 | +0.0381 |
| ocsvm | SVM-base | 9 | 6 | +0.0242 | -0.0172 to +0.0303 | +0.0406 |
| ocsvm | SVM-log | 9 | 6 | +0.0174 | -0.0224 to +0.0391 | +0.0439 |
| ocsvm | SVM-quantile | 9 | 8 | +0.0392 | -0.0135 to +0.0592 | +0.0503 |

**86 of 102 pairs improve validation ROC-AUC.** IF-raw improves in all 12 pairs; AE-log in all 14. Positive medians for every recipe suggest a broad, modest benefit within this dataset rather than an improvement restricted to one selected winner. These pairs are correlated evaluations on the same validation sessions, not independent trials supporting a significance claim.

### 12.2 Independently tuned recipe winners

Winner-level validation ROC changes:

| Recipe | Off ROC | On ROC | On minus off |
| --- | --- | --- | --- |
| IF-raw | 0.7519 | 0.7847 | +0.0328 |
| IF-log | 0.7618 | 0.7818 | +0.0200 |
| SVM-base | 0.7361 | 0.7664 | +0.0303 |
| SVM-log | 0.6549 | 0.6919 | +0.0370 |
| SVM-YJ | 0.6379 | 0.6870 | +0.0491 |
| SVM-quantile | 0.6433 | 0.6822 | +0.0389 |
| AE-base | 0.7420 | 0.7419 | -0.0001 |
| AE-log | 0.7317 | 0.7445 | +0.0127 |
| AE-YJ | 0.6554 | 0.6546 | -0.0007 |

AE-YJ can improve in most matched configurations while its independently selected maximum changes slightly downward. These statements are compatible: per-configuration deltas and the difference between two maxima answer different questions.

For IF and AE, the overall family winner may also change recipe between augmentation conditions. Do not attribute that whole winner-to-winner change solely to augmentation.

### 12.3 Bounded AE depth comparison

These are matched two-layer minus one-layer validation ROC-AUC changes at batch 32/lr 0.001, for hidden sizes 16 and 32 in both augmentation conditions.

| Recipe | Pairs | Positive ROC deltas | Median ROC delta | ROC delta range |
| --- | --- | --- | --- | --- |
| AE-YJ | 4 | 2 | +0.0014 | -0.0021 to +0.0042 |
| AE-base | 4 | 4 | +0.0082 | +0.0045 to +0.0130 |
| AE-log | 4 | 3 | +0.0036 | -0.0042 to +0.0107 |

Nine of 12 pairs improve. Base gains are consistent but small; log has one negative pair; YJ is mixed. All base/log recipe winners use two layers, while YJ selects one layer. There is no evidence here that two layers universally improve temporal anomaly learning.

### 12.4 Epoch cap and optimizer budget

| Recipe | Augmentation | Fits | At 100 epochs | Epoch range |
| --- | --- | --- | --- | --- |
| AE-YJ | off | 14 | 14 | 100–100 |
| AE-YJ | on | 14 | 13 | 93–100 |
| AE-base | off | 14 | 13 | 97–100 |
| AE-base | on | 14 | 10 | 80–100 |
| AE-log | off | 14 | 13 | 78–100 |
| AE-log | on | 14 | 14 | 100–100 |

**77 of 84 main AE fits reach 100 epochs.** The remaining seven train for 78–97 epochs. Reaching the cap is not the same as triggering early stopping.

At batch 32 and 100 epochs, augmentation off commonly yields 2,100 updates and augmentation on 6,300. This extra optimization is part of the tested augmentation-on approach. It prevents a pure “same-compute additional-data diversity” interpretation.

The sum of candidate fit-and-validation-score/metric timings is **2189.89 seconds (36.50 minutes)**; AE accounts for **2140.54 seconds (35.68 minutes), approximately 97.7%**. These sums exclude dataset preparation, repeats, ensemble/threshold work, plotting, and other orchestration. They are not an end-to-end wall-clock measurement.

Cap-hitting and comparatively weak test ranking justify describing AE as performance **under the declared training budget**. They do not prove undertraining or justify extending epochs after inspecting test. Further budget/architecture comparisons would belong to a separately documented follow-up.

### 12.5 Seed sensitivity: all eight repeat rows

The finalists are chosen once from seed-42 validation results: IF-log/off, IF-raw/on, AE-base/off, and AE-log/on. Only initialization changes to seeds 123 and 2026. The prepared bundle, augmentation realizations, preprocessing, recipe, and model hyperparameters stay fixed. Each repeat uses its own normal-validation checkpoint and validation-selected threshold; no “best seed” replaces seed 42.

| Family | Recipe / augmentation | Seed | Val ROC | Val AP | Test ROC | Test AP | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| if | IF-log / off | 123 | 0.7462 | 0.5427 | 0.6916 | 0.7397 | 0.7778 | 0.5463 | 0.6418 |
| if | IF-log / off | 2026 | 0.7487 | 0.4979 | 0.7005 | 0.7478 | 0.7853 | 0.6244 | 0.6957 |
| ae | AE-base / off | 123 | 0.7395 | 0.5469 | 0.6385 | 0.6930 | 0.7259 | 0.4780 | 0.5765 |
| ae | AE-base / off | 2026 | 0.7411 | 0.5404 | 0.6364 | 0.6920 | 0.7218 | 0.4683 | 0.5680 |
| if | IF-raw / on | 123 | 0.7557 | 0.5936 | 0.6997 | 0.7509 | 0.7945 | 0.5659 | 0.6610 |
| if | IF-raw / on | 2026 | 0.7739 | 0.5880 | 0.7034 | 0.7432 | 0.7590 | 0.6146 | 0.6792 |
| ae | AE-log / on | 123 | 0.7401 | 0.5631 | 0.6321 | 0.6717 | 0.7059 | 0.4683 | 0.5630 |
| ae | AE-log / on | 2026 | 0.7387 | 0.5623 | 0.6328 | 0.6716 | 0.7132 | 0.4488 | 0.5509 |

Including their primary seed-42 row:

- IF-log/off test ROC ranges 0.6881–0.7005; recall ranges 0.5463–0.6537.
- IF-raw/on test ROC ranges 0.6997–0.7138; recall ranges 0.5659–0.6634.
- AE-base/off test ROC ranges 0.6364–0.6385.
- AE-log/on test ROC ranges 0.6314–0.6328.

AE's weaker ranking is quite consistent across these three seeds. IF ranking varies modestly, while thresholded recall changes by roughly ten percentage points. This tests model-initialization sensitivity, not independence from the data split or robustness on new students. OCSVM and ensembles are not re-tuned for each repeat seed.

## 13. AE-Yeo–Johnson threshold diagnostic

Both AE-YJ primary approaches predict every test window as normal. Yet their test ROC-AUCs are about 0.5804/0.5919 and AP about 0.6123/0.6196: weak but nonconstant ranking can coexist with zero recall at a transferred threshold.

A read-only diagnostic reloaded the saved processors/models, transformed the same prepared validation/test sequences, and used the **exact threshold from reporting_manifest.json**, without fitting anything:

| Condition | Exact stored threshold | Validation TP | Validation FP | Validation recall | Maximum test score | Test TP / FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| Off | 19.097293853759766 | 1 | 1 | 1/82 = 0.0122 | 8.469143867492676 | 0 / 0 / 205 |
| On | 17.166751861572266 | 1 | 1 | 1/82 = 0.0122 | 8.301196098327637 | 0 / 0 / 205 |

The precision-floor rule accepts two validation predictions, one anomalous and one normal. It therefore attains precision 0.5, avoids the F1 fallback, and retains an extremely high threshold with almost no recall. All test scores are below it.

This is an observed **operating-threshold transfer failure**, not evidence that test annotations were used for training or that the model emitted NaNs. The same two-point decision is a fragile basis for a practical threshold, particularly because overlapping windows are dependent.

**Precision correction to the earlier discussion:** both conditions have one validation false positive when the exact manifest threshold is used. A previous CSV-based diagnostic reported none for augmentation off because of boundary floating-point parsing. The authoritative JSON-threshold diagnostic agrees with the stored validation precision of 0.5. This does not change the all-normal test result.

The exact feature responsible for the extreme validation tail was not established here. YJ includes strongly fitted powers on sparse continuous features, and normalizing training marginals does not guarantee useful reconstruction-error ordering on held-out anomalies. Do not claim a specific causal feature without a separate diagnostic.

The failure stays in Experiment 1's results. Lowering this threshold to improve known test recall would be test-informed tuning, not valid completion of the frozen comparison.

## 14. Interpretation, limitations, and engineering caveats

### 14.1 Supported conclusions

- Normal-only fitted detectors can rank the annotated behaviors better than chance in this split, with classical models around test ROC 0.69–0.73 for the validation-selected family finalists/ensembles.
- Feature distribution/weighting choices materially affect performance. SVM-base beats its transformed alternatives on validation, while AE's preferred recipe depends on augmentation state.
- Augmentation usually improves validation ranking within matched settings, with small or inconsistent gains in some operational/test metrics.
- Bounded depth gains and seed stability do not rescue AE's weaker generalization.
- Threshold-free ranking and a practically useful operating threshold are different properties; AE-YJ demonstrates that distinction.

These are descriptive proof-of-concept findings, not statistical-significance or deployment-safety claims.

### 14.2 Statistical and experimental limitations

1. **Few independent sessions:** normal-validation calibration/early stopping uses four sessions, mixed validation five, test ten contributing sessions. Overlap and synthetic duplication do not create independent participants.
2. **Student overlap:** file-level disjointness does not establish unseen-person performance. Some normal/mixed recordings correspond to the same participant.
3. **Different validation/test compositions:** validation combines pure-normal and mixed sessions; test is mixed-only. Prevalence and conditional behavior differ.
4. **Selection optimism:** all grids, recipes, weights, checkpoints, and thresholds reuse the same small validation partition. Reported validation maxima are not unbiased generalization estimates.
5. **Historical exploration:** earlier validation ablations informed design, and test reporting existed during previous development. Experiment 1 should be described as exploratory, not a newly untouched confirmatory holdout.
6. **Whole-recipe comparisons:** nonlinear operations, count treatment, and block scaling are not independently factorialized. A winning recipe does not identify one causally beneficial transformation.
7. **Unequal searches:** IF/OCSVM/AE have 48/72/84 main fits; greater search opportunity can affect validation maxima.
8. **AE augmentation compute:** more rows mean more updates; augmentation diversity and extra optimization are confounded.
9. **Event-count windows:** physical window durations vary. Metrics assess the >=50% event-annotation label, not arbitrary real-time response latency or detection of every short anomalous segment.
10. **Feature semantics/fallbacks:** sparse data creates zero aggregates; lone clicks are undercounted; “sharp angles,” typing rate, and idle ratio have the exact proxy definitions above.
11. **No independent uncertainty estimate:** no session-bootstrap interval, repeated group split, nested validation, or unseen-student evaluation is included.
12. **Anomaly is not misconduct:** legitimate clipboard use, navigation, rapid/irregular movement, accessibility needs, or device differences can affect signals. A model prediction alone is not a basis for disciplinary judgment.

### 14.3 Leakage boundaries and remaining implementation caveats

The implemented study fits feature preprocessing only on original training; normal validation is intentionally used for AE checkpoint selection and ensemble score calibration; combined validation is used for candidate/weight/threshold selection. Test is evaluated by the separate reporting module, not supplied to the search objective.

However, code conventions and a “frozen” flag do not make the artifacts tamper-proof:

- Final reporting currently verifies prepared-manifest **version**, not the full prepared-bundle identity/digest.
- Locked directories/artifacts can still be overwritten by a subsequent search/repeat/report invocation.
- Reporting snapshots do not bind all model-source hashes, full run configuration identity, and fitted-artifact hashes together.
- Current candidate handling records fit/score-call exceptions and missing cells, but metric calculation remains outside that exception handler: a returned nonfinite score vector can still abort the sweep rather than be recorded as a failed candidate. All 204 candidates are successful in this snapshot, so this is a failure-handling limitation rather than an observed failed run.
- Default detector repeated fit still treats an internally fitted preprocessor as frozen; fresh study candidates avoid that path, but ordinary reusable-model refit semantics remain a separate concern.
- Bundle construction uses main_seed for augmentation rather than the declared aug_seed field. Both are 42 here; implemented repeats reuse the bundle and therefore keep actual noise fixed.
- Older brief/inventory status text may lag the executed artifacts. This report distinguishes saved execution evidence from those planning labels.

These caveats limit the strength of “fully frozen/reproducible” claims. They are not evidence that the recorded numerical run necessarily used the wrong split. This report records the exact artifact hashes available now; it does not silently modify models or declare those safeguards repaired.

## 15. Artifact inventory, provenance, and reproduction

### 15.1 Source-of-truth hierarchy

For **reported values**, use saved full-precision run/test artifacts. For **implemented calculations**, use the source modules. Planning prose explains intent but is not sufficient evidence of completed execution.

All artifact paths below are relative to the repository root:

| Artifact/source | Role |
| --- | --- |
| `study/phase3.log` | Preparation/cleaning log, including zero-event files |
| `study/phase4.log` | Per-cell search/winner and ensemble messages |
| `study/phase5.log` | Eight repeat artifacts recorded |
| `study/phase6.log` | 28 reported approaches recorded |
| `study/artifacts/phase_01_audit/` | Original-only feature/timing/transform/kernel diagnostics |
| `study/artifacts/phase_02_readiness/` | Earlier readiness bundle, not the main search's authoritative bundle |
| `study/artifacts/phase_03_study/manifest.json` | Main prepared-bundle raw/source hashes, ordered split, geometry, augmentation, package versions |
| `study/artifacts/phase_03_study/preprocessing.json` | Fitted recipe summaries and gamma centers |
| `study/artifacts/phase_03_study/session_readiness.csv` | All assigned files and inclusion/exclusion reasons |
| `study/artifacts/phase_03_study/prepared.joblib` | Paired representations and frozen fitted processors |
| `study/artifacts/phase_04_study/candidates.csv` | All 204 candidate configurations, validation ROC/AP, timing, status, diagnostics |
| `study/artifacts/phase_04_study/reporting_manifest.json` | Primary/repeat identities, fitted artifact locations, selected parameters, thresholds, validation values |
| `study/artifacts/phase_04_study/augmentation_matched.csv` | 102 matched augmentation comparisons |
| `study/artifacts/phase_04_study/augmentation_winners.csv` | Nine independently tuned recipe comparisons |
| `study/artifacts/phase_04_study/depth_comparison.csv` | 12 matched AE depth comparisons |
| `study/artifacts/phase_04_study/winner_*.joblib` | Original fitted primary models/ensembles |
| `study/artifacts/phase_04_study/phase_05_seed_sensitivity/` | Eight fitted repeat models |
| `study/artifacts/phase_04_study/phase_06_reporting/test_results.csv` | All 28 test rows, confusion counts, fitted parameters, prevalence |
| `study/artifacts/phase_04_study/phase_06_reporting/test_results.json` | JSON scalar test results |
| `study/artifacts/phase_04_study/phase_06_reporting/dataset_summary.csv` | Partition session/window/prevalence counts |
| `study/artifacts/phase_04_study/phase_06_reporting/figures/` | 32 grouped primary figures: ROC, PR, score distributions, confusion matrices |

The saved JSON/CSV results omit raw score vectors, while fitted models and the prepared bundle support read-only recomputation. Comparison CSVs summarize validation; final figures summarize primary test approaches. Sensitivity rows remain in the table rather than being averaged into confusion matrices.

### 15.2 Important implementation entry points

- `src/cheatdetect/data/cleaning.py`: raw event coercion/flags.
- `src/cheatdetect/data/features/{mouse,keyboard,actions,extract}.py`: feature formulas and labels.
- `src/cheatdetect/data/feature_schema.py`: fixed order and transform/scaling blocks.
- `src/cheatdetect/data/preprocessing.py`: fitted feature pipeline and gamma center.
- `src/cheatdetect/data/{paired,dataset}.py`: shared realized parent/micro data, frozen processors, manifests.
- `src/cheatdetect/models/{isolation_forest,ocsvm,lstm_ae,ensemble,threshold,selection}.py`: models, ordered search, calibration, and thresholds.
- `src/cheatdetect/experiments/experiment_1/{audit,config,runner,analysis,reporting}.py`: audit, declared study, search/repeats, comparisons, final reports.
- `src/cheatdetect/experiments/experiment_1/notebooks/`: research execution/reporting notebooks.
- `src/cheatdetect/pipeline/{train,search,report}.py`: production orchestration using shared preparation/selection/reporting components.

No new production promotion, model fit, protocol change, notebook edit, or Git commit is performed to create this report.

### 15.3 Snapshot checksums

SHA-256 hashes at report creation:

- `study/artifacts/phase_03_study/manifest.json`: `b58959842cb2ae41b9a7f39c9262d196de2fea9adb5ff621bc8bfc5e31a39c52`
- `study/artifacts/phase_04_study/reporting_manifest.json`: `67bbee56063c5bb2316d27f5fededeeace52d46cb8df48afc77747ee827fecb4`
- `study/artifacts/phase_04_study/candidates.csv`: `531950816bc5a10f55e22b5126cba22a81531ddef915d43a714a36e964540602`
- `study/artifacts/phase_04_study/phase_06_reporting/test_results.csv`: `e40f8b61070387b33a4e758be35ab7b06f3a38c0b4746a5179cace81bfdab86d`

The prepared manifest records NumPy **2.4.6**, pandas **3.0.3**, and scikit-learn **1.9.0**. Package versions listed here come from that saved manifest, not assumed current environment versions. It does not record a complete Python/PyTorch/hardware execution environment, so byte-identical reruns are not guaranteed.

The report-generation HEAD identifies repository context, not a guaranteed clean commit containing every change used in the numerical run. Raw session files and generated `study/` artifacts are ignored locally; the report alone cannot reconstruct private raw events or prove raw artifact availability in a clean clone.

### 15.4 Reproduction sequence and safe use

With the original data available, the project environment/dependencies installed, and the original audit split metadata available:

```bash
uv run python -m cheatdetect.experiments.experiment_1.audit
uv run python -m cheatdetect.experiments --phase 3
uv run python -m cheatdetect.experiments --phase 4 --run-dir study/artifacts/experiment_1_reproduction
uv run python -m cheatdetect.experiments --phase 5 --run-dir study/artifacts/experiment_1_reproduction
uv run python -m cheatdetect.experiments --phase 6 --run-dir study/artifacts/experiment_1_reproduction
```

These are documentation commands, not commands executed to write this report. Phases 4–6 use a fresh run directory to avoid overwriting the original result set. Audit and Phase 3 still use their configured default artifact directories: preserve the original artifacts first or reproduce in a separate checkout/environment.

The audit currently depends on existing ignored `dataset/processed/split_info.json`, verifies the seed-42 assignments against it, and produces the audit manifest used by the default bundle path. Supplying data without those prerequisite assignments is not a self-contained clean-clone bootstrap. Do not bootstrap it by accidentally running a train-and-test workflow merely to obtain split metadata.

AE training and loading AE artifacts require the research dependency PyTorch. Lazy imports keep unrelated paths lightweight; they do not make the full default Phase 4 search or AE reporting runnable without PyTorch.

Readiness reuse is guarded by freshly computed prepared manifests. Changes to source hashes/raw files can intentionally invalidate the original cache. Regeneration is a new artifact snapshot; never relabel changed outputs as byte-identical reproductions of the checksums above. Only load trusted local joblib files.

Regression coverage exists for original-only fitting/audit reads, paired noise identity, schema/recipe correctness, injected processors, ordered selection, model controls/convergence, repeats, CLI routing, analysis tables, and reporting. Test success supports implementation checks, not statistical validity. No additional test run is required merely to add this documentation report.

## 16. Final study statement

**Experiment 1 demonstrates that the available behavioral data contains a modest learnable anomaly signal under normal-only training and session-file-held-out evaluation. Classical models are more convincing than the tested AE. Small coordinate augmentation usually improves validation ranking, but can increase false positives, and its AE effect includes additional optimizer work. Limited depth changes yield modest gains; seed repeats suggest AE weakness is consistent rather than one unlucky initialization.**

**The overall winner remains the augmentation-on IF-raw/SVM-base ensemble selected by validation ROC-AUC, with test ROC-AUC 0.7191, AP 0.7587, precision 0.7670, recall 0.6585, and F1 0.7087.** The stronger observed OCSVM test ranking is reported, not used to rewrite the winner.

**The study is exploratory, small-session, and threshold-sensitive. It establishes feasibility, not reliable misconduct determination or unseen-student performance.** AE-YJ's zero test recall, validation/test prevalence differences, dependence among windows, and incomplete freezing safeguards are part of the whole picture rather than exceptions to hide. Any future protocol/threshold/budget changes should be identified as a new experiment rather than a post-test repair of this one.

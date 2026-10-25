> **SUPERSEDED.** Archived for provenance only. The canonical Experiment 1
> report is `../03_experiments_and_models.md` (preprocessing and trials) and
> `../04_selected_models_and_conclusions.md` (final selections).

# Experiment 1 — Preprocessing rationale and trial reference

**Date:** 2026-10-10  
**Purpose:** Explain why the feature types and the three models led to the preprocessing trials, and preserve the exact recipes/configurations that were actually tested.  
**Reading guide:** Section 1 is the recipe lookup, Sections 2–7 explain the decisions, Section 8 lists every search axis, and Section 9 records the selected configurations and observations.  
**Companion:** The [complete Experiment 1 report](experiment_1_summary.md) covers the dataset, feature calculations, full metrics, thresholds, and execution/provenance limitations.

This is a focused transformation/scaling reference, not a second full pipeline report. The reasons below distinguish **a defensible reason to test an option** from **evidence that the option helped**. A more symmetric training distribution is not automatically a better anomaly-detection representation.

## 1. Quick lookup: what was actually compared?

All recipes use the same frozen **25 source features**. Direction encoding replaces one categorical code with nine indicators, so all models operate on **33 encoded columns**. The source feature list is not reselected for each recipe.

| Model | Recipe | Continuous-feature transformation | Selected count transformation | Feature scaling | Main fits per recipe, off + on |
| --- | --- | --- | --- | --- | --- |
| Isolation Forest | IF-raw | None | None | None | 12 + 12 |
| Isolation Forest | IF-log | Selective ordinary/signed log | log1p on the fixed log-count block | None | 12 + 12 |
| One-Class SVM | SVM-base | None | None | Block StandardScaler | 9 + 9 |
| One-Class SVM | SVM-log | Selective ordinary/signed log | log1p on the fixed log-count block | Block StandardScaler | 9 + 9 |
| One-Class SVM | SVM-YJ | Training-fitted Yeo–Johnson | log1p on the fixed log-count block | Block StandardScaler | 9 + 9 |
| One-Class SVM | SVM-quantile | Training-fitted quantile-to-normal | log1p on the fixed log-count block | Block StandardScaler | 9 + 9 |
| LSTM autoencoder | AE-base | None | None | Block StandardScaler | 14 + 14 |
| LSTM autoencoder | AE-log | Selective ordinary/signed log | log1p on the fixed log-count block | Block StandardScaler | 14 + 14 |
| LSTM autoencoder | AE-YJ | Training-fitted Yeo–Johnson | log1p on the fixed log-count block | Block StandardScaler | 14 + 14 |

“Block StandardScaler” means standardize **21 numerical features**, but not copy_events, paste_events, window_switch_events, or the nine direction indicators. A transformed window-switch count is still **not standardized**.

“Selected counts” means keyboard_burst_count, keyboard_pause_count, and window_switch_events. YJ/quantile recipes use **log1p**, not YJ/quantile, for those counts.

“Raw”/“base” do not have the same scaling meaning across models: **IF-raw is unscaled; SVM-base and AE-base are scaled.** “No continuous transformation” means no nonlinear transformation, not identical end-to-end preprocessing.

The experiment is nine whole recipes under augmentation off/on: **204 main fits**, plus eight locked-finalist seed repeats. It is not a grid over every scaler and every feature independently.

## 2. What in these features motivated preprocessing?

The dataset mixes quantities with very different meanings and numerical behavior:

- Distances in pixels, rates in pixels/second, and higher-order derivatives in pixels/second² or pixels/second³.
- Signed turning-rate and jerk statistics.
- Bounded ratios such as straightness.
- Discrete event counts, including mostly-zero rare events.
- A categorical direction code that should not be treated as an ordinary magnitude.

The original-normal-training audit provides concrete reasons to investigate both **shape transformation** and **scale normalization**:

| Original parent feature observation | Why it matters |
| --- | --- |
| mouse_acceleration_mean median about 3,459; maximum about 8.27 million; skew about 25.6 | A few large derivative values can occupy most of the numerical range |
| mouse_jerk_mean maximum about 8.42 billion; skew about 25.7; 15.9% negative values | Large tails require sign-aware treatment; ordinary log is not appropriate across the entire domain |
| keyboard_typing_rate maximum about 2,000; skew about 17.4 | Rate features can be extreme when event gaps are very short |
| mouse_angular_velocity_mean is negative in about 40.5% of parents | Negative values are genuine signed behavior, not values to discard or shift arbitrarily |
| mouse_angular_velocity_min is negative in about 85.8% of parents | A predominantly negative distribution is not a positive-only log problem |
| copy_events, paste_events, window_switch_events are zero in about 98.3%, 98.2%, 97.6% of parents | Standardization can strongly amplify an otherwise small rare count |
| mouse_curvature_min is zero in about 83.4% of parents | A nominally continuous feature can still have a substantial point mass at zero |

These observations come from **training**, not test-driven transform selection. Micro-window distributions differ: acceleration/jerk skew is even stronger, and clipboard/window-switch counts are more often zero. Therefore, the AE fits its transformations on micro-features rather than borrowing fitted parent-feature parameters.

A derivative can become very large when its time denominator is small. Such a value can be finite and valid under the extractor without necessarily being useful behavioral signal. Compression is a hypothesis about managing its influence, not proof that the largest observed values are recording errors.

### 2.1 Scaling and distribution transformation are different jobs

**Scaling** asks: “How much influence does this feature's numerical size have relative to other features?”

**A nonlinear transformation** asks: “How should differences within this feature be represented—for example, a change from 1 to 10 versus 1,000 to 10,000?”

Standardizing a long-tailed feature does not remove its skew. Logging a feature does not, on its own, give it the same variance as other features. This is why the SVM/AE trials apply the nonlinear step **before** the common block scaler.

## 3. Exact feature-to-transformation map

### 3.1 Continuous block: 13 features

For this entire block:

- Base: retain the numerical value.
- Log: apply log1p to nonnegative features, signed log1p to the three signed features.
- YJ: fit one Yeo–Johnson parameter per feature on original normal training.
- Quantile: fit one empirical quantile mapping per feature on original normal training.
- IF: no subsequent StandardScaler.
- SVM/AE: subsequent StandardScaler.

| Feature | Domain/type relevant to preprocessing | Why it belongs in the continuous trial block |
| --- | --- | --- |
| `elapsed_time` | Nonnegative duration | Variable time for a fixed event count; a potential right tail |
| `mouse_path_length` | Nonnegative distance | Large range of cursor-travel magnitudes |
| `mouse_velocity_mean` | Nonnegative rate | Speed can have a long upper tail |
| `mouse_acceleration_mean` | Nonnegative derivative magnitude | Particularly large derivative tails in the audit |
| `mouse_jerk_mean` | **Signed** derivative of acceleration magnitude | Negative and positive tails; sign must be retained |
| `mouse_angular_velocity_mean` | **Signed** turning rate | Directional sign and potentially asymmetric tails |
| `mouse_angular_velocity_std` | Nonnegative variability of turning rate | A magnitude/dispersion feature, unlike signed angular mean |
| `mouse_angular_velocity_min` | **Signed** minimum turning rate | Often negative; ordinary positive-only log is unsuitable |
| `mouse_curvature_mean` | Nonnegative geometric bending | Strong tail and heterogeneous magnitudes |
| `mouse_curvature_std` | Nonnegative variability of bending | Strong variability/tail behavior |
| `mouse_curvature_min` | Nonnegative, strongly zero-heavy | Included by the fixed continuous block, despite sparse positive values |
| `mouse_largest_deviation` | Nonnegative distance from the start-to-end line | Geometric magnitude, with different parent/micro ranges |
| `keyboard_typing_rate` | Nonnegative keyboard-category event rate | Short time gaps can produce very large rates |

This is a **fixed semantically chosen block**, not a rule that logs every feature whose measured skew happens to exceed a threshold. It contains features with milder skew too; the whole-recipe comparison tests this common block consistently.

### 3.2 Other source features: exact exceptions

The StandardScaler column applies to SVM/AE; IF leaves every numerical feature unscaled.

| Feature | Nonlinear step in base | Nonlinear step in log/YJ/quantile | SVM/AE StandardScaler? | Reason for this treatment |
| --- | --- | --- | --- | --- |
| `copy_events` | None | None | No | Very rare count; preserve small count magnitude instead of variance amplification |
| `paste_events` | None | None | No | Same rare-event weighting choice |
| `window_switch_events` | None | log1p | No | Compress occasional larger totals, but avoid amplifying sparse occurrence through scaling |
| `keyboard_burst_count` | None | log1p | Yes | Count with a useful range; compress larger totals before common scaling |
| `keyboard_pause_count` | None | log1p | Yes | Same count-compression trial, distinct from a rate feature |
| `mouse_direction_changes` | None | None | Yes | Keep the existing interpretable turn-count magnitude; no separate nonlinear-count trial |
| `mouse_click_count` | None | None | Yes | Keep count differences; not placed in the extremely rare clipboard block |
| `mouse_sharp_angles` | None | None | Yes | Existing turn-change count, rather than a continuous angular rate |
| `mouse_straightness` | None | None | Yes | Bounded [0,1] ratio; no need to treat its source range like a huge derivative tail |
| `mouse_idle_time_ratio` | None | None | Yes | Bounded [0,1] ratio; common scaling controls its relative influence |
| `mouse_sum_of_angles` | None | None | Yes | Aggregate turning amount left outside the selective continuous transform block |
| `mouse_direction_class` | Nine indicators | Nine indicators | No | Categorical representation, not a continuous or ordinal magnitude |

These exceptions are **design/weighting choices**, not separate conclusions from per-feature ablation. The experiment did not establish that logging click counts, using a different rare-event weight, or transforming ratios would be worse.

A source ratio remains bounded, but its standardized representation need not remain in [0,1]. That is expected: the standardized value is a training-relative deviation, not the original ratio.

## 4. What each transformation means—and why it was tried

### 4.1 Base: the necessary control

Base leaves source magnitudes unchanged before the model's assigned scaling. It asks whether **shape-changing transformation is needed at all**.

Without a base trial, improvement from a complicated recipe could not be compared with a simple common-scaling reference. Base also preserves absolute within-feature gaps that a nonlinear transform might compress.

This matters for anomaly detection: an unusually large magnitude may be useful signal, not something to automatically make less extreme.

### 4.2 Selective log: a fixed compression hypothesis

For a nonnegative continuous feature:

`transformed = log(1 + x)`.

For a signed continuous feature:

`transformed = sign(x) * log(1 + abs(x))`.

The added 1 preserves zero and makes the function well-defined at zero. Signed log preserves negative/positive direction and compresses both tails with the same fixed rule.

A positive example illustrates the change in geometry:

| Input | log1p(input), approximately |
| --- | --- |
| 0 | 0 |
| 10 | 2.398 |
| 1,000 | 6.909 |
| 1,000,000 | 13.816 |

A million-valued derivative no longer dwarfs a thousand-valued derivative by a factor of 1,000 in the transformed coordinate. This can make differences in the ordinary range more visible to a distance/MSE-based model.

**Why try it here?** The training audit shows strong derivative/rate tails, including valid negative jerk/angular values. Log is a simple reproducible comparison with no learned transformation-strength parameter.

**Why might it hurt?** It compresses large anomaly magnitudes too. If unusually large movement is the discriminating signal, the model can lose useful separation. IF's randomized split positions also change under log, so monotonic order preservation does not guarantee identical forests or ROC-AUC.

The count block gets ordinary log1p in every non-base recipe. Thus “SVM-YJ” is not a claim that every numeric column is power-transformed.

### 4.3 Yeo–Johnson: a learned power-transform hypothesis

Yeo–Johnson is a **parametric power transformation**. It learns a parameter lambda for each continuous training feature rather than applying one fixed compression strength to all features.

It can handle positive, zero, and negative values. This is relevant to jerk and signed angular statistics, and is why it is more directly applicable than Box–Cox, which requires strictly positive inputs. Scikit-learn estimates its parameter from each feature through maximum likelihood. [PowerTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

A compact view of its branches:

- Positive x: a power of (x+1), with a logarithmic limit when lambda=0.
- Negative x: a power of (1-x), governed by (2-lambda), with a logarithmic limit when lambda=2.
- Lambda=1 gives the identity transformation.

The fitted lambda is **not selected using anomaly labels or a separate validation grid**. It is learned from original normal training, and the same fitted function is applied to all subsequent inputs.

**Why try YJ in addition to log?**

1. The features do not all have the same tail strength or symmetry.
2. A single signed-log rule treats both tails symmetrically; YJ can treat an asymmetric signed distribution differently.
3. A learned per-feature power offers an alternative to the fixed log compression.
4. For SVM/AE, changing marginal geometry may improve which relative deviations the model can use.

**What it does not imply:** neither the RBF OCSVM nor the reconstruction AE requires all input features to be Gaussian. A Gaussian RBF kernel is a similarity function, not an assumption that the data follows a Gaussian distribution. AE trains on reconstruction loss, not a feature-normality objective. YJ is therefore a reasonable **trial**, not a prerequisite for either model.

**Potential costs:** small, dependent training data can produce unstable or strong fitted powers; a large zero mass remains a zero mass; held-out extremes can be represented differently from what the model saw; making training marginals more symmetric may suppress useful anomaly magnitude. YJ is not uniformly tail compression.

Implementation is `PowerTransformer(method="yeo-johnson", standardize=False)`. The experiment then applies its **separate block scaler**, preserving the same count/category exceptions as the other recipes. The reasoning is to compare nonlinear choices without allowing the transformer's default automatic scaling to bypass those exceptions.

### 4.4 Quantile-to-normal: an empirical rank/distribution hypothesis

The quantile trial approximately maps:

`x -> fitted_training_CDF(x) -> inverse_normal_CDF(probability)`.

Unlike log, it is fitted; unlike YJ, it uses empirical distribution landmarks rather than one power parameter. It offers a more flexible way to represent strongly skewed marginals.

The mapping is nonlinear, can alter relationships/distances among features, and maps unseen values outside the fitted range to output boundaries. Therefore, very different beyond-range anomaly magnitudes can become similar after transformation. [QuantileTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)

**Why try it for OCSVM?** The RBF boundary depends on transformed distances. Quantile-to-normal tests whether a training-relative position representation is more useful than retaining tail magnitude or using a parametric power.

**Why be cautious?** With small-session, overlapping data, an empirical distribution can reflect a limited normal sample. Ties, such as many zeros, are not turned into genuinely new information. Saturation can weaken the distinction between moderately and extremely unusual values.

Exact settings: `output_distribution="normal"`, `n_quantiles=min(100, original_fit_rows)`, `subsample=None`, seed 42. For this experiment's OCSVM parent rows, n_quantiles=100. The selected numerical block is still standardized afterward.

Quantile was not tested for IF or AE. That is a **bounded recipe catalog**, not a result demonstrating that quantile cannot help those models. An AE quantile trial would also change what a large reconstruction error means by reducing original-magnitude distinctions.

## 5. Why the scaling differs by model

### 5.1 Isolation Forest: unscaled input is a defensible baseline

IF does not compute one Euclidean distance across all input coordinates. It selects features and random split values and uses isolation structure to score observations.

For a single feature, an ideal positive affine rescaling changes its numeric interval but not its normalized split positions. The forest consequently does not have the same “large unit dominates distance” problem as an RBF SVM or raw-MSE AE. Numerical implementation details aside, variance normalization is not the main reason to preprocess IF.

This is why both IF recipes are **unscaled**, and why the bounded trial focuses on **raw versus nonlinear log**.

Log is different from simple affine scaling: it redistributes intervals and the resulting random cuts. Extreme tails that occupy much of the raw range can leave ordinary values concentrated in a narrow interval; log changes that arrangement. It can help or hurt.

No Gaussian marginal assumption or StandardScaler requirement is imposed on IF. The fact that only raw/log were compared does not establish that YJ/quantile are inappropriate for trees.

### 5.2 RBF One-Class SVM: feature scale is part of the boundary

An RBF similarity is:

`K(x, y) = exp(-gamma * sum_j((x_j - y_j)^2))`.

If one unscaled derivative differs by millions and a ratio differs by 0.1, the derivative can dominate the squared distance. A global gamma cannot independently rebalance all features.

Hence the SVM applies common block StandardScaler after each nonlinear recipe. This makes the studied continuous/scaled-count/ratio coordinates training-relative before choosing kernel width. It does not make every source equally informative.

The experiment then asks: **with scaling controlled, do log, YJ, or quantile improve the normal-boundary geometry relative to base?**

Gamma remains coupled to the resulting representation. The study uses an original-training variance-based center, then tests multipliers around it, rather than letting augmented training recompute gamma automatically. The standard `gamma="scale"` convention is 1/(n_features*X.var()); this experiment supplies its corresponding frozen numeric reference explicitly. [OneClassSVM documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

### 5.3 LSTM AE: scaling changes the reconstruction objective

AE reconstructs the 33 encoded features over nine timesteps and scores mean squared error:

`score = sum((input - reconstruction)^2) / (9*33)`.

Without feature scaling, large-unit errors can dominate this loss. The network could spend most of its capacity reducing jerk/acceleration errors and gain little from accurately reconstructing smaller-unit behavior.

Block StandardScaler is therefore part of every AE trial. The optional nonlinear step then asks whether **ordinary versus extreme deviations** should be weighted differently before standardization.

This has two consequences:

- Log/YJ can improve optimization or make ordinary variations more visible.
- They can also weaken anomaly magnitude or emphasize a different collection of reconstruction errors.

A lower training/normal-validation MSE after one transformation is not automatically better anomaly discrimination. MSE numbers across recipes are in different coordinate systems and should not be compared as a universal common-unit quality measure.

AE fits on **micro features**, not parent aggregates: a 10-event velocity/rate/curvature statistic does not have the same distribution as its 50-event counterpart. Flattening original `(662,9,25)` inputs supplies 5,958 micro rows to the feature-wise transform/scaler; the tensor shape is then restored. Scaling is shared across timesteps, not fitted separately per sequence or timestep.

## 6. Why StandardScaler, why exclusions, and why not every scaler?

### 6.1 What StandardScaler contributes

For each selected numerical feature after its nonlinear step:

`z = (transformed_value - original_training_mean) / original_training_std`.

The mean and population standard deviation are learned once. A standardized unit is a training-relative deviation, not a physical pixel/second quantity. StandardScaler is sensitive to outliers and does not, by itself, remove distribution skew. [StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

That limitation is intentional in the comparison: **base + StandardScaler** asks whether common scale handling is sufficient, while **log/YJ/quantile + the same scaler** test changing shape first.

The study standardizes ratios and several ordinary counts too. Their physical source ranges may already be small, but their relative variation still determines distance/loss weighting. Choosing to scale them is a weighting decision, not proof that every bounded feature needs scaling.

### 6.2 Rare counts: low variance can exaggerate an event

Consider an illustrative binary count present in only 1% of training observations. Its standard deviation is about 0.0995, so a value of one becomes roughly 9.95 standardized units. Its contribution to a squared distance or squared reconstruction error can then be very large.

This arithmetic explains the concern with the mostly-zero copy/paste/window-switch block in this dataset. Leaving these features unscaled avoids automatically inflating the magnitude of a single count solely because it was rare in normal training.

- Copy/paste: retain the count, without nonlinear transform or StandardScaler.
- Window switching: use log1p in non-base recipes to compress larger totals, but do not StandardScale.
- Keyboard bursts/pauses: log1p in non-base recipes **and** StandardScale.

This is not a claim that rare events are unimportant. It chooses their numerical contribution explicitly rather than giving them a large inverse-variance weight by default. The experiment does not include a separate learned weighting/ablation study to prove that this is optimal.

### 6.3 Why not RobustScaler or MinMaxScaler throughout?

Experiment 1 does **not** compare feature-level StandardScaler, RobustScaler, MinMaxScaler, and no scaling in a full factorial grid.

The practical rationale is to keep one common, interpretable scale policy for SVM/AE while testing nonlinear shape changes within a bounded budget. Otherwise, a recipe change could alter nonlinear mapping, scaler type, feature weighting, and model parameters together, substantially increasing search opportunity on a small validation set.

Earlier ad-hoc development used log plus RobustScaler in parts of the workflow. That history is not a RobustScaler trial row in the current nine-recipe experiment.

RobustScaler uses median/IQR, so it answers a different scale question; MinMaxScaler ties the represented range to fitted extrema. Neither is proven inferior by this study. They are **not tested feature-scaler alternatives**, rather than rejected winners.

RobustScaler **does** appear in the ensemble, but on two model-score columns—not on the 25/33 input features. Those roles must not be confused.

### 6.4 Direction encoding and its weight

Direction class 0–8 is categorical. Integer encoding would assign arbitrary distances between sector numbers, including a large numerical distance between nearby wraparound sectors.

Nine indicators avoid that ordinal interpretation. The output has 24 numerical columns and nine indicators: 33 columns total. Indicators are not standardized, avoiding inverse-variance amplification of rare categories.

One-hot is not a complete circular-distance model: adjacent and opposite categories remain distinct indicators. It also changes the number and relative influence of direction-related coordinates. AE uses ordinary MSE over these indicator targets with an unconstrained linear reconstruction head; it does not use a separate categorical cross-entropy objective.

Thus “33 features” is a representation choice, not eight newly measured behaviors, and neither automatically beneficial nor automatically diluted signal. It is held fixed across all trials.

## 7. Fit scope, augmentation, and ensemble score scaling

### 7.1 Original-only feature-transform fitting

- IF/OCSVM fit recipe parameters on 662 original normal parent rows.
- AE fits on 5,958 original normal micro rows from the same 662 parents.
- Fitted transform/scaling parameters are reused for synthetic, validation, and test inputs.
- Each model candidate receives the frozen processor for its recipe; candidate fitting does not re-estimate it on augmentation-on training.

This keeps the feature coordinate system identical between augmentation off/on. The many micro rows and overlapping parents are not independent sessions; there are still 15 original training sessions.

### 7.2 Augmentation is a separate trial axis

Both conditions use the same recipe catalog:

- Off: 662 original normal parents/sequences.
- On: 662 originals + 1,324 synthetic copies = 1,986.

Each synthetic raw parent receives Gaussian mouse-coordinate noise with sigma drawn from [2,5] pixels, seed 42, and clipping to 1920×1080. Parent and micro representations share the same noisy realization; timestamps/actions/labels do not change.

The intention is local perturbation, but derivative features can react strongly when time gaps are tiny. The study does not assume that small coordinate noise guarantees identical engineered-feature distributions.

For AE, on also means roughly three times as many updates per epoch. Matched recipe/configuration comparisons therefore include both changed training inputs and extra optimizer work.

### 7.3 Ensemble normalization is a different preprocessing layer

The selected IF and OCSVM may have different **feature** recipes and unrelated **score** scales. Each retains its own frozen feature processor.

The ensemble separately applies:

`normalized_score_j = (score_j - median_normal_val_j) / IQR_normal_val_j`.

Median/IQR are fitted on 197 unaugmented pure-normal validation windows. The two normalized scores are combined using weights 0, 0.3, 0.5, 0.7, or 1.0 for IF, selecting by combined-validation ROC-AUC.

This is score-level RobustScaler calibration, not a feature-scaler trial or anomaly-probability conversion. It is intentionally validation-dependent calibration; feature preprocessing remains original-training-only. The 197 windows come from only four normal-validation sessions, which limits calibration robustness.

## 8. Exact model trials/configurations: persistent lookup

All main candidates use model seed 42 and both augmentation conditions. Recipe/configuration selection uses combined-validation ROC-AUC, with the first candidate winning ties. Average precision (AP, stored as pr_auc) is reported but not used to choose a configuration.

### 8.1 Model 1 — Isolation Forest

| Axis/control | Values tested/fixed |
| --- | --- |
| Preprocessing | IF-raw, IF-log |
| n_estimators | 100, 300, 500 |
| max_samples | 128, 256 |
| max_features | 0.75, 1.0 |
| contamination | Fixed "auto" |
| Feature scaler | None for either recipe |
| Fit count | 12 per recipe/condition; 24 per condition; **48 main fits** |

This compares whether tree count, sampled rows, feature subsampling, and selective log geometry help. “Auto” contamination does not supply the final binary threshold; the operating threshold is selected separately on validation.

YJ, quantile, and a feature-scaler grid were **not** tested for IF.

### 8.2 Model 2 — RBF One-Class SVM

| Axis/control | Values tested/fixed |
| --- | --- |
| Preprocessing | SVM-base, SVM-log, SVM-YJ, SVM-quantile |
| nu | 0.01, 0.05, 0.10 |
| Gamma multiplier | 0.25, 1.0, 4.0 |
| Kernel | Fixed RBF |
| tol | Fixed 0.001 |
| max_iter | Fixed -1 |
| Feature scaler | Fixed block StandardScaler |
| Fit count | Nine per recipe/condition; 36 per condition; **72 main fits** |

Numeric gamma is multiplier × the recipe's original-training center:

| Recipe | Gamma center | Gamma grid |
| --- | --- | --- |
| SVM-base | 0.04359031420070035 | 0.010897578550175088; 0.04359031420070035; 0.1743612568028014 |
| SVM-log | 0.0452880618000325 | 0.011322015450008126; 0.0452880618000325; 0.18115224720013 |
| SVM-YJ | 0.0452880618000325 | 0.011322015450008126; 0.0452880618000325; 0.18115224720013 |
| SVM-quantile | 0.0452880618000325 | 0.011322015450008126; 0.0452880618000325; 0.18115224720013 |

`center = 1 / (33 * variance(all entries of original transformed matrix, ddof=0))`.

Equal centers for three recipes do not imply equal distances/kernels. Scaling can yield equal overall variance while nonlinear mappings change the geometry.

nu controls the fitted boundary, training-error bound, and support-vector fraction behavior; it is not an assumed cheating prevalence. Other kernels or feature scalers were not tested.

### 8.3 Model 3 — LSTM autoencoder

| Block | Preprocessing | Hidden dimension | Batch size | Learning rate | Layers | Fits per recipe/condition |
| --- | --- | --- | --- | --- | --- | --- |
| Main | AE-base, AE-log, AE-YJ | 16, 32 | 16, 32, 64 | 0.0003, 0.001 | 1 encoder / 1 decoder | 12 |
| Matched depth | Same three | 16, 32 | 32 | 0.001 | 2 encoder / 2 decoder | Two |

Total: 14 candidates per recipe/condition, 42 per condition, **84 main AE fits**. The depth block is not a full re-tuning of every two-layer parameter.

| Fixed control | Value |
| --- | --- |
| Feature scaler | Block StandardScaler |
| Sequence | Nine timesteps × 33 encoded features |
| Loss/anomaly score | Mean squared reconstruction error over all encoded features/timesteps |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Latent dropout | 0.1 |
| Built-in LSTM dropout | 0.0 for either depth |
| Epoch cap | 100 |
| Early-stopping patience | 10 |
| Significant-improvement tolerance | Relative normal-validation loss reduction of at least 0.001 under the implemented comparison |
| Gradient clipping | Global norm 1.0 |
| Early-stopping input | Unaugmented pure-normal validation sequences |
| Batch policy | Shuffle; retain final incomplete batch |

The built-in/latent dropout separation ensures the two-layer trial does not also silently introduce inter-layer dropout. Quantile preprocessing, RobustScaler/MinMaxScaler variants, dropout grids, and weight-decay grids were **not** tested for AE.

### 8.4 Additional trials and totals

- Main: 48 IF + 72 OCSVM + 84 AE = **204 model fits**.
- Finalist repeats: IF-log/off, IF-raw/on, AE-base/off, AE-log/on each repeated at seeds 123 and 2026 = **eight extra fits**. Recipe, feature preprocessing, model hyperparameters, and augmentation realization stay fixed; these are not new preprocessing trials.
- Total base-model fits: **212**.
- Ensemble: five score weights in each augmentation condition, using selected base models without retraining.
- Primary reporting: 18 recipe/condition winners + two ensembles = 20.
- Sensitivity reporting: eight repeat rows, for 28 total approaches.

The experiment does not select the best repeat seed. No preprocessing/model/threshold choice is reselected from test results.

## 9. What the trials showed

### 9.1 Recipe-level lookup

These are the best validation ROC-AUC values within each recipe/augmentation condition, after that recipe's declared model grid.

| Recipe | Validation ROC, augmentation off | Validation ROC, augmentation on |
| --- | --- | --- |
| IF-raw | 0.7519 | 0.7847 |
| IF-log | 0.7618 | 0.7818 |
| SVM-base | 0.7361 | 0.7664 |
| SVM-log | 0.6549 | 0.6919 |
| SVM-YJ | 0.6379 | 0.6870 |
| SVM-quantile | 0.6433 | 0.6822 |
| AE-base | 0.7420 | 0.7419 |
| AE-log | 0.7317 | 0.7445 |
| AE-YJ | 0.6554 | 0.6546 |

Interpretation by model:

- **IF:** IF-log wins off, but IF-raw wins on. Log was a reasonable trial, not a universal requirement.
- **OCSVM:** SVM-base wins both conditions. In this split, common block scaling without continuous nonlinear transformation beats log/YJ/quantile on validation.
- **AE:** AE-base wins off and AE-log wins on. YJ is weaker in both. Model-specific loss/geometry matters more than choosing a transform solely because it makes marginals look more normal.

These are outcomes of the **whole fitted recipes plus bounded tuning**, not isolated measurements of one feature's benefit.

### 9.2 Every selected model configuration

Gamma is rounded to eight decimals here; the exact candidates are in Section 8.2. IF contamination, SVM solver/kernel, and AE fixed controls remain as listed above.

| Recipe | Augmentation | Selected model configuration | Val ROC-AUC | Val AP | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| IF-raw | off | trees=100, samples=256, features=1.0 | 0.7519 | 0.5721 | 0.7051 |
| IF-log | off | trees=100, samples=256, features=1.0 | 0.7618 | 0.5595 | 0.6881 |
| SVM-base | off | nu=0.01, gamma=0.04359031 | 0.7361 | 0.4511 | 0.7205 |
| SVM-log | off | nu=0.05, gamma=0.18115225 | 0.6549 | 0.3358 | 0.6653 |
| SVM-YJ | off | nu=0.01, gamma=0.18115225 | 0.6379 | 0.2855 | 0.6811 |
| SVM-quantile | off | nu=0.01, gamma=0.18115225 | 0.6433 | 0.3452 | 0.6468 |
| AE-base | off | layers=2, hidden=32, batch=32, lr=0.001 | 0.7420 | 0.5408 | 0.6368 |
| AE-log | off | layers=2, hidden=32, batch=32, lr=0.001 | 0.7317 | 0.5472 | 0.6211 |
| AE-YJ | off | layers=1, hidden=32, batch=16, lr=0.001 | 0.6554 | 0.3025 | 0.5804 |
| IF-raw | on | trees=100, samples=256, features=1.0 | 0.7847 | 0.6273 | 0.7138 |
| IF-log | on | trees=100, samples=256, features=1.0 | 0.7818 | 0.5728 | 0.7004 |
| SVM-base | on | nu=0.01, gamma=0.04359031 | 0.7664 | 0.5134 | 0.7323 |
| SVM-log | on | nu=0.01, gamma=0.18115225 | 0.6919 | 0.4083 | 0.6873 |
| SVM-YJ | on | nu=0.1, gamma=0.18115225 | 0.6870 | 0.3365 | 0.7026 |
| SVM-quantile | on | nu=0.1, gamma=0.18115225 | 0.6822 | 0.3956 | 0.6781 |
| AE-base | on | layers=2, hidden=16, batch=32, lr=0.001 | 0.7419 | 0.5598 | 0.6414 |
| AE-log | on | layers=2, hidden=32, batch=32, lr=0.001 | 0.7445 | 0.5717 | 0.6314 |
| AE-YJ | on | layers=1, hidden=32, batch=32, lr=0.001 | 0.6546 | 0.3036 | 0.5919 |

AP is average precision, not a trapezoidal PR-curve integral. Validation prevalence is about 0.2087 and test prevalence about 0.5601, so their AP values should not be compared as though they had the same baseline.

All 204 main candidates succeeded; all 72 OCSVM candidates reported convergence. No missing recipe cells are recorded.

### 9.3 Observations specifically relevant to preprocessing

**Good rationale is not the same as a winning transformation.** Training skew and mixed signs justified log/YJ trials, but SVM-base and AE-base/log outperform YJ on validation. Distribution normalization is not the same objective as anomaly discrimination.

**AE-YJ illustrates score-transfer fragility.** Its exact thresholds are about 19.10/off and 17.17/on. Each attains validation precision 0.5 by predicting two windows, one anomalous and one normal, with recall only 1/82. Every test score is below the selected threshold, so test recall is zero. This does not prove that one particular YJ-transformed feature caused the failure, or that YJ is universally unsuitable.

**Augmentation generally helps ranking within a fixed recipe.** Validation ROC improves in 86 of 102 matched configurations. IF-raw improves in 12/12; AE-log in 14/14. The small winner-level AE-base/YJ changes show why matched and independently tuned winner comparisons should both be reported.

**Depth is not a substitute for a useful representation.** Two layers improve nine of 12 matched validation comparisons, mostly modestly. AE-YJ remains weak despite extra capacity, and 77/84 AE fits reach the epoch cap. Cap-hitting alone does not establish undertraining.

**The final ensemble keeps the base models' own recipes.** Off selects IF-log + SVM-base; on selects IF-raw + SVM-base. Both select IF weight 0.7. The on ensemble is the overall validation winner, ROC-AUC 0.7946, with test ROC-AUC 0.7191. The larger observed SVM-base/on test ROC-AUC, 0.7323, is reported without changing the winner after test.

## 10. What this experiment did not establish

- It did not compare every possible transformation for every model.
- It did not prove StandardScaler is superior to RobustScaler or MinMaxScaler.
- It did not separately tune per-feature weights or test every log/count/ratio exception.
- It did not establish that YJ's fitted lambda should be chosen from test or validation anomaly labels.
- It did not test IF-YJ, IF-quantile, or AE-quantile.
- It did not isolate augmentation diversity from AE's extra optimizer updates.
- It did not prove that original-training Gaussian-looking marginals guarantee useful held-out tails.
- It did not make overlapping windows independent or demonstrate unseen-student generalization.
- It did not promote a new model into the API automatically.

The reusable takeaway is a **model-sensitive hypothesis**: consider domain/sign/tail structure, decide how numerical influence should work for that model, and test a bounded set of whole recipes against a simple base. Keep actual winning evidence separate from the reasons that motivated the trial.

## 11. Sources and implementation lookup

The choices and trial counts are verified against this repository's implementation and saved artifacts. Official documentation supports the operator descriptions, not claims that an operator is optimal for this dataset.

### Repository evidence

- [Fixed feature/transform blocks](../src/cheatdetect/data/feature_schema.py).
- [Implemented nonlinear/scaling operations](../src/cheatdetect/data/preprocessing.py).
- [Frozen Experiment 1 recipe/model grids](../src/cheatdetect/experiments/experiment_1/config.py).
- [Study evaluator and processor reuse](../src/cheatdetect/experiments/experiment_1/runner.py).
- [Original parent-feature audit](../study/artifacts/phase_01_audit/parent_features.csv) and [micro-feature audit](../study/artifacts/phase_01_audit/micro_features.csv).
- [All main candidates](../study/artifacts/phase_04_study/candidates.csv).
- [Selected parameters and thresholds](../study/artifacts/phase_04_study/reporting_manifest.json).
- [Matched augmentation comparison](../study/artifacts/phase_04_study/augmentation_matched.csv).
- [Matched depth comparison](../study/artifacts/phase_04_study/depth_comparison.csv).
- [Saved test results](../study/artifacts/phase_04_study/phase_06_reporting/test_results.csv).
- [Full Experiment 1 report](experiment_1_summary.md) for the full methodological/results account and artifact hashes.

### Official operator references

- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): training-relative mean/std scaling and its outlier sensitivity.
- [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html): Yeo–Johnson, Box–Cox, and fitted power parameters.
- [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html): empirical distribution mapping and boundary behavior.
- [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html): gamma/nu and solver controls.

The saved prepared manifest records scikit-learn 1.9.0; the linked current documentation is explanatory, not a claim that the study ran with a different version. Generated study artifacts are ignored/local, so some evidence links require that local snapshot. No new fits, preprocessing trials, protocol changes, or Git commits were performed to write this reference.


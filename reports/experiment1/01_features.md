# Experiment 1 — Features and the feature pipeline

Feature source, derivations, redundancy handling, and the frozen 25-feature
schema used by every Experiment 1 model. Every formula below is the implemented
behavior in `src/cheatdetect/data/features/`; numbers come from the frozen
original-training audit (`study/artifacts/phase_01_audit/`) unless noted.

## 1. Pipeline overview

```
raw events ──cleaning──▶ cleaned events
   │
   ├─ window extraction (50 events / stride 25; micro 10 / stride 5)
   │
   ├─ per-window predictors: 34
   │    ├─ actions module (6): elapsed_time, copy, paste, blur, focus, tab_switch
   │    ├─ mouse module (25)
   │    └─ keyboard module (3)
   │
   ├─ merge stage: blur + focus + tab_switch → window_switch_events   (32 predictors)
   │
   ├─ selection stage: zero-variance removal + Tier-1 correlation pruning
   │    └─ excluded 7 redundant mouse aggregates                      (25 source features)
   │
   └─ fixed schema and model preprocessing
        └─ 24 numerical columns + 9 direction indicators = 33 encoded outputs
```

Counts are code-verified: 35 extractor keys including the `is_cheating` label →
34 predictors; 32 after the merge; 25 retained by the frozen schema; 33 model
inputs after direction encoding.

## 2. Shared mouse mathematics

Consecutive **time-sorted mouse events** with coordinates $(x_i, y_i)$ and
times $t_i$:

```text
dx_i   = x_i − x_{i−1}
dy_i   = y_i − y_{i−1}
dt_i   = t_i − t_{i−1}
vx_i   = dx_i / dt_i
vy_i   = dy_i / dt_i
speed_i     = sqrt(vx_i² + vy_i²)
ax_i   = (vx_i − vx_{i−1}) / dt_i
ay_i   = (vy_i − vy_{i−1}) / dt_i
acc_i       = sqrt(ax_i² + ay_i²)
jerk_i      = (acc_i − acc_{i−1}) / dt_i
θ_i         = arctan2(dy_i, dx_i)
turn_i      = ((θ_i − θ_{i−1} + π) mod 2π) − π        # wrapped into [−π, π)
ω_i         = turn_i / dt_i
κ_i         = |vx_i·ay_i − vy_i·ax_i| / speed_i³
```

Handling rules:

- Only mouse-category rows (`mousemove`, `click`) enter the mouse subset,
  sorted by time inside the window. Parent-window membership follows raw event
  order.
- ±∞ from `dt = 0` or zero speed is replaced by NaN and skipped by the
  aggregations; an empty valid aggregate returns `0.0`.
- Standard deviations use the population convention (`ddof=0`). Degenerate
  windows (all timestamps equal) give `elapsed_time = 0`.
- Fewer than two mouse events returns zeros for every mouse feature (a lone
  click is therefore not counted). A zero can mean "not enough data" or a
  genuine zero.
- Small positive `dt` is **not** clipped; large but finite derivative values
  remain possible.

## 3. All 34 extracted predictors

Block key (schema membership; "—" = excluded by selection; "merged" = folded
into `window_switch_events`):
**continuous** (13: elapsed_time, path_length, velocity_mean, acceleration_mean,
jerk_mean, angular_velocity_mean/std/min, curvature_mean/std/min,
largest_deviation, keyboard_typing_rate), **signed** (3 of the continuous:
jerk_mean, angular_velocity_mean, angular_velocity_min), **log-count** (3:
keyboard_burst_count, keyboard_pause_count, window_switch_events),
**unscaled rare** (3: copy_events, paste_events, window_switch_events),
**bounded ratio** (2: straightness, idle_time_ratio), **ordinary count** (3:
direction_changes, click_count, sharp_angles), **angle aggregate** (1:
sum_of_angles), **categorical** (1: direction_class).

| # | Feature | Derivation (window of cleaned events) | Units / domain | Block | Schema |
|---|---|---|---|---|---|
| 1 | `elapsed_time` | `t_last − t_first` over the window | seconds, ≥ 0 | continuous | kept |
| 2 | `copy_events` | `Σ is_copy` | count | unscaled rare | kept |
| 3 | `paste_events` | `Σ is_paste` | count | unscaled rare | kept |
| 4 | `blur_events` | `Σ is_blur` | count | merged into window_switch | merged |
| 5 | `focus_events` | `Σ is_focus` | count | merged into window_switch | merged |
| 6 | `tab_switch_events` | `Σ is_tab_switch` | count | merged into window_switch | merged |
| 7 | `mouse_path_length` | `Σ sqrt(dx² + dy²)` | pixels, ≥ 0 | continuous | kept |
| 8 | `mouse_straightness` | `end_to_end / path_length` (0 if path = 0) | [0, 1] | bounded ratio | kept |
| 9 | `mouse_velocity_mean` | `mean(speed)` | pixels/s, ≥ 0 | continuous | kept |
| 10 | `mouse_velocity_std` | `std(speed)` | pixels/s, ≥ 0 | — | **excluded** |
| 11 | `mouse_velocity_max` | `max(speed)` | pixels/s, ≥ 0 | — | **excluded** |
| 12 | `mouse_acceleration_mean` | `mean(acc)` | pixels/s², ≥ 0 | continuous | kept |
| 13 | `mouse_acceleration_std` | `std(acc)` | pixels/s², ≥ 0 | — | **excluded** |
| 14 | `mouse_acceleration_max` | `max(acc)` | pixels/s², ≥ 0 | — | **excluded** |
| 15 | `mouse_jerk_mean` | `mean(jerk)` | pixels/s³, signed | continuous, signed | kept |
| 16 | `mouse_jerk_std` | `std(jerk)` | pixels/s³, ≥ 0 | — | **excluded** |
| 17 | `mouse_direction_changes` | `Σ [abs(turn) > π/4]` | count | ordinary count | kept |
| 18 | `mouse_click_count` | `Σ [Event Type = click]` | count | ordinary count | kept |
| 19 | `mouse_idle_time_ratio` | `Σ dt[speed ≤ P10(speed)] / Σ dt` | [0, 1] | bounded ratio | kept |
| 20 | `mouse_angular_velocity_mean` | `mean(ω)` | rad/s, signed | continuous, signed | kept |
| 21 | `mouse_angular_velocity_std` | `std(ω)` | rad/s, ≥ 0 | continuous | kept |
| 22 | `mouse_angular_velocity_min` | `min(ω)` | rad/s, signed | continuous, signed | kept |
| 23 | `mouse_angular_velocity_max` | `max(ω)` | rad/s, signed | — | **excluded** |
| 24 | `mouse_curvature_mean` | `mean(κ)` | 1/pixels, ≥ 0 | continuous | kept |
| 25 | `mouse_curvature_std` | `std(κ)` | 1/pixels, ≥ 0 | continuous | kept |
| 26 | `mouse_curvature_min` | `min(κ)` | 1/pixels, ≥ 0 | continuous | kept |
| 27 | `mouse_curvature_max` | `max(κ)` | 1/pixels, ≥ 0 | — | **excluded** |
| 28 | `mouse_direction_class` | net-displacement sector (below) | {0,…,8} | categorical | kept |
| 29 | `mouse_sum_of_angles` | `Σ abs(turn)` | radians, ≥ 0 | angle aggregate | kept |
| 30 | `mouse_largest_deviation` | `max` perpendicular distance to the start→end line | pixels, ≥ 0 | continuous | kept |
| 31 | `mouse_sharp_angles` | `Σ [abs(turn) > 0.0005]` | count | ordinary count | kept |
| 32 | `keyboard_typing_rate` | `n_keyboard_events / span` (0 if n < 2 or span ≤ 0) | events/s, ≥ 0 | continuous | kept |
| 33 | `keyboard_burst_count` | `Σ [keyboard gaps < 0.2 s]` | count | log-count | kept |
| 34 | `keyboard_pause_count` | `Σ [keyboard gaps > 1.0 s]` | count | log-count | kept |

Direction class: with `end_to_end > 0`, `degrees = (degrees(atan2(Δy, Δx)) + 360) mod 360`
and `class = min(floor(degrees / 45) + 1, 8)`; sector 1 is 0–45° from +X, and
screen Y increases downward (positive rotation is clockwise). With zero net
displacement or fewer than two mouse events, the class is 0 — which also covers
a path returning to its starting point.

Merge stage: `window_switch_events = blur_events + focus_events + tab_switch_events`
(all three denote leaving the exam window). `copy_events` and `paste_events`
remain separate. After the merge the pipeline carries 32 predictors.

Keyboard events include `copy` and `paste`, so `keyboard_typing_rate` is an
event-rate proxy, not characters per second.

## 4. Selection stage

The frozen schema is the output of a two-stage selection applied to the
original normal-training feature matrix:

1. **Zero-variance removal** — constant columns carry no split information.
   In the frozen training snapshot no column is constant, so this stage removed
   nothing; constants remain representable (retained, not silently dropped).
2. **Tier-1 correlation pruning** — for every pair with
   Pearson $|r| > 0.85$ on original training rows, the pair is classified by
   base name after stripping the aggregation suffixes
   `_mean`, `_std`, `_max`, `_min`, `_sum`:
   - **Tier 1 (same base signal, different aggregation)**: keep `_mean`; drop
     `_max`/`_min` before `_std`.
   - **Tier 2 (different signals, e.g. velocity vs acceleration)**: kept —
     they capture distinct physical quantities even when correlated.
   - Sparse/rare columns are deliberately kept even when zero in most normal
     windows: in one-class detection they are discriminative rather than noise.

Excluded by the selection (7 mouse aggregates, each a same-base duplicate of
the retained mean/std statistic):

| Excluded feature | Same-base partner(s) | Reason |
|---|---|---|
| `mouse_velocity_std` | `mouse_velocity_mean` | same signal, alternative aggregation |
| `mouse_velocity_max` | `mouse_velocity_mean` | same signal, max aggregation |
| `mouse_acceleration_std` | `mouse_acceleration_mean` | same signal, alternative aggregation |
| `mouse_acceleration_max` | `mouse_acceleration_mean` | same signal, max aggregation |
| `mouse_jerk_std` | `mouse_jerk_mean` | same signal, alternative aggregation |
| `mouse_angular_velocity_max` | `mouse_angular_velocity_mean` | same signal, max aggregation |
| `mouse_curvature_max` | `mouse_curvature_mean` | same signal, max aggregation |

Retained Tier-1 survivors (`_mean`/`_std`/`_min` combinations such as
`mouse_angular_velocity_mean/std/min` and `mouse_curvature_mean/std/min`) are
kept because their same-base pairs either stayed below the correlation
threshold or their statistic was the designated keep. Selection reduced
redundancy; it does not claim the retained pairs are uncorrelated. The 25-feature
schema is then fixed for every recipe, augmentation state, and model — no
re-selection is performed per condition or on held-out data.

## 5. Frozen schema and preprocessing blocks

25 source features in fixed order (same as the table above, kept rows).
Preprocessing blocks (`src/cheatdetect/data/feature_schema.py`):

| Block | Members |
|---|---|
| Continuous (13) | elapsed_time, path_length, velocity_mean, acceleration_mean, jerk_mean, angular_velocity_mean/std/min, curvature_mean/std/min, largest_deviation, keyboard_typing_rate |
| Signed (3) | jerk_mean, angular_velocity_mean, angular_velocity_min |
| Log-count (3) | keyboard_burst_count, keyboard_pause_count, window_switch_events |
| Unscaled rare (3) | copy_events, paste_events, window_switch_events |
| Counts (8) | copy_events, paste_events, direction_changes, click_count, sharp_angles, burst, pause, window_switch |
| Direction (1) | mouse_direction_class |

Model-input encoding: the 24 numerical columns plus 9 direction indicators =
**33 outputs**, in the order 24 numerical → indicators 0–8. The indicator block
replaces the integer code so categories do not carry ordinal distances; it
does not encode circular adjacency between sectors. Downstream treatment
(imputation medians, nonlinear recipes, scaling exclusions, constant bypass) is
documented in `03_experiments_and_models.md` §4.

## 6. Extraction correctness notes

- Angle arithmetic is wrapped into $[-\pi, \pi)$ consistently for direction
  changes, sharp angles, the angle sum, and angular velocity, so crossing ±π is
  a small turn rather than an artificial ~2π jump.
- Training-file timing: 12 timestamp decreases across three files and two
  duplicate mouse timestamps across two files; the minimum positive mouse gap
  is ≈ 0.001 s. Derivatives amplify tiny gaps — original-training parent
  acceleration reaches ≈ 8.27e6 px/s², parent jerk ≈ 8.42e9 px/s³, and micro
  jerk ≈ 5.41e10 px/s³. Keyboard rate reaches ≈ 2,000 events/s via the
  count/span definition. Finite does not imply physically plausible; no dt
  floor or epsilon is applied.
- Aggregation caveats that affect interpretation: jerk is the derivative of
  scalar acceleration magnitude (signed, not vector-magnitude jerk); the
  "sharp angle" threshold is 0.0005 rad (~0.029°); idle ratio is relative to
  each window's own speed distribution; largest deviation is measured from the
  infinite start→end line; direction class 0 covers both no movement and
  closed-loop displacement.
- Distributions (`figures/feature_distributions.png`) and the correlation
  heatmap (`figures/correlation_heatmap.png`) are the original-training EDA
  views for the frozen schema.

## 7. Provenance

- Implementation: `src/cheatdetect/data/features/{mouse,keyboard,actions,extract}.py`,
  `src/cheatdetect/data/build.py` (merge), `src/cheatdetect/data/feature_schema.py`
  (order and blocks), `src/cheatdetect/data/cleaning.py` (flags and coercion).
- Audit evidence: `study/artifacts/phase_01_audit/{parent_features.csv,micro_features.csv,sessions.csv,kernel_geometry.csv,transformed_features.csv}`.
- Selection rules: zero-variance + Tier-1 correlation at $|r| > 0.85$, applied
  to original normal-training rows; the 7 excluded aggregates and the 25-feature
  output are the frozen schema. The selection implementation and its
  description are in the repository's `data/selection.py` lineage and
  `reports/report.md` §3.4.

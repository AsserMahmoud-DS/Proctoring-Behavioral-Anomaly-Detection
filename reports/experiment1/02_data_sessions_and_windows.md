# Experiment 1 — Sessions, splits, windows, and augmentation

Dataset composition, session-file splitting, window geometry, labels, and the
paired training-only augmentation. Counts are verified from the frozen audit
(`study/artifacts/phase_01_audit/`), the prepared bundle
(`study/artifacts/phase_03_study/`), and the reporting dataset summary
(`tables/dataset_summary.csv`).

## 1. Corpus

| Item | Value |
|---|---|
| Pure-normal session files | 19 |
| Mixed (annotated anomalous segments) session files | 17 |
| Assigned files (frozen split) | 36 |
| Files contributing windows | 34 |
| Raw CSV rows | 38,039 |
| Cleaned events | 36,784 |

File counts are not unique-student counts: the same participant may have normal
and mixed recordings, or several recordings. The two zero-event mixed-test files
(`mixed - El Farouk Omar.csv`, `youssef ahmed- mixed.csv`) stay in the frozen
assignment and are explicitly excluded from window evaluation; no replacement
files are drawn.

## 2. Raw-event cleaning

`clean_session_data` performs, in order:

1. Keep `Time (seconds)`, `Event Type`, `X/Y Coordinate`, `Action`, `Is Cheating`.
2. Strip the footer from the `SESSION SUMMARY` marker onward.
3. Coerce time to numeric; drop rows with invalid time.
4. Tag mouse events (`mousemove`, `click`), keyboard events
   (`keydown`, `copy`, `paste`), and action events (`tab-switch`).
5. Coerce coordinates, forward-fill X/Y within the session, drop remaining missing.
6. Create copy/paste/blur/focus/tab-switch flags (from `Action` and/or `Event Type`).
7. Parse the annotation: stripped uppercase comparison to `TRUE`; absent
   annotation defaults to normal.

Cleaning preserves raw event order; mouse and keyboard feature modules sort
their own subsets by time (see `01_features.md`).

## 3. Split design

Splitting happens at the **session-file level before any windowing**:

- Pure-normal files: 80% train / 20% normal validation via
  `train_test_split(..., random_state=42)` on sorted file lists → 15 and 4 files.
- Mixed files: 30% validation / 70% test → 5 and 12 assigned files.
- No training-session window randomly appears in validation or test; windows
  from one session never straddle partitions.
- Ordered file assignments (not just the seed) are persisted in
  `dataset/processed/split_info.json`, re-verified by the Phase 1 audit, and
  frozen in the audit manifest consumed by every study phase.

This is stronger than random overlapping-window splitting but is **not an
unseen-student split**. Validation combines pure-normal sessions with mixed
sessions; test contains mixed sessions only, so their normal-window populations
may differ.

| Partition | Assigned files | Contributing files | Parent windows | Anomalous windows | Prevalence |
|---|---|---|---|---|---|
| Original normal training | 15 | 15 | 662 | 0 | 0% |
| Synthetic normal training | 15 originals | 15 originals | 1,324 | 0 | 0% |
| Pure-normal validation | 4 | 4 | 197 | 0 | 0% |
| Mixed validation | 5 | 5 | 196 | 82 | 41.84% |
| Combined validation | 9 | 9 | 393 | 82 | 20.87% |
| Mixed test | 12 | 10 | 366 | 205 | 56.01% |

Training uses 662 parent windows with augmentation off and
**1,986 = 662 + 1,324** with augmentation on.

## 4. Window geometry

- **Parent window:** 50 consecutive cleaned events, stride 25. For a session
  with $E$ cleaned events, usable parents =
  `max(0, floor((E − 50) / 25) + 1)`.
- **AE micro-window:** inside each parent, 10 events with stride 5 →
  `floor((50 − 10) / 5) + 1 = 9` timesteps.
- Flat models receive one 25-source aggregate per parent (33 columns after
  encoding); the AE receives the aligned `(N, 9, 25)` source tensor (encoded to
  `(N, 9, 33)`). Both views are extracted from the **same realized 50-event
  window**, in the same order, with the same parent label.
- Parent windows overlap (stride 25 < 50) and adjacent micro-windows overlap
  (stride 5 < 10); 5,958 micro rows are expanded representations, not
  independent samples.

## 5. Labels

A parent is anomalous when the fraction of annotated cheating events inside it
meets the frozen threshold:

`mean(event_is_cheating within the 50-event parent) >= 0.5`.

The label is attached to both the flat row and the AE sequence; micro-window
labels are never targets. Normal training and its synthetic copies are
normal-only — construction rejects any anomalous event inside a normal
training window. Annotations are supplied session labels, not independently
verified misconduct findings.

## 6. Augmentation

Applied to original normal-training parents only:

1. Draw one $\sigma \sim U[2, 5]$ pixels per copy.
2. Add independent zero-mean Gaussian noise of that σ to the X/Y coordinates of
   mouse/click events; timestamps, event types, action flags, event order, and
   annotations stay unchanged.
3. Clip synthetic coordinates to the screen bounds `[0, 1920] × [0, 1080]`.
4. Re-extract the parent aggregate **and** its nine micro-feature vectors from
   that same noisy realization.
5. Store identity `(session_file, parent_start_event, copy_index)` with copy 0
   original and copies 1–2 synthetic.

Two copies per original parent; generation seed 42; the realized noisy windows
are shared across the nine recipes and all model seeds. No normal validation,
mixed validation, or test row is augmented. Consequences to keep in view:

- Local coordinate perturbation does not guarantee unchanged engineered-feature
  distributions; derivative/curvature features are nonlinear and dt-sensitive.
- For the AE, augmentation also increases the number of optimizer updates under
  the fixed epoch budget (at batch 32: 21 updates/epoch on 662 rows vs 63 on
  1,986; 2,100 vs 6,300 updates over 100 epochs). Data diversity and extra
  optimization are not separated.

## 7. Excluded/zero-window sessions

| File | Partition | Raw rows | Cleaned events |
|---|---|---|---|
| `mixed - El Farouk Omar.csv` | mixed_test | 32 | 0 |
| `youssef ahmed- mixed.csv` | mixed_test | 32 | 0 |

They remain part of the frozen 12-file mixed-test assignment; the audit and
readiness reports keep them with explicit exclusion reasons, and no metric
counts them as windows.

## 8. Provenance

- Implementation: `src/cheatdetect/data/{loader,cleaning,paired,augment,dataset}.py`,
  `src/cheatdetect/data/build.py`.
- Evidence: `study/artifacts/phase_01_audit/{manifest.json,sessions.csv}`,
  `study/artifacts/phase_03_study/{manifest.json,session_readiness.csv}`,
  `tables/dataset_summary.csv`.

# Experiment 1 — Integration & Execution Plan

> Historical document: Engineering phases A–E record the integration workflow and its earlier status. Study phases 0–6 are defined in [the scientific protocol](../study/plans/experiment_plan.md). The [final report](../reports/experiment1/README.md) records the executed study; this checklist is retained as history, not a current completion dashboard.

## Context

Experiment 1 is the **preprocessing & model feasibility study**. Its research
protocol is `study/plans/experiment_plan.md`; its aim and file inventory are in
`src/cheatdetect/experiments/experiment_1/{brief.md,files.md}`. This document
coordinates the **engineering integration** that turns the study from parallel
files into one coherent, tracked workflow, and then executes the study protocol.

The problem being solved: the study previously lived entirely under Git-ignored
`study/` with a second, parallel data path (fixed schema + `FeaturePreprocessor`)
that never met the production pipeline (dynamic `select_features` +
detector-internal `Log1pSkewed`/`RobustScaler`). Integration unifies them.

## Branch workflow

- `experiment1` — published base (`origin/experiment1`); contains Phase A.
- `experiment1-phase-b` — Phase B working branch (off `experiment1`).
- Each phase is a branch off `experiment1`; fast-forward-merge a completed phase
  back into `experiment1` before starting the next.

## Locked decisions

1. Experiment framework code is tracked under `src/cheatdetect/experiments/`;
   per-experiment code lives in `experiments/experiment_1/`.
2. The dataset builder lives in `src/cheatdetect/data/dataset.py`.
3. Production **unifies** on the fixed 25→33 schema + `FeaturePreprocessor`,
   replacing dynamic feature selection and detector-internal transforms.
4. Selection is separated from reporting: shared `search` (validation-only) and
   `report` (test) entry points in `src/cheatdetect/pipeline/`.
5. Experiments consume the frozen audit split; seed-42 split *generation* is
   retained for future data but not re-derived for the current study.
6. Study findings stay untracked; they are summarized in the final
   system/experiments report.
7. Legacy modules are removed in Phase B; docs, cleanup, and the final report are
   deferred to Phase E.
8. Conventional commits; one phase per branch; never commit or merge without an
   explicit instruction.

## Module disposition

| Action | Modules |
|---|---|
| Keep | `data/cleaning.py`, `build.py`, `loader.py`, `augment.py`, `features/*` |
| Promoted (tracked) | `data/feature_schema.py`, `paired.py`, `preprocessing.py`, `dataset.py` |
| Refactor | `models/{isolation_forest,ocsvm,lstm_ae}.py`, `pipeline/train.py` |
| Add | `pipeline/{search,report}.py`, `experiments/experiment_1/{config,runner,reporting}.py`, `experiments/__main__.py` |
| Remove (Phase B) | `data/{selection,transform,sequences,pipeline}.py`, `tests/test_sequences.py`, `.npz`/`.pkl` caches |

---

## Phase A — Tracked framework  (DONE)

Reusable experiment code moved out of Git-ignored `study/` into tracked modules so
a clean clone can reproduce the study.

- [x] `src/cheatdetect/experiments/` package + `experiment_1/audit.py`.
- [x] `data/{feature_schema,paired,preprocessing,dataset}.py` promoted to tracked.
- [x] `config.py` package/source dir constants + study artifact paths.
- [x] `tests/{test_audit,test_dataset,test_recipe_specification,test_preprocessing,test_paired,test_mouse_features}.py`.
- [x] Removed `study/{data,audit,preprocessing,__init__}.py`, `study/tests/`, and
      the empty `src/cheatdetect/{study,research}/` packages.
- [x] App threshold equality (`score >= threshold`) + regression.
- [x] Committed on `experiment1` (5 commits); published to `origin/experiment1`.

**Done when:** `uv run pytest` green (156 passed) and no import resolves into
Git-ignored code.

---

## Phase B — Unify on the fixed schema + remove legacy  (IN PROGRESS)

**Purpose:** make the study's frozen data/preprocessing contract the one
production path.

- [x] B1. Detectors wrap `FeaturePreprocessor(recipe, scale)` instead of
      `Log1pSkewed`+`RobustScaler`:
      - IF: `recipe ∈ {base, log}`, `scale=False`, add `max_features`.
      - OCSVM: `recipe ∈ {base, log, yj, quantile}`, `scale=True`, numeric `gamma`.
      - LSTM: same recipe path.
      Done: `grid_search` dropped `skewed_cols`; LSTM encodes 25→33 inside the
      detector and builds the AE with the encoded dim; `config.lstm_recipe`
      replaces `lstm_preprocessing`; `_build_matrices` selects the frozen
      `SOURCE_FEATURES` (detectors own imputation). Tests use valid 25-feature
      synthetic data (`tests/synthetic.py`). `uv run pytest` -> 156 passed.
      Commit: `refactor(models): own fixed-schema preprocessing in detectors`.
      Files: models/{isolation_forest,ocsvm,lstm_ae}.py, pipeline/train.py,
      config.py, tests/{test_lstm,test_model_selection,synthetic}.py
- [x] B2. `data/dataset.py` becomes the single builder (frozen split, paired
      views, recipes fitted once); replace `prepare_data`'s dynamic
      `select_features` and cache-pickle path.
      Done: `RECIPES` catalog + raw-view helpers (`training_raw`,
      `training_sequences`, `training_labels`); geometry/label/seed/recipes
      params threaded through `prepare_study`/`input_manifest` with unchanged
      defaults; `prepare_data` = split → `prepare_study(recipes=())` and
      returns paired raw matrices/labels. Pickle caches removed.
      Commit: `refactor(data): make dataset builder the single production path`.
- [x] B3. Split `pipeline/train.py` into `pipeline/search.py` (validation-only, no
      test import) and `pipeline/report.py` (test); `train_pipeline(config)` =
      dataset → search → report → serialize. Keep seed-42 split generation for
      future data; the study reads the frozen audit split.
      Done: `search_validation`/`search_lstm`/`resolve_lstm_es` are
      validation-only (AST test forbids test names); `report.py` owns test
      scoring; `train.py` is orchestration only. Seed-42 split generation kept.
      Commit: `refactor(pipeline): separate validation search from test reporting`.
- [x] B4. LSTM consumes `PreparedData` sequences; retire the `.npz` caches.
      Done: `prepare_lstm_data(config, data)` reads aligned sequences/labels
      from the same `PreparedStudy`; `_run_lstm` derives `input_dim` from the
      sequence shape. `.npz` caches and `_load_or_build_sequences` removed.
      Commit: `refactor(data): make dataset builder the single production path`.
- [x] B5. Migrate `app/`: config drops dynamic `features_to_keep`; `service.py`
      slices `SOURCE_FEATURES`; preprocessing lives inside the detector artifact.
      Done: `InferenceConfig` drops the field (legacy JSON still loads via
      `extra="ignore"`); `_serialize_model` no longer persists it; added a
      service test with a real fitted detector. Commit:
      `refactor(app): infer on the fixed source schema`.
- [x] B6. Migrate `notebooks/train_evaluate.py` + `eda.*`; delete stale caches and
      the removed legacy modules/tests.
      Done: `audit.py` rebuilt on `build_paired_representation` (option a) with
      clarified docs/comments; removed `data/{pipeline,selection,sequences,transform}.py`,
      `tests/test_sequences.py`, dead config cache paths, and on-disk caches;
      `data/__init__.py` trimmed; `eda.py` rewritten for the fixed schema and
      jupytext-synced; `train_evaluate` needed no code change; README/files.md
      updated. Commits: `refactor(experiments)`, `chore(data)`, `docs(readme)`.
- [x] B7. Tests: update to the fixed schema; remove `test_sequences.py`; add a
      regression that a `search` winner reproduces through `train_pipeline`.
      Done: all tests on the fixed schema; `test_sequences.py` removed (B6);
      added `tests/test_search_reproduction.py` (winner/threshold/scores and
      serialized artifact parity, self-contained `FeaturePreprocessor`) and
      `tests/test_data_exports.py` (legacy modules/names guard). `uv run
      pytest` -> 157 passed.

**Done when:** full suite green; production trains end-to-end on the fixed
contract; app tests pass (outside sandbox).

---

## Phase C — Study runner  (Study protocol Phase 3)

**Purpose:** bounded grids without automatic test evaluation.

- [x] C1. `experiments/experiment_1/config.py`: freeze recipes/grids/geometry/
      seeds/budgets from `study/plans/experiment_plan.md`.
      Done: frozen `StudyConfig` + candidate builders (`if_candidates`,
      `ocsvm_candidates` with numeric gamma scales, `ae_candidates` with the
      depth block) + `validate_budgets` matching 24/36/42, 204/8/212.
- [x] C2. `experiments/experiment_1/runner.py`: frozen-grid search over
      `PreparedData` via `pipeline.search`; per-candidate artifacts + selection;
      never imports test.
      Done (core): `build_study`/`load_study` (manifest-guarded phase-3 bundle),
      paired-condition sweeps with deep-copied frozen preprocessors (Option A),
      per-cell winners + thresholds, per-condition ensembles, frozen reporting
      manifest, candidates table. Shared sweep extracted to
      `models/selection.py` (used by all grid searches and the runner).
      Phase-5 repeats and the reporting gate land with C5.
- [x] C3. AE controls: separate latent vs built-in dropout, weight decay,
      gradient clipping, relative-improvement tolerance, best-checkpoint restore,
      epoch/update diagnostics.
      Done: `LSTMAutoencoder` takes `latent_dropout=0.1` / `lstm_dropout=0.0`
      (built-in stays 0, also for 2-layer depth); `_train_lstm_ae` uses Adam
      weight decay 1e-4, global clip 1.0, relative-improvement early stop, and
      records epochs/best epoch/optimizer updates/best ES loss; detector and
      runner pass the frozen `StudyConfig` controls; production grid key is now
      `latent_dropout`.
- [x] C4. OCSVM numeric gamma candidates + convergence reporting.
      Done: numeric gamma candidates come from the frozen centres (C1);
      `convergence_diagnostics()` reports `n_iter`/`converged` (warns on
      non-convergence); runner and grid searches persist the columns.
- [x] C5. `pipeline.report` gated behind a frozen manifest; shared CLI
      `python -m cheatdetect.experiments --phase N`.
      Done: `reporting.py` scores the frozen primary + sensitivity artifacts
      once (fails closed without manifest/repeats or on bundle mismatch);
      `run_seed_repeats` repeats the per-condition IF/AE finalists at seeds
      123/2026 without overwriting seed-42 artifacts and appends a frozen
      sensitivity list; `experiments/__main__.py` dispatches phases 3-6 with
      lazy torch. Tests: reporting gate, repeats, CLI.

- [x] C6. Execution notebooks under `experiments/experiment_1/notebooks/`:
      `study_run` (phases 3-4), `study_sensitivity` (phase 5),
      `study_report` (phase 6 tables/figures). Tracked `.ipynb`, ignored `.py`
      twins; thin callers of `runner`/`reporting` with no selection logic.

**Done when:** a synthetic study selects and persists paired recipe winners
without running final test reporting or touching `best_models/`.

---

## Phase D — Execution  (Study protocol Phases 4–6)

### D0 — Persist comparisons and figures  (DONE)

- [x] `analysis.py`: matched augmentation, independently tuned winner, and AE
      depth comparison tables; `run_search` writes
      `augmentation_matched.csv`, `augmentation_winners.csv`,
      `depth_comparison.csv` next to `candidates.csv`.
- [x] `reporting.run_phase6` enriches the summary with locked params and test
      prevalence, writes `dataset_summary.csv`, and persists grouped figures
      (ROC/PR/scores/confusion per family x condition) via `write_figures`.
- [x] Notebooks are guarded viewers: `study_run`/`study_sensitivity` skip
      refitting when persisted artifacts exist; `study_report` displays the
      persisted tables/figures. Caveats belong to E4.
- [x] Tests: `test_analysis.py` joins, reporting persistence/figures, runner
      comparison outputs.
- [x] P1 progress logging: per-cell INFO (start/locked) and per-candidate
      DEBUG; P2 missing-cell reporting: `missing_cells` recorded in the frozen
      manifest, displayed in `study_run`, warned on in phases 5/6.

- [ ] D1. Phase 4: seed-42 full paired grids (204 main fits); matched
      augmentation/depth comparisons; lock 18 recipe winners + 2 ensembles;
      validation-only selection/thresholds; freeze the reporting manifest.
      Surfaces: `experiment_1/notebooks/study_run.ipynb` or
      `python -m cheatdetect.experiments --phase 4`.
- [ ] D2. Phase 5: repeat locked IF/AE finalists at seeds 123/2026 (8 fits; 212
      total); no best-seed selection.
      Surfaces: `study_sensitivity.ipynb` or CLI `--phase 5`.
- [ ] D3. Phase 6: final test tables, prevalence baselines, ROC/PR and score
      plots, confusion matrices, seed sensitivity, caveats. The user runs the
      thin Jupytext notebook.
      Surfaces: `study_report.ipynb` or CLI `--phase 6` (figures only in the
      notebook).

**Done when:** locked-model tables, figures, artifacts, and conclusions agree; no
deployment promotion.

---

## Phase E — Cleanup, docs & final report

- [ ] E1. Update `plans/architecture.md` (LSTM/ES/selector + integration paths).
- [ ] E2. Residual cleanup (dead exports, docstrings).
- [ ] E3. Refresh and run the final notebooks (user executes).
- [ ] E4. Write the system/experiments report summarizing the untracked study
      findings (methodology, winners, augmentation/depth comparisons, seed
      sensitivity, limitations).

---

## Risks

- Unifying changes the deployed model; `best_models/` is stale until retrained.
- `experiments/` must never be imported by `app/` (`--no-dev` has no torch).
- `eda` notebook imports from `cheatdetect.data` must be updated once legacy
  exports are removed.
- The saved Phase 2 readiness manifest hashes the old `study/` sources; rebuild
  it when the study next runs.

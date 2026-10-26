# Experiment 1 — File inventory

This inventory describes the files selected for the Experiment 1 snapshot.
See the [final report](../../../../reports/experiment1/README.md) and
[reproduction guide](../../../../reports/experiment1/REPRODUCING.md).
Historical development labels have been removed; Git records commit status.

## Shared experiment code

| File | Role |
|---|---|
| `src/cheatdetect/data/feature_schema.py` | Fixed 25-source / 33-output schema and preprocessing blocks |
| `src/cheatdetect/data/paired.py` | Aligned flat + sequence extraction from shared realized parent windows |
| `src/cheatdetect/data/preprocessing.py` | `FeaturePreprocessor` recipes, `gamma_center`, `validate_features` |
| `src/cheatdetect/data/dataset.py` | Manifest-guarded `PreparedStudy` builder, save/load, `input_manifest` |
| `src/cheatdetect/data/__init__.py` | Exports the above public API |
| `src/cheatdetect/experiments/__init__.py` | Experiment framework package marker (generic) |
| `src/cheatdetect/experiments/experiment_1/__init__.py` | Experiment 1 subpackage marker |
| `src/cheatdetect/experiments/experiment_1/audit.py` | Original-training audit (Phase 1) entry point |
| `src/cheatdetect/config.py` | Study/experiment path constants |

## Tests

| File | Role |
|---|---|
| `tests/test_recipe_specification.py` | `FeaturePreprocessor` matches the frozen Phase 1 spec, all recipes × scale |
| `tests/test_dataset.py` | Paired extraction, fit-once, manifest invalidation, isolation, exclusions |
| `tests/test_audit.py` | Audit statistics, signed log, gamma geometry, train-only reads |
| `tests/test_preprocessing.py` | Imputation/transform/scaling/encoding unit coverage |
| `tests/test_paired.py` | Parent/micro alignment and identity coverage |
| `tests/test_mouse_features.py` | Feature-extraction regressions (angle wrapping) |

## Documentation

| File | Role |
|---|---|
| `src/cheatdetect/experiments/experiment_1/brief.md` | This experiment's aim and protocol summary |
| `src/cheatdetect/experiments/experiment_1/files.md` | This inventory |

## Shared pipeline entry points

| File | Role |
|---|---|
| `src/cheatdetect/pipeline/search.py` | Validation-only search used by the runner and production |
| `src/cheatdetect/pipeline/report.py` | Test reporting entry point (called after selection freezes) |
| `src/cheatdetect/models/selection.py` | Ordered candidate sweep: ROC-AUC selection, first-candidate tie |

## Experiment 1 runner (Study Phase 3)

| File | Role |
|---|---|
| `src/cheatdetect/experiments/experiment_1/config.py` | Frozen `StudyConfig`: recipes, grids, seeds, budgets |
| `src/cheatdetect/experiments/experiment_1/runner.py` | Validation-only grid runner; winners, ensembles, seed repeats, frozen manifest |
| `src/cheatdetect/experiments/experiment_1/analysis.py` | Matched augmentation, tuned-winner, and AE depth comparisons (no refits) |
| `src/cheatdetect/experiments/experiment_1/reporting.py` | Final test reporting behind the frozen manifest (fails closed); persists tables and figures |
| `src/cheatdetect/experiments/__main__.py` | Shared CLI (`python -m cheatdetect.experiments --phase N`) |

## Notebooks (Study Phases 3-6)

Both `.py` and `.ipynb` forms are preserved. Edit the `.py` twin and
Jupytext-sync it. The run/sensitivity notebooks skip fitting when completion
markers exist; the reporting notebook scores saved models and writes outputs.

| File | Role |
|---|---|
| `.../experiment_1/notebooks/study_run.ipynb` | Phase 3-4: run or review the locked search + persisted comparisons |
| `.../experiment_1/notebooks/study_sensitivity.ipynb` | Phase 5: per-seed repeat table (no best-seed selection) |
| `.../experiment_1/notebooks/study_report.ipynb` | Phase 6: frozen test tables, grouped figures, sensitivity summary |

Persisted Phase 4 outputs include `candidates.csv`,
`augmentation_matched.csv`, `augmentation_winners.csv`, and
`depth_comparison.csv`; Phase 6 adds `test_results.csv`/`.json`,
`dataset_summary.csv`, and `figures/`.

## Preserved research materials and results

- [study/README.md](../../../../study/README.md) inventories the four historical
  notebook pairs, scientific protocol/findings, and all original artifact stages.
- [plans/experiment_1_phases.md](../../../../plans/experiment_1_phases.md) preserves
  engineering phases A–E; scientific phases 0–6 are defined in
  [study/plans/experiment_plan.md](../../../../study/plans/experiment_plan.md).
- `dataset/processed/split_info.json` preserves the audit prerequisite.
- Root `notebooks/eda` and `notebooks/train_evaluate` retain both paired formats.
- `reports/experiment1/` contains the final chapters, tables, figures, and earlier
  supporting references under `archive/`.
- `pyproject.toml`, `uv.lock`, shared model/evaluation/pipeline modules, and the
  full `tests/` suite remain part of the repository snapshot.
- `study/SHA256SUMS` records original artifact and split-metadata fingerprints.

Private raw sessions require an external backup. Logs, caches, unrelated plans,
duplicate report exports, and future run directories remain ignored.

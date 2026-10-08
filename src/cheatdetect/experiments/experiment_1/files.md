# Experiment 1 — File inventory

The tracked files that belong to Experiment 1. **Status** is one of: `tracked`
(committed), `new` (present, not yet committed), or `planned` (Phase 3+).

## Shared experiment code

| File | Status | Role |
|---|---|---|
| `src/cheatdetect/data/feature_schema.py` | new | Fixed 25-source / 33-output schema and preprocessing blocks |
| `src/cheatdetect/data/paired.py` | new | Aligned flat + sequence extraction from shared realized parent windows |
| `src/cheatdetect/data/preprocessing.py` | new | `FeaturePreprocessor` recipes, `gamma_center`, `validate_features` |
| `src/cheatdetect/data/dataset.py` | new | Manifest-guarded `PreparedStudy` builder, save/load, `input_manifest` |
| `src/cheatdetect/data/__init__.py` | tracked | Exports the above public API |
| `src/cheatdetect/experiments/__init__.py` | new | Experiment framework package marker (generic) |
| `src/cheatdetect/experiments/experiment_1/__init__.py` | new | Experiment 1 subpackage marker |
| `src/cheatdetect/experiments/experiment_1/audit.py` | new | Original-training audit (Phase 1) entry point |
| `src/cheatdetect/config.py` | tracked | Study/experiment path constants |

## Tests

| File | Status | Role |
|---|---|---|
| `tests/test_recipe_specification.py` | new | `FeaturePreprocessor` matches the frozen Phase 1 spec, all recipes × scale |
| `tests/test_dataset.py` | new | Paired extraction, fit-once, manifest invalidation, isolation, exclusions |
| `tests/test_audit.py` | new | Audit statistics, signed log, gamma geometry, train-only reads |
| `tests/test_preprocessing.py` | new | Imputation/transform/scaling/encoding unit coverage |
| `tests/test_paired.py` | new | Parent/micro alignment and identity coverage |
| `tests/test_mouse_features.py` | new | Feature-extraction regressions (angle wrapping) |

## Documentation

| File | Status | Role |
|---|---|---|
| `src/cheatdetect/experiments/experiment_1/brief.md` | new | This experiment's aim and protocol summary |
| `src/cheatdetect/experiments/experiment_1/files.md` | new | This inventory |

## Shared pipeline entry points

| File | Status | Role |
|---|---|---|
| `src/cheatdetect/pipeline/search.py` | new | Validation-only search used by the runner and production |
| `src/cheatdetect/pipeline/report.py` | new | Test reporting entry point (called after selection freezes) |

## Planned (Study Phase 3+)

| File | Status | Role |
|---|---|---|
| `src/cheatdetect/experiments/experiment_1/config.py` | planned | Frozen `StudyConfig`: recipes, grids, seeds, budgets |
| `src/cheatdetect/experiments/experiment_1/runner.py` | planned | Validation-only grid runner; per-candidate artifacts |
| `src/cheatdetect/experiments/experiment_1/reporting.py` | planned | Final test reporting behind a frozen manifest |
| `src/cheatdetect/experiments/__main__.py` | planned | Shared CLI (`python -m cheatdetect.experiments --phase N`) |

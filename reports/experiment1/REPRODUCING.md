# Inspecting and reproducing Experiment 1

## Read the original results

The [final report](README.md), copied tables, and figures can be read without
Python, raw sessions, or retraining. Original artifacts are preserved under
`study/artifacts/`; the [inventory](../../study/README.md) explains each stage.
From the repository root, verify their recorded bytes with:

```bash
sha256sum --check study/SHA256SUMS
```

The checksums cover the original artifact files and frozen split metadata.
They verify file integrity, not statistical validity or that reruns will be
byte-identical. Original manifests and fitted models have not been rewritten.
The snapshot commit identifies preserved files; it does not establish that the
numerical run originally executed from a clean working tree at that commit.

## Environment and private data

Use Python 3.12 and the checked-in `uv.lock`. To use the project's custom
environment consistently, run these commands from the repository root and retain
`UV_PROJECT_ENVIRONMENT` in the shell for subsequent `uv run` commands:

```bash
export UV_PROJECT_ENVIRONMENT=cheatdetect
uv venv cheatdetect
uv sync --locked
uv run pytest
```

The default development dependencies include CPU PyTorch, needed to train the
LSTM autoencoder and load its fitted artifacts. The original prepared manifest
records NumPy/pandas/scikit-learn versions, but not a complete historical
Python/PyTorch/hardware environment; exact rerun identity is not guaranteed.

Full reproduction also requires the original private CSV sessions at:

- `dataset/raw/pure normal/`
- `dataset/raw/mixed/`

Raw data is deliberately ignored. Keep a separate private backup outside this
repository before cleaning local files; this change does not create that backup.
Restore original filenames and contents: manifests hash the inputs, and split
assignments are frozen. The preserved `dataset/processed/split_info.json` is
required by the audit and is checked against the seed-42 assignment.

The prepared bundle contains derived session features and labels, so keeping raw
CSVs outside Git does not mean the snapshot contains no session-derived data.

## Study stages and rerunning

| Stage | Purpose | Entry point |
|---|---|---|
| 1 | Original normal-training audit | `python -m cheatdetect.experiments.experiment_1.audit` |
| 2 | Earlier data-readiness check | `study/notebooks/data_readiness.py` (historical notebook) |
| 3 | Build the final prepared bundle | CLI `--phase 3` |
| 4 | Validation search and lock | CLI `--phase 4` |
| 5 | Repeat locked finalists at predefined seeds | CLI `--phase 5` |
| 6 | Evaluate locked models and write test reports | CLI `--phase 6` |

Use a separate checkout for reproduction. Audit and stage 3 write their default
snapshot directories; a new `--run-dir` only isolates stages 4–6. Do not regenerate
original outputs merely to inspect them. These commands perform the full study
and include expensive fitting:

```bash
uv run python -m cheatdetect.experiments.experiment_1.audit
uv run python -m cheatdetect.experiments --phase 3
uv run python -m cheatdetect.experiments --phase 4 --run-dir study/artifacts/experiment_1_reproduction
uv run python -m cheatdetect.experiments --phase 5 --run-dir study/artifacts/experiment_1_reproduction
uv run python -m cheatdetect.experiments --phase 6 --run-dir study/artifacts/experiment_1_reproduction
```

New run directories remain ignored. Selection uses validation data; test results
must not guide new model choices. There is no automatic deployment promotion.

## Portability and notebooks

Original reporting manifests contain absolute model paths from the run machine.
They are retained as evidence. Inspecting tables needs no path changes. To execute
reporting elsewhere, perform a fresh run in the separate checkout above; simply
copying the original manifest is not a portable model-loading procedure.
`runner.load_study` also checks raw-file hashes, source hashes, and package
versions, so having `prepared.joblib` alone is insufficient for its guarded load.
The earlier stage-2 bundle may reference modules that were subsequently moved;
it is historical evidence rather than the current CLI input.

Both `.py` and `.ipynb` notebook sources are preserved. Edit only the `.py` twin,
then run `uv run jupytext --sync path/to/notebook.py`. Notebook execution is
optional and can write files: the readiness notebook rebuilds stage 2, the EDA
and final-report notebooks export figures/tables, and the stage-6 notebook scores
and writes reports again. The study-run and sensitivity notebooks skip fitting
when their persisted completion markers exist. Read saved outputs first.

Engineering phases A–E in the [integration plan](../../plans/experiment_1_phases.md)
are development history, separate from these numbered study stages.

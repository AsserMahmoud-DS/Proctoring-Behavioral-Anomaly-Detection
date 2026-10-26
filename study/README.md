# Experiment 1 — Preserved research materials

This directory preserves the original research evidence for the publication
snapshot. Start with the [final report](../reports/experiment1/README.md);
execution code lives in `src/cheatdetect/experiments/experiment_1/` and shared
modules under `src/cheatdetect/`.

| Contents | Role |
|---|---|
| `plans/experiment_plan.md` | Scientific protocol and planned comparisons |
| `plans/phase_01_findings.md` | Original-training audit findings |
| `plans/phase_02_readiness.md` | Earlier data/preprocessing readiness findings |
| `notebooks/original_training_audit.py` + `.ipynb` | Audit viewer |
| `notebooks/data_readiness.py` + `.ipynb` | Historical readiness execution and checks |
| `notebooks/eda.py` + `.ipynb` | Original-training EDA presentation/export |
| `notebooks/final_reports.py` + `.ipynb` | Final-variant presentation/export |
| `artifacts/phase_01_audit/` | Audit tables and frozen session assignment |
| `artifacts/phase_02_readiness/` | Earlier readiness bundle, retained as historical evidence |
| `artifacts/phase_03_study/` | Final prepared bundle, preprocessing information, and manifest |
| `artifacts/phase_04_study/` | Candidates, comparisons, fitted winners, reporting manifest |
| `artifacts/phase_04_study/phase_05_seed_sensitivity/` | Eight fitted seed-repeat artifacts |
| `artifacts/phase_04_study/phase_06_reporting/` | Original test tables and grouped figures |
| `SHA256SUMS` | Checksums for original artifacts and frozen split metadata |

The stage-2 and stage-3 bundles are distinct snapshots; neither is deleted or
silently replaced. Historical plans retain their earlier checkboxes, module
locations, and tracking statements. Those describe development at that time;
consult the final report for actual results and the
[engineering phase plan](../plans/experiment_1_phases.md) for integration history.

Both forms of each notebook are preserved. Edit the Python twin and Jupytext-sync
it; execution can overwrite outputs. The current execution notebooks are also
preserved under `src/cheatdetect/experiments/experiment_1/notebooks/`.

Logs, Python caches, and future run directories remain ignored. The ignored
`study/eda/` and `study/final_reports/` exports duplicate the committed report
figures/tables byte for byte at snapshot preparation time. Their generator
notebooks are retained, and no local exports have been deleted.

For environment setup, private raw-data requirements, absolute-path limitations,
and safe rerun instructions, see the
[reproduction guide](../reports/experiment1/REPRODUCING.md).

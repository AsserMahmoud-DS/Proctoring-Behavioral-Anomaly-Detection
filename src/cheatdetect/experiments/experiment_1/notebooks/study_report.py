# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: cheatdetect
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiment 1 — Final reporting (Phase 6)
#
# Scores the frozen 20 primary approaches plus the 8 predefined seed repeats on
# the held-out test split, once. Requires the frozen Phase 4 manifest and the
# Phase 5 sensitivity artifacts (`reporting.run_phase6` fails closed otherwise).
# No model selection or test-driven tuning happens here.
#
# Tables and grouped figures are persisted by `run_phase6`; this notebook is a
# thin viewer over those artifacts.

# %%
import pandas as pd
from IPython.display import Image, display

from cheatdetect.experiments.experiment_1 import reporting, runner
from cheatdetect.experiments.experiment_1.config import StudyConfig

config = StudyConfig()
study = runner.load_study(config)
run_dir = runner.DEFAULT_RUN_DIR
results, summary = reporting.run_phase6(study, run_dir=run_dir)
display(summary)

# %%
report_dir = run_dir / reporting.REPORT_DIR_NAME
display(pd.read_csv(report_dir / "dataset_summary.csv"))

# %% [markdown]
# ## Grouped figures
#
# Panels are grouped by model family and augmentation condition. Confusion
# matrices use the seed-42 primary artifacts only and are never averaged across
# seeds. PR curves and score distributions are saved alongside.

# %%
figures = report_dir / reporting.FIGURE_DIR_NAME
for path in sorted(figures.glob("roc_*.png")) + sorted(figures.glob("confusion_*.png")):
    display(Image(filename=str(path)))
print(f"All figures written under {figures}")

# %% [markdown]
# ## Seed sensitivity
#
# Descriptive only: all three seeds, no best-seed selection.

# %%
sensitivity = pd.DataFrame(
    [
        {
            "id": row["model"],
            "seed": row["seed"],
            "family": row["family"],
            "recipe": row["recipe"],
            "condition": "aug_on" if row["augmented"] else "aug_off",
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "threshold": row["threshold"],
        }
        for row in results
        if row["seed"] is not None
    ]
)
display(sensitivity)

# %% [markdown]
# ## Search and training summary
#
# Candidate counts, failures, runtime, and AE early-stopping status come from
# the Phase 4 candidates table.

# %%
candidates = pd.read_csv(run_dir / "candidates.csv")
display(
    candidates.groupby("family").agg(
        candidates=("status", "size"),
        failures=("status", lambda status: int((status == "failed").sum())),
        max_seconds=("seconds", "max"),
    )
)

ae_ok = candidates[(candidates["family"] == "ae") & (candidates["status"] == "ok")].copy()
ae_ok["hit_epoch_cap"] = ae_ok["epochs_trained"] >= config.lstm_epochs
display(
    ae_ok.groupby(["recipe", "augmented"]).agg(
        candidates=("status", "size"),
        epoch_cap_fraction=("hit_epoch_cap", "mean"),
        max_epochs=("epochs_trained", "max"),
        max_updates=("optimizer_updates", "max"),
    )
)

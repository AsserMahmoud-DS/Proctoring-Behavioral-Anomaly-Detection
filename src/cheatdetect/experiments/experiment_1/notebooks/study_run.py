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
# # Experiment 1 — Study run (Phases 3–4)
#
# Builds the frozen prepared bundle (Phase 3) and runs the validation-only
# paired search that locks the 18 recipe winners + 2 ensembles and freezes the
# reporting manifest (Phase 4). No test metric is computed here.
#
# Heavy runs are better executed headlessly:
# `uv run python -m cheatdetect.experiments --phase 4`. This notebook detects a
# completed run and only reviews the persisted artifacts instead of refitting.

# %%
import json

import pandas as pd
from IPython.display import display

from cheatdetect.experiments.experiment_1 import runner
from cheatdetect.experiments.experiment_1.config import StudyConfig

config = StudyConfig()
run_dir = runner.DEFAULT_RUN_DIR

# %% [markdown]
# ## Execution (skipped when a run already exists)

# %%
if (run_dir / "reporting_manifest.json").exists():
    print(
        f"Existing run found in {run_dir}; skipping Phase 3-4. "
        "Delete the directory to re-run."
    )
else:
    study = runner.build_study(config)
    run = runner.run_search(study, config)

# %% [markdown]
# ## Review — prepared bundle

# %%
study = runner.load_study(config)
display(
    study.session_report.groupby("partition", sort=False).agg(
        assigned_session_files=("session_file", "size"),
        contributing_session_files=("included", "sum"),
        parent_windows=("parent_windows", "sum"),
    )
)

# %% [markdown]
# ## Review — locked winners

# %%
manifest = json.loads((run_dir / "reporting_manifest.json").read_text())
winners = pd.DataFrame(
    [
        {
            "id": entry["id"],
            "family": entry["family"],
            "recipe": entry["recipe"],
            "condition": "aug_on" if entry["augmented"] else "aug_off",
            "roc_auc": entry["roc_auc"],
            "pr_auc": entry["pr_auc"],
            "threshold": entry["threshold"],
            "fallback_used": entry["threshold_info"]["fallback_used"],
        }
        for entry in manifest["primary"]
    ]
)
display(winners)

# %% [markdown]
# Missing cells (no valid candidate) must be resolved before Phase 5/6.

# %%
missing = pd.DataFrame(manifest.get("missing_cells", []))
if len(missing):
    print("WARNING: cells with no valid candidate:")
    display(missing)
else:
    print("All expected recipe/condition cells locked.")

# %% [markdown]
# ## Review — top candidates per family

# %%
candidates = pd.read_csv(run_dir / "candidates.csv")
ok = candidates[candidates["status"] == "ok"]
for family in ("if", "ocsvm", "ae"):
    print(f"=== {family.upper()} — top 5 candidates by validation ROC-AUC ===")
    display(
        ok[ok["family"] == family]
        .sort_values("roc_auc", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

# %% [markdown]
# ## Review — matched comparisons
#
# Matched augmentation pairs (identical configs off/on), independently tuned
# condition winners, and the AE depth comparison are persisted by the runner.

# %%
display(pd.read_csv(run_dir / "augmentation_matched.csv"))
display(pd.read_csv(run_dir / "augmentation_winners.csv"))
display(pd.read_csv(run_dir / "depth_comparison.csv"))

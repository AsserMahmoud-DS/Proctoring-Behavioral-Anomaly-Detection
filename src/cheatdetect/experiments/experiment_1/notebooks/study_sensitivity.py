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
# # Experiment 1 — Seed sensitivity (Phase 5)
#
# Repeats each augmentation condition's overall IF and AE finalists at seeds
# 123 and 2026 (eight fits) with fixed split, augmentation, and frozen
# preprocessing. Each repeat gets its own validation threshold and AE
# checkpoint. Never select a best seed; the seed-42 artifacts stay untouched.
#
# Headless alternative: `uv run python -m cheatdetect.experiments --phase 5`.

# %%
import json

import pandas as pd
from IPython.display import display

from cheatdetect.experiments.experiment_1 import runner
from cheatdetect.experiments.experiment_1.config import StudyConfig

config = StudyConfig()
manifest_path = runner.DEFAULT_RUN_DIR / "reporting_manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(
        "Run Phase 4 first (study_run notebook or CLI --phase 4)"
    )

# %% [markdown]
# ## Execution (skipped when the sensitivity artifacts already exist)

# %%
manifest = json.loads(manifest_path.read_text())
if manifest.get("sensitivity_frozen"):
    print("Existing sensitivity run found; skipping Phase 5.")
else:
    study = runner.load_study(config)
    runner.run_seed_repeats(study, config)
    manifest = json.loads(manifest_path.read_text())

# %%
table = pd.DataFrame(
    [
        {
            "id": entry["id"],
            "seed": entry["seed"],
            "family": entry["family"],
            "recipe": entry["recipe"],
            "condition": "aug_on" if entry["augmented"] else "aug_off",
            "roc_auc": entry["roc_auc"],
            "pr_auc": entry["pr_auc"],
            "threshold": entry["threshold"],
        }
        for entry in manifest["sensitivity"]
    ]
).sort_values(["family", "condition", "seed"])
display(table)

# %%
# Descriptive spread only — no best-seed selection, no artifact replacement.
display(
    table.groupby(["family", "condition"])["roc_auc"].agg(
        ["mean", "std", "min", "max"]
    )
)

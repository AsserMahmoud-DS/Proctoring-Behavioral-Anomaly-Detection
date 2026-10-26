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
# # Study final reports — locked variants
#
# Reports the four final Experiment 1 variants on the held-out test split,
# exactly as locked in the frozen Phase 4 manifest:
#
# - `if:IF-raw:aug_on`
# - `ocsvm:SVM-base:aug_on`
# - `ae:AE-log:aug_on`
# - `ensemble:IF-raw+SVM-base:aug_on` (IF weight 0.7 / OCSVM 0.3)
#
# No refitting, no re-selection: artifacts and thresholds are read from the
# frozen manifest. Visuals are saved under `study/final_reports/`.

# %%
import json
from pathlib import Path

import joblib
import pandas as pd
from IPython.display import Image, display

from cheatdetect.eval import (
    evaluate_model,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
)
from cheatdetect.experiments.experiment_1 import runner
from cheatdetect.experiments.experiment_1.config import StudyConfig
from cheatdetect.utils import _find_project_root

FINAL_VARIANTS = [
    "if:IF-raw:aug_on",
    "ocsvm:SVM-base:aug_on",
    "ae:AE-log:aug_on",
    "ensemble:IF-raw+SVM-base:aug_on",
]

REPORT_DIR = _find_project_root() / "study" / "final_reports"
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

config = StudyConfig()
study = runner.load_study(config)
run_dir = runner.DEFAULT_RUN_DIR
manifest = json.loads((run_dir / "reporting_manifest.json").read_text())

# %% [markdown]
# ## Frozen variant selection

# %%
entries = {entry["id"]: entry for entry in manifest["primary"]}
missing = [variant for variant in FINAL_VARIANTS if variant not in entries]
if missing:
    raise ValueError(f"Final variants missing from the frozen manifest: {missing}")

pd.DataFrame(
    [
        {
            "id": entry["id"],
            "family": entry["family"],
            "recipe": entry["recipe"],
            "augmented": entry["augmented"],
            "if_weight": entry.get("if_weight"),
            "validation_roc_auc": entry["roc_auc"],
            "threshold": entry["threshold"],
            "fallback_used": entry["threshold_info"]["fallback_used"],
        }
        for entry in (entries[variant] for variant in FINAL_VARIANTS)
    ]
)

# %% [markdown]
# ## Test evaluation (frozen thresholds)

# %%
raw = study.raw["mixed_test"]
flat_x, sequence_x, y_test = raw.parent, raw.sequences, raw.labels

results = []
for variant in FINAL_VARIANTS:
    entry = entries[variant]
    detector = joblib.load(entry["artifact"])
    X = sequence_x if entry["family"] == "ae" else flat_x
    result = evaluate_model(
        entry["id"], detector.decision_function(X), float(entry["threshold"]), y_test
    )
    result.update(
        {
            "family": entry["family"],
            "recipe": entry["recipe"],
            "if_weight": entry.get("if_weight"),
        }
    )
    results.append(result)

pd.DataFrame(
    [
        {
            "id": result["model"],
            "family": result["family"],
            "recipe": result["recipe"],
            "if_weight": result["if_weight"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "pr_auc": result["pr_auc"],
            "pr_auc_baseline": result["pr_auc_baseline"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "tp": result["tp"],
            "tn": result["tn"],
            "fp": result["fp"],
            "fn": result["fn"],
        }
        for result in results
    ]
).to_csv(REPORT_DIR / "final_metrics.csv", index=False)

display(pd.read_csv(REPORT_DIR / "final_metrics.csv"))

# %% [markdown]
# ## Per-variant visuals — ROC, PR, confusion matrix

# %%
for result in results:
    safe = result["model"].replace(":", "_").replace("+", "_")
    scores = {result["model"]: result["scores"]}
    plot_roc_curves(scores, y_test, FIGURE_DIR / f"roc_{safe}.png")
    plot_pr_curves(scores, y_test, FIGURE_DIR / f"pr_{safe}.png")
    plot_confusion_matrices([result], FIGURE_DIR / f"confusion_{safe}.png")
    print(f"=== {result['model']} ===")
    display(Image(filename=str(FIGURE_DIR / f"roc_{safe}.png")))
    display(Image(filename=str(FIGURE_DIR / f"pr_{safe}.png")))
    display(Image(filename=str(FIGURE_DIR / f"confusion_{safe}.png")))

# %% [markdown]
# ## Combined comparisons — ROC and PR

# %%
all_scores = {result["model"]: result["scores"] for result in results}
plot_roc_curves(all_scores, y_test, FIGURE_DIR / "roc_comparison.png")
plot_pr_curves(all_scores, y_test, FIGURE_DIR / "pr_comparison.png")
display(Image(filename=str(FIGURE_DIR / "roc_comparison.png")))
display(Image(filename=str(FIGURE_DIR / "pr_comparison.png")))
print(f"All final-report figures written under {FIGURE_DIR}")

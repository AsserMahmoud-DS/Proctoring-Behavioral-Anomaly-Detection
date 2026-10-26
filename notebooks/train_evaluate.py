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
# # Training & Evaluation
#
# Runs the full experiment via `train_pipeline` (data prep → model selection
# → threshold tuning → test evaluation → serialization) and plots the
# validation and test results.

# %%
from IPython.display import Image, display

from cheatdetect.config import TEST_DIR, VAL_DIR, ExperimentConfig
from cheatdetect.eval import (
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distributions,
)
from cheatdetect.pipeline import train_pipeline

config = ExperimentConfig(lstm_enabled=True)

# %%
# ============================================================
# Train + evaluate (serializes the best model to models/)
# ============================================================
results = train_pipeline(config)

val_scores = results["val_scores"]
y_val = results["y_val"]

test_results = results["test_results"]
test_scores = {r["model"]: r["scores"] for r in test_results}
y_test = results["y_test"]

# %%
# ============================================================
# Grid Search Results — Top Configs by Validation ROC-AUC
# ============================================================
print(f"Validation PR-AUC prevalence baseline: {y_val.mean():.4f}")
print("=== Isolation Forest — Top 5 configs by ROC-AUC ===")
display(results["if_grid_results"].head(5))

print("\n=== One-Class SVM — Top 5 configs by ROC-AUC ===")
display(results["ocsvm_grid_results"].head(5))

print("\n=== Ensemble — Weight sweep by ROC-AUC ===")
display(results["ensemble_grid_results"].head(5))

print("\n=== LSTM-AE — Top 5 configs by ROC-AUC ===")
display(results["lstm_grid_results"].head(5))

# %%
# ============================================================
# Validation — PR, ROC Curves & Score Distributions
# ============================================================
plot_pr_curves(val_scores, y_val, VAL_DIR / "pr_curves.png")
display(Image(filename=str(VAL_DIR / "pr_curves.png")))

plot_roc_curves(val_scores, y_val, VAL_DIR / "roc_curves.png")
display(Image(filename=str(VAL_DIR / "roc_curves.png")))

plot_score_distributions(val_scores, y_val, VAL_DIR / "score_distributions.png")
display(Image(filename=str(VAL_DIR / "score_distributions.png")))

# %%
# ============================================================
# Final Evaluation — Test Set
# ============================================================
display(results["metrics_df"])

best_name = results["best_name"]
best_test = results["metrics_df"].loc[best_name]
print(
    f"=== Selected Model: {best_name} "
    f"(Validation ROC-AUC = {results['selection_score_val']:.4f}) ==="
)
print(
    f"Test ROC-AUC = {best_test['roc_auc']:.4f}; "
    f"PR-AUC (average precision) = {best_test['pr_auc']:.4f}; "
    f"prevalence baseline = {best_test['pr_auc_baseline']:.4f}"
)

# %%
# ============================================================
# Test Plots — PR, ROC, Confusion Matrices
# ============================================================
plot_pr_curves(test_scores, y_test, TEST_DIR / "pr_curves.png")
display(Image(filename=str(TEST_DIR / "pr_curves.png")))

plot_roc_curves(test_scores, y_test, TEST_DIR / "roc_curves.png")
display(Image(filename=str(TEST_DIR / "roc_curves.png")))

plot_confusion_matrices(test_results, TEST_DIR / "confusion_matrices.png")
display(Image(filename=str(TEST_DIR / "confusion_matrices.png")))

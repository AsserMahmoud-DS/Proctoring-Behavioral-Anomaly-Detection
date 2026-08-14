"""Plotting utilities for model evaluation.

Each function draws a matplotlib figure and saves it to disk. Figures are
not shown here — the caller (e.g. the notebook) may call ``plt.show()``
afterward for inline display.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc


def plot_pr_curves(model_scores: dict, y_true, save_path: Path) -> None:
    """Precision-recall curves for every model in *model_scores*."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, scores in model_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax.plot(rec, prec, linewidth=2, label=f"{label} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(model_scores: dict, y_true, save_path: Path) -> None:
    """Side-by-side histograms of scores for normal vs cheating chunks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, scores in model_scores.items():
        normal_scores = scores[y_true == 0]
        cheat_scores = scores[y_true == 1]
        axes[0].hist(normal_scores, bins=30, alpha=0.5, density=True, label=f"{label} normal")
        axes[1].hist(cheat_scores, bins=30, alpha=0.5, density=True, label=f"{label} cheat")
    axes[0].set_title("Score Distribution — Normal")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Anomaly Score")
    axes[1].set_title("Score Distribution — Cheating")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Anomaly Score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(model_scores: dict, y_true, save_path: Path) -> None:
    """ROC curves for every model in *model_scores*."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(test_results: list[dict], save_path: Path) -> None:
    """Confusion-matrix heatmaps for each result in *test_results*."""
    n = len(test_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for i, res in enumerate(test_results):
        cm = res["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Cheat"],
            yticklabels=["Normal", "Cheat"],
            ax=axes[i],
            cbar=False,
        )
        axes[i].set_title(
            f"{res['model']}\nTP={res['tp']}  TN={res['tn']}  "
            f"FP={res['fp']}  FN={res['fn']}",
            fontsize=10,
        )
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

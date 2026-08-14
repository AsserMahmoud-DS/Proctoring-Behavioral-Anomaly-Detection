"""Model evaluation: metrics and plots."""

from .metrics import compare_models, evaluate_model
from .plots import (
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distributions,
)

__all__ = [
    "evaluate_model",
    "compare_models",
    "plot_pr_curves",
    "plot_score_distributions",
    "plot_roc_curves",
    "plot_confusion_matrices",
]

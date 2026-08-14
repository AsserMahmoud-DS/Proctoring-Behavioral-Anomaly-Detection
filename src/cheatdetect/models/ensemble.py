"""Weighted ensemble of Isolation Forest and One-Class SVM detectors."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .base import AnomalyDetector


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to ``[0, 1]``; returns zeros if constant."""
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return np.zeros_like(scores)


class EnsembleDetector(AnomalyDetector):
    """Combine IF and OCSVM scores via a weighted average of normalized scores."""

    def __init__(
        self,
        if_detector: AnomalyDetector,
        ocsvm_detector: AnomalyDetector,
        if_weight: float = 0.5,
    ):
        self.if_detector = if_detector
        self.ocsvm_detector = ocsvm_detector
        self.if_weight = if_weight

    def fit(self, X: pd.DataFrame) -> "EnsembleDetector":
        self.if_detector.fit(X)
        self.ocsvm_detector.fit(X)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if_scores = normalize_scores(self.if_detector.decision_function(X))
        ocsvm_scores = normalize_scores(self.ocsvm_detector.decision_function(X))
        return self.if_weight * if_scores + (1 - self.if_weight) * ocsvm_scores


def grid_search(
    if_detector: AnomalyDetector,
    ocsvm_detector: AnomalyDetector,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    weights: list[float] | tuple[float, ...],
) -> tuple[EnsembleDetector, pd.DataFrame]:
    """Sweep the IF weight and return the best ensemble by validation PR-AUC.

    Takes already-fitted sub-detectors — the weight does not change the
    underlying models, only how their scores are combined.

    Returns:
        ``(best_ensemble, results_df)`` where results are sorted by PR-AUC.
    """
    results = []
    for w in weights:
        ensemble = EnsembleDetector(if_detector, ocsvm_detector, if_weight=w)
        pr_auc = average_precision_score(y_val, ensemble.decision_function(X_val))
        results.append({"if_weight": w, "ocsvm_weight": 1 - w, "pr_auc": pr_auc})

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    best_w = results_df.iloc[0]["if_weight"]
    best_ensemble = EnsembleDetector(if_detector, ocsvm_detector, if_weight=best_w)
    return best_ensemble, results_df

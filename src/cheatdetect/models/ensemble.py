"""Weighted ensemble of Isolation Forest and One-Class SVM detectors."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from .base import AnomalyDetector, validate_selection_labels


class EnsembleDetector(AnomalyDetector):
    """Combine anomaly scores using a persisted median/IQR score scaler."""

    def __init__(
        self,
        if_detector: AnomalyDetector,
        ocsvm_detector: AnomalyDetector,
        if_weight: float = 0.5,
    ):
        if not np.isfinite(if_weight) or not 0 <= if_weight <= 1:
            raise ValueError("if_weight must be finite and between 0 and 1")
        self.if_detector = if_detector
        self.ocsvm_detector = ocsvm_detector
        self.if_weight = if_weight
        self.score_scaler: RobustScaler | None = None

    def fit(self, X: pd.DataFrame) -> "EnsembleDetector":
        """Fit both detectors and normalize their scores on the training data.

        Use ``fit_score_normalizer`` on held-out normal data to calibrate
        already-selected detectors without retraining them.
        """
        self.score_scaler = None
        self.if_detector.fit(X)
        self.ocsvm_detector.fit(X)
        return self.fit_score_normalizer(X)

    def _score_matrix(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.column_stack(
            [
                self.if_detector.decision_function(X),
                self.ocsvm_detector.decision_function(X),
            ]
        ).astype(float, copy=False)
        if not np.isfinite(scores).all():
            raise ValueError("Ensemble detector scores must be finite")
        return scores

    def fit_score_normalizer(self, X_ref: pd.DataFrame) -> "EnsembleDetector":
        """Freeze score statistics from unaugmented held-out normal windows."""
        if len(X_ref) == 0:
            raise ValueError("Score normalization requires nonempty reference data")
        scores = self._score_matrix(X_ref)
        quartiles = np.percentile(scores, [25, 75], axis=0)
        score_iqr = quartiles[1] - quartiles[0]
        tolerance = 10 * np.finfo(scores.dtype).eps * np.max(np.abs(scores), axis=0)
        if np.any(score_iqr <= tolerance):
            raise ValueError(
                "Cannot normalize detector scores with zero or negligible reference IQR"
            )
        self.score_scaler = RobustScaler().fit(scores)
        return self

    def _normalized_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self.score_scaler is None:
            raise RuntimeError("Ensemble score normalizer is not fitted")
        return self.score_scaler.transform(self._score_matrix(X))

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._normalized_scores(X)
        return self.if_weight * scores[:, 0] + (1 - self.if_weight) * scores[:, 1]


def grid_search(
    if_detector: AnomalyDetector,
    ocsvm_detector: AnomalyDetector,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    weights: list[float] | tuple[float, ...],
    *,
    X_ref: pd.DataFrame,
) -> tuple[EnsembleDetector, pd.DataFrame]:
    """Sweep the IF weight and return the best ensemble by validation ROC-AUC.

    Takes already-fitted sub-detectors — the weight does not change the
    underlying models, only how their scores are combined.
    Score scaling is fitted once on ``X_ref`` (held-out pure-normal windows).
    Only ``X_val`` and ``y_val`` determine the selected weight.

    Returns:
        ``(best_ensemble, results_df)`` sorted by ROC-AUC, with PR-AUC reported.
    """
    validate_selection_labels(y_val)
    if not weights or any(
        not np.isfinite(weight) or not 0 <= weight <= 1 for weight in weights
    ):
        raise ValueError("weights must contain finite values between 0 and 1")
    ensemble = EnsembleDetector(if_detector, ocsvm_detector).fit_score_normalizer(X_ref)
    scores = ensemble._normalized_scores(X_val)
    results = []
    best_weight = weights[0]
    best_roc_auc = -np.inf
    for weight in weights:
        combined = weight * scores[:, 0] + (1 - weight) * scores[:, 1]
        roc_auc = roc_auc_score(y_val, combined)
        pr_auc = average_precision_score(y_val, combined)
        results.append({
            "if_weight": weight, "ocsvm_weight": 1 - weight,
            "roc_auc": roc_auc, "pr_auc": pr_auc,
        })
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_weight = weight

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False, kind="stable"
    )
    ensemble.if_weight = best_weight
    return ensemble, results_df

"""Abstract base class shared by all anomaly detectors."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def validate_selection_labels(y_val: np.ndarray) -> None:
    """ROC-AUC selection requires both binary validation classes."""
    if not np.array_equal(np.unique(y_val), [0, 1]):
        raise ValueError(
            "ROC-AUC model selection requires both normal (0) and anomalous (1) validation labels"
        )


def is_fitted_preprocessor(preprocessor) -> bool:
    """True when a ``FeaturePreprocessor`` holds fit-time schema/statistics.

    Detectors accept an already-fitted preprocessor from the study runner so
    the frozen original-training statistics are reused instead of refit on
    augmented or held-out inputs.
    """
    return hasattr(preprocessor, "output_features_")


class AnomalyDetector(ABC):
    """Interface for all anomaly detectors.

    Concrete implementations must provide ``fit`` and ``decision_function``.
    ``decision_function`` must return scores where **higher = more anomalous**
    (the underlying sklearn estimators return the opposite sign, so each
    detector negates internally).
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        """Fit the detector on (normal) training data."""

    @abstractmethod
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores; higher is more anomalous."""

    def predict(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """Binary prediction: anomalous where score >= threshold."""
        return (self.decision_function(X) >= threshold).astype(int)


class SequenceAnomalyDetector(ABC):
    """Interface for sequence-based anomaly detectors.

    The flat :class:`AnomalyDetector` consumes one feature row per window.
    Temporal models (e.g. the LSTM autoencoder) instead consume a sequence of
    feature vectors per window: an array of shape
    ``(n_samples, seq_len, n_features)``.

    Like :class:`AnomalyDetector`, ``decision_function`` must return scores
    where **higher = more anomalous**.
    """

    @abstractmethod
    def fit(
        self, X: np.ndarray, X_es: np.ndarray | None = None
    ) -> "SequenceAnomalyDetector":
        """Fit the detector on (normal) training sequences.

        Args:
            X: Training sequences ``(n_samples, seq_len, n_features)``.
            X_es: Optional held-out **normal** sequences used for early
                stopping. Should come from the same distribution as ``X``.
                When ``None``, training runs the full fixed epoch budget
                without early stopping.
        """

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores; higher is more anomalous."""

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Binary prediction: anomalous where score >= threshold."""
        return (self.decision_function(X) >= threshold).astype(int)

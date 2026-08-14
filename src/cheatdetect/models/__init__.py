"""Anomaly detection models.

Provides a shared :class:`AnomalyDetector` interface, concrete Isolation
Forest and One-Class SVM detectors, a weighted ensemble, and threshold
tuning.
"""

from .base import AnomalyDetector
from .isolation_forest import IsolationForestDetector
from .ocsvm import OCSVMDetector
from .ensemble import EnsembleDetector, normalize_scores
from .threshold import tune_threshold

__all__ = [
    "AnomalyDetector",
    "IsolationForestDetector",
    "OCSVMDetector",
    "EnsembleDetector",
    "normalize_scores",
    "tune_threshold",
]

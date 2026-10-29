from cheatdetect.models import AnomalyDetector
import numpy as np


class FakeDetector(AnomalyDetector):
    """Returns predefined scores, ignoring input data."""

    def __init__(self, scores: list[float]):
        self._scores = iter(scores)

    def fit(self, X):
        return self

    def decision_function(self, X):
        return np.array([next(self._scores) for _ in range(len(X))])
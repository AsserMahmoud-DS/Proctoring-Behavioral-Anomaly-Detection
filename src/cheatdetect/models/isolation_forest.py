"""Isolation Forest anomaly detector."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from cheatdetect.data import Log1pSkewed

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest wrapped in a ``log1p(skewed) -> model`` pipeline."""

    def __init__(
        self,
        skewed_cols: list[str],
        n_estimators: int = 100,
        max_samples: int | float | str = "auto",
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.skewed_cols = list(skewed_cols)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.pipeline = Pipeline(
            [
                ("log1p", Log1pSkewed(self.skewed_cols)),
                (
                    "model",
                    IsolationForest(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        contamination=contamination,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        self.pipeline.fit(X)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        # sklearn returns lower = more anomalous; negate for higher = anomalous.
        return -self.pipeline.decision_function(X)

    @classmethod
    def grid_search(
        cls,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        skewed_cols: list[str],
        param_grid: dict,
        random_state: int = 42,
    ) -> tuple["IsolationForestDetector", pd.DataFrame]:
        """Search *param_grid* and return the best detector by validation PR-AUC.

        Args:
            X_train, X_val: training / validation feature matrices.
            y_val: binary labels (1 = anomalous).
            skewed_cols: columns for the ``Log1pSkewed`` transform.
            param_grid: dict of constructor params → list of candidates.
            random_state: seed for reproducibility.

        Returns:
            ``(best_detector, results_df)`` where results are sorted by PR-AUC.
        """
        results = []
        for params in ParameterGrid(param_grid):
            detector = cls(skewed_cols=skewed_cols, random_state=random_state, **params)
            detector.fit(X_train)
            pr_auc = average_precision_score(y_val, detector.decision_function(X_val))
            results.append({**params, "pr_auc": pr_auc})

        results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
        best_params = {k: results_df.iloc[0][k] for k in param_grid}
        best_detector = cls(skewed_cols=skewed_cols, random_state=random_state, **best_params)
        best_detector.fit(X_train)
        return best_detector, results_df

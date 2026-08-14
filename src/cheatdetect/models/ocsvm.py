"""One-Class SVM anomaly detector."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from cheatdetect.data import Log1pSkewed

from .base import AnomalyDetector


class OCSVMDetector(AnomalyDetector):
    """One-Class SVM wrapped in a ``log1p(skewed) -> RobustScaler -> model`` pipeline."""

    def __init__(
        self,
        skewed_cols: list[str],
        nu: float = 0.05,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        random_state: int = 42,
    ):
        self.skewed_cols = list(skewed_cols)
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state

        self.pipeline = Pipeline(
            [
                ("log1p", Log1pSkewed(self.skewed_cols)),
                ("scaler", RobustScaler()),
                ("model", OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)),
            ]
        )

    def fit(self, X: pd.DataFrame) -> "OCSVMDetector":
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
    ) -> tuple["OCSVMDetector", pd.DataFrame]:
        """Search *param_grid* and return the best detector by validation PR-AUC.

        See :meth:`IsolationForestDetector.grid_search` for parameter details.
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

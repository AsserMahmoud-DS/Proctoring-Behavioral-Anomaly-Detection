"""One-Class SVM anomaly detector."""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from cheatdetect.data import FeaturePreprocessor

from .base import AnomalyDetector, is_fitted_preprocessor
from .selection import sweep_candidates

OCSVM_RECIPES = ("base", "log", "yj", "quantile")


class OCSVMDetector(AnomalyDetector):
    """One-Class SVM wrapped in a fixed-schema ``FeaturePreprocessor -> model`` pipeline.

    The detector consumes the raw 25-feature behavioral schema and owns its
    preprocessing (including scaling), so the serialized artifact is
    self-contained.
    """

    def __init__(
        self,
        recipe: str = "base",
        nu: float = 0.05,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        tol: float = 1e-3,
        max_iter: int = -1,
        random_state: int = 42,
        preprocessor: FeaturePreprocessor | None = None,
    ):
        if recipe not in OCSVM_RECIPES:
            raise ValueError(
                f"Unknown OCSVM recipe '{recipe}'; expected one of {OCSVM_RECIPES}"
            )
        self.recipe = recipe
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else FeaturePreprocessor(recipe, scale=True)
        )

        self.pipeline = Pipeline(
            [
                ("preprocess", self.preprocessor),
                (
                    "model",
                    OneClassSVM(
                        nu=nu, gamma=gamma, kernel=kernel, tol=tol, max_iter=max_iter
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame) -> "OCSVMDetector":
        # An injected preprocessor is already fitted on original training; only
        # the estimator is trained on (possibly augmented) raw rows.
        if is_fitted_preprocessor(self.preprocessor):
            transformed = self.preprocessor.transform(X)
            self.pipeline.named_steps["model"].fit(transformed)
        else:
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
        param_grid: dict,
        random_state: int = 42,
        preprocessor: FeaturePreprocessor | None = None,
    ) -> tuple["OCSVMDetector | None", pd.DataFrame]:
        """Search *param_grid* and return the best detector by validation ROC-AUC.

        See :meth:`IsolationForestDetector.grid_search` for parameter details.
        An injected ``preprocessor`` is reused fitted; only the estimator is
        refit per candidate.
        """

        def fit_and_score(params: dict):
            detector = cls(
                random_state=random_state, preprocessor=preprocessor, **params
            )
            detector.fit(X_train)
            return detector, detector.decision_function(X_val), {}

        return sweep_candidates(ParameterGrid(param_grid), fit_and_score, y_val)

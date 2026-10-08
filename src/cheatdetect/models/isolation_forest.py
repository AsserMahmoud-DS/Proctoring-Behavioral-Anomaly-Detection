"""Isolation Forest anomaly detector."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from cheatdetect.data import FeaturePreprocessor

from .base import AnomalyDetector, is_fitted_preprocessor
from .selection import sweep_candidates

IF_RECIPES = ("base", "log")


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest wrapped in a fixed-schema ``FeaturePreprocessor -> model`` pipeline.

    The detector consumes the raw 25-feature behavioral schema and owns its
    preprocessing, so the serialized artifact is self-contained: the same
    fitted pipeline transforms training and inference inputs.
    """

    def __init__(
        self,
        recipe: str = "base",
        n_estimators: int = 100,
        max_samples: int | float | str = "auto",
        max_features: float = 1.0,
        contamination: float = 0.1,
        random_state: int = 42,
        preprocessor: FeaturePreprocessor | None = None,
    ):
        if recipe not in IF_RECIPES:
            raise ValueError(
                f"Unknown IF recipe '{recipe}'; expected one of {IF_RECIPES}"
            )
        self.recipe = recipe
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else FeaturePreprocessor(recipe, scale=False)
        )

        self.pipeline = Pipeline(
            [
                ("preprocess", self.preprocessor),
                (
                    "model",
                    IsolationForest(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        max_features=max_features,
                        contamination=contamination,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
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
    ) -> tuple["IsolationForestDetector | None", pd.DataFrame]:
        """Search *param_grid* and return the best detector by validation ROC-AUC.

        Args:
            X_train, X_val: raw 25-feature training / validation matrices.
            y_val: binary labels (1 = anomalous).
            param_grid: dict of constructor params → list of candidates.
            random_state: seed for reproducibility.
            preprocessor: optional pre-fitted processor; when given, the
                original-training statistics are frozen and only the estimator
                is refit per candidate.

        Returns:
            ``(best_detector, results_df)`` sorted by ROC-AUC, with PR-AUC and
            per-candidate status reported.
        """

        def fit_and_score(params: dict):
            detector = cls(
                random_state=random_state, preprocessor=preprocessor, **params
            )
            detector.fit(X_train)
            return detector, detector.decision_function(X_val), {}

        return sweep_candidates(ParameterGrid(param_grid), fit_and_score, y_val)

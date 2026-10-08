"""Isolation Forest anomaly detector."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from cheatdetect.data import FeaturePreprocessor

from .base import AnomalyDetector, validate_selection_labels

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

        self.pipeline = Pipeline(
            [
                ("preprocess", FeaturePreprocessor(recipe, scale=False)),
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
    ) -> tuple["IsolationForestDetector", pd.DataFrame]:
        """Search *param_grid* and return the best detector by validation ROC-AUC.

        Args:
            X_train, X_val: raw 25-feature training / validation matrices.
            y_val: binary labels (1 = anomalous).
            param_grid: dict of constructor params → list of candidates.
            random_state: seed for reproducibility.

        Returns:
            ``(best_detector, results_df)`` sorted by ROC-AUC, with PR-AUC reported.
        """
        validate_selection_labels(y_val)
        results = []
        best_detector: IsolationForestDetector | None = None
        best_roc_auc = -np.inf
        for params in ParameterGrid(param_grid):
            detector = cls(random_state=random_state, **params)
            detector.fit(X_train)
            scores = detector.decision_function(X_val)
            roc_auc = roc_auc_score(y_val, scores)
            pr_auc = average_precision_score(y_val, scores)
            results.append({**params, "roc_auc": roc_auc, "pr_auc": pr_auc})
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_detector = detector

        results_df = pd.DataFrame(results).sort_values(
            "roc_auc", ascending=False, kind="stable"
        )
        return best_detector, results_df

"""Feature-level transformations for modeling."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def find_skewed_features(
    X: pd.DataFrame, skew_threshold: float = 2.0
) -> list[str]:
    """Identify columns eligible for a log1p transform.

    A column is eligible when it is right-skewed (skew > *skew_threshold*)
    and strictly non-negative, since ``log1p`` is undefined for negative
    inputs.

    Args:
        X: Feature matrix. Rows are chunks, columns are features.
        skew_threshold: Minimum skew above which a column is considered
            skewed. Defaults to 2.0.

    Returns:
        List of column names to pass to :class:`Log1pSkewed`.
    """
    skew = X.skew()
    non_negative = X.min() >= 0
    return skew[(skew > skew_threshold) & non_negative].index.tolist()


class Log1pSkewed(BaseEstimator, TransformerMixin):
    """Apply log1p to skewed (non-negative) features, pass through the rest."""

    def __init__(self, skewed_cols: list[str]):
        self.skewed_cols = skewed_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.skewed_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(0))
        return X
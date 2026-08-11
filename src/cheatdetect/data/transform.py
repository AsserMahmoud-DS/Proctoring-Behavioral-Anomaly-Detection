"""Feature-level transformations for modeling."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
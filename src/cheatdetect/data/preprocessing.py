"""Reusable behavioral preprocessing; fit on original normal training before reuse."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .feature_schema import (
    CONTINUOUS, COUNT_LOG, COUNTS, DIRECTION, NUMERICAL, OUTPUT_FEATURES,
    SCALED, SIGNED, SOURCE_FEATURES,
)



def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the fixed schema and finite-value domains without discarding missingness."""
    if frame.columns.duplicated().any() or set(frame.columns) != set(SOURCE_FEATURES):
        raise ValueError("Behavioral input must contain exactly the 25 distinct source features")
    values = frame.loc[:, list(SOURCE_FEATURES)].astype(float)
    if not np.isin(values[DIRECTION], np.arange(9)).all():
        raise ValueError("Direction categories must be finite integers in 0..8")
    finite = values.where(np.isfinite(values))
    nonnegative = [feature for feature in NUMERICAL if feature not in SIGNED]
    if (finite[nonnegative] < -1e-12).any().any():
        raise ValueError("Negative value in a nonnegative source feature")
    bounded = finite[["mouse_straightness", "mouse_idle_time_ratio"]]
    if (bounded > 1 + 1e-12).any().any():
        raise ValueError("Ratio source features must be within [0,1]")
    count_values = finite[list(COUNTS)]
    if ((count_values - count_values.round()).abs() > 1e-12).any().any():
        raise ValueError("Count source features must be integers")
    return values


class FeaturePreprocessor(TransformerMixin, BaseEstimator):
    """Train-median imputation, selective transforms, scaling, and fixed encoding."""

    def __init__(self, recipe: str = "base", scale: bool = True):
        self.recipe = recipe
        self.scale = scale

    def fit(self, X: pd.DataFrame, y=None):
        if self.recipe not in {"base", "log", "yj", "quantile"}:
            raise ValueError(f"Unknown preprocessing recipe: {self.recipe}")
        values = validate_features(X)
        if values.empty:
            raise ValueError("Preprocessing requires original training rows")
        numerical = values[list(NUMERICAL)].replace([np.inf, -np.inf], np.nan)
        self.medians_ = numerical.median()
        if self.medians_.isna().any():
            raise ValueError("Cannot impute a source feature with no finite training values")
        numerical = numerical.fillna(self.medians_)
        self.constants_ = numerical.nunique().eq(1)
        self.transformers_ = {}
        self.n_original_rows_ = len(values)
        if self.recipe in {"yj", "quantile"}:
            for feature in CONTINUOUS:
                if self.constants_[feature]:
                    continue
                transformer = (
                    PowerTransformer(method="yeo-johnson", standardize=False)
                    if self.recipe == "yj" else QuantileTransformer(
                        n_quantiles=min(100, len(values)), output_distribution="normal",
                        subsample=None, random_state=42,
                    )
                )
                transformer.fit(numerical[[feature]])
                self.transformers_[feature] = transformer
        transformed = self._nonlinear(numerical)
        self.scaler_ = None
        if self.scale:
            self.scaler_ = StandardScaler().fit(transformed[list(SCALED)])
        self.feature_names_in_ = np.array(SOURCE_FEATURES, dtype=object)
        self.n_features_in_ = len(SOURCE_FEATURES)
        self.output_features_ = list(OUTPUT_FEATURES)
        self.transform(X)
        return self

    def _nonlinear(self, numerical: pd.DataFrame) -> pd.DataFrame:
        transformed = numerical.copy()
        if self.recipe == "log":
            positive = [feature for feature in CONTINUOUS if feature not in SIGNED]
            transformed[positive] = np.log1p(transformed[positive])
            transformed[list(SIGNED)] = np.sign(transformed[list(SIGNED)]) * np.log1p(
                np.abs(transformed[list(SIGNED)])
            )
        elif self.recipe in {"yj", "quantile"}:
            for feature, transformer in self.transformers_.items():
                transformed[feature] = transformer.transform(transformed[[feature]]).ravel()
        if self.recipe != "base":
            transformed[list(COUNT_LOG)] = np.log1p(transformed[list(COUNT_LOG)])
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError(f"Nonfinite values after {self.recipe} transform")
        return transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_features_")
        values = validate_features(X)
        if values.empty:
            return pd.DataFrame(index=X.index, columns=self.output_features_, dtype=float)
        numerical = values[list(NUMERICAL)].replace([np.inf, -np.inf], np.nan).fillna(self.medians_)
        transformed = self._nonlinear(numerical)
        if self.scaler_ is not None:
            transformed[list(SCALED)] = self.scaler_.transform(transformed[list(SCALED)])
        indicators = (values[DIRECTION].to_numpy()[:, None] == np.arange(9)).astype(float)
        result = pd.DataFrame(
            np.column_stack([transformed.to_numpy(), indicators]),
            index=X.index, columns=self.output_features_,
        )
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Nonfinite values after behavioral preprocessing")
        return result

    def transform_sequences(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3 or sequences.shape[1] < 1 or sequences.shape[2] != len(SOURCE_FEATURES):
            raise ValueError("Behavioral sequences must have shape (N,T,25) with T >= 1")
        frame = pd.DataFrame(sequences.reshape(-1, len(SOURCE_FEATURES)), columns=SOURCE_FEATURES)
        return self.transform(frame).to_numpy().reshape(len(sequences), sequences.shape[1], len(OUTPUT_FEATURES))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_features_")
        return np.array(self.output_features_, dtype=object)


def gamma_center(original_transformed: pd.DataFrame) -> float:
    values = original_transformed.to_numpy(dtype=float)
    variance = float(values.var(ddof=0))
    if not np.isfinite(values).all() or not np.isfinite(variance) or variance <= 0:
        raise ValueError("Gamma requires finite original features with positive global variance")
    return 1 / (values.shape[1] * variance)

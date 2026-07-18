"""Post-extraction feature transformation for modeling.

Cleans the extracted feature DataFrame to be model-ready: removes
non-feature columns, replaces infinite values, and fills NaNs with
medians.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are labels or metadata — never used as model input.
EXCLUDE_COLUMNS = {"is_cheating", "session_name", "chunk_start_time"}


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a feature DataFrame for model training.

    Strips label/metadata columns, replaces ±inf with NaN, then
    fills all NaNs with column medians.

    Args:
        df: DataFrame as produced by ``extract_features_from_session``
            (includes ``is_cheating`` and ``session_name`` columns).

    Returns:
        A tuple ``(X, feature_cols)`` where:
            - **X** is a clean numeric DataFrame ready for an estimator.
            - **feature_cols** is the list of column names used.
    """
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    return X, feature_cols
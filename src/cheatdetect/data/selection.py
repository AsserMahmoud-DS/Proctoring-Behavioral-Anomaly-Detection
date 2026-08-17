"""Feature selection via zero-variance detection and correlation analysis.

Identifies redundant features using two criteria:
    1. Zero variance — constant features that carry no signal.
    2. Tier 1 correlation — pairs of the same underlying signal measured
       with different aggregations (e.g. ``velocity_mean`` vs ``velocity_max``).
"""

import warnings

import pandas as pd

_AGG_SUFFIXES = ("_mean", "_std", "_max", "_min", "_sum")


def detect_zero_variance(X: pd.DataFrame) -> list[str]:
    """Return column names whose variance is zero."""
    return X.columns[X.var() == 0].tolist()


def find_correlated_pairs(
    X: pd.DataFrame, threshold: float = 0.85
) -> list[tuple[str, str, float]]:
    """Find feature pairs with absolute correlation above *threshold*.

    Returns:
        List of ``(feat1, feat2, correlation)`` tuples.
    """
    corr = X.corr()
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pairs.append((cols[i], cols[j], float(r)))
    return pairs


def classify_correlation_pair(feat1: str, feat2: str) -> str:
    """Classify a correlated pair by whether they measure the same signal.

    Strips common aggregation suffixes (``_mean``, ``_std``, ``_max``,
    ``_min``, ``_sum``) from both names.  If the remaining base names are
    identical the pair is **Tier 1** — same signal, redundant.  Otherwise
    it is **Tier 2** — derived but distinct (e.g. velocity vs acceleration).

    Returns:
        ``"Tier 1 (drop)"`` or ``"Tier 2 (keep)"``.
    """

    def _base(name: str) -> str:
        for suffix in _AGG_SUFFIXES:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    return "Tier 1 (drop)" if _base(feat1) == _base(feat2) else "Tier 2 (keep)"


def _pick_tier1_drop(feat1: str, feat2: str) -> str:
    """Decide which of two Tier-1 features to drop.

    Priority: keep ``_mean``, drop ``_max``/``_min`` before ``_std``.
    """
    if feat1.endswith(("_max", "_min")):
        return feat1
    if feat2.endswith(("_max", "_min")):
        return feat2
    if feat1.endswith("_std") and not feat2.endswith("_mean"):
        return feat1
    if feat2.endswith("_std") and not feat1.endswith("_mean"):
        return feat2
    return feat2


def _tier1_drops(X_no_zero: pd.DataFrame, threshold: float) -> set[str]:
    """Return features dropped due to Tier-1 correlation redundancy."""
    pairs = find_correlated_pairs(X_no_zero, threshold=threshold)
    drops: set[str] = set()
    for feat1, feat2, _ in pairs:
        if classify_correlation_pair(feat1, feat2) == "Tier 1 (drop)":
            drops.add(_pick_tier1_drop(feat1, feat2))
    return drops


def _drop_zero_var(X: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Return ``(X without zero-variance columns, zero-variance column set)``."""
    zero_var_drops = set(detect_zero_variance(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X_no_zero = X.drop(columns=zero_var_drops, errors="ignore")
    return X_no_zero, zero_var_drops


def select_features(
    X: pd.DataFrame, corr_threshold: float = 0.85
) -> tuple[list[str], list[str]]:
    """Select a feature subset via zero-variance and Tier-1 correlation filtering.

    Runs only on the training data — the caller is responsible for applying
    the returned keep/drop lists to validation and test sets.

    Args:
        X: Training feature matrix (rows = chunks, columns = features).
        corr_threshold: Minimum absolute Pearson correlation to classify
            a pair as highly correlated.  Defaults to 0.85.

    Returns:
        ``(features_to_keep, features_to_drop)`` — both sorted lists.
    """
    X_no_zero, zero_var_drops = _drop_zero_var(X)
    tier1_drops = _tier1_drops(X_no_zero, corr_threshold)

    all_drops = zero_var_drops | tier1_drops
    features_to_keep = [f for f in X.columns if f not in all_drops]
    features_to_drop = sorted(all_drops)

    return features_to_keep, features_to_drop


def feature_summary(X: pd.DataFrame, corr_threshold: float = 0.85) -> pd.DataFrame:
    """Summarize the keep/drop decision and its reason for every feature.

    Args:
        X: Training feature matrix.
        corr_threshold: Minimum absolute Pearson correlation for the
            redundancy check.

    Returns:
        DataFrame indexed by feature name with columns ``variance``,
        ``zero_variance``, ``tier1_redundant``, and ``decision``
        (``"KEEP"`` or ``"DROP"``).
    """
    X_no_zero, zero_var_drops = _drop_zero_var(X)
    tier1_drops = _tier1_drops(X_no_zero, corr_threshold)
    all_drops = zero_var_drops | tier1_drops

    features = list(X.columns)
    return pd.DataFrame(
        {
            "feature": features,
            "variance": X.var().values,
            "zero_variance": [f in zero_var_drops for f in features],
            "tier1_redundant": [f in tier1_drops for f in features],
            "decision": ["DROP" if f in all_drops else "KEEP" for f in features],
        }
    )

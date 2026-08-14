"""Evaluation metrics for anomaly detection models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(
    name: str, scores: np.ndarray, threshold: float, y_true: np.ndarray
) -> dict:
    """Compute classification metrics for one model at a fixed threshold.

    Args:
        name: Model name (e.g. ``"IF"``, ``"OCSVM"``, ``"Ensemble"``).
        scores: Anomaly scores (higher = more anomalous).
        threshold: Decision boundary; ``scores >= threshold`` is predicted anomalous.
        y_true: Binary ground-truth labels (1 = anomalous).

    Returns:
        Dict with scalar metrics (``pr_auc``, ``roc_auc``, ``precision``,
        ``recall``, ``f1``, ``threshold``), confusion-matrix counts
        (``tp``, ``tn``, ``fp``, ``fn``), and the raw ``cm`` and ``scores``
        arrays for downstream plotting.
    """
    y_pred = (scores >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = auc(*roc_curve(y_true, scores)[:2])

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": threshold,
        "tp": int(cm[1, 1]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "cm": cm,
        "scores": scores,
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Aggregate ``evaluate_model`` dicts into a summary DataFrame.

    Drops the non-tabular fields (``cm`` and ``scores``), keeping only
    scalar metrics, indexed by model name.
    """
    df = pd.DataFrame(results).set_index("model")
    return df.drop(columns=["cm", "scores"])

"""Threshold tuning for turning anomaly scores into binary predictions."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def tune_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    precision_floor: float = 0.5,
) -> dict:
    """Find the threshold that maximizes recall subject to a precision floor.

    Sweeps candidate thresholds from the unique score values. If no threshold
    meets ``precision >= precision_floor``, falls back to the threshold that
    maximizes F1.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        y_true: Binary ground-truth labels (1 = anomalous).
        precision_floor: Minimum acceptable precision.

    Returns:
        Dict with keys ``threshold``, ``precision``, ``recall``, ``f1``,
        and ``fallback_used``.
    """
    unique_scores = np.sort(np.unique(scores))

    result = {
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fallback_used": False,
    }

    # Pass 1: maximize recall while precision >= floor
    best_recall = -1.0
    for thr in unique_scores:
        y_pred = (scores >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if prec >= precision_floor and rec > best_recall:
            best_recall = rec
            result["threshold"] = thr
            result["precision"] = prec
            result["recall"] = rec

    # Pass 2: fallback to max F1 if the floor was never met
    if result["threshold"] is None:
        result["fallback_used"] = True
        best_f1 = -1.0
        for thr in unique_scores:
            y_pred = (scores >= thr).astype(int)
            if y_pred.sum() == 0:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                result["threshold"] = thr
                result["recall"] = recall_score(y_true, y_pred, zero_division=0)
                result["precision"] = precision_score(y_true, y_pred, zero_division=0)

    result["f1"] = f1_score(y_true, (scores >= result["threshold"]).astype(int), zero_division=0)
    return result

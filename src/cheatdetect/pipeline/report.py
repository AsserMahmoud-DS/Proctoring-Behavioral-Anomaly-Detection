"""Test-set reporting.

Called only after validation selection is frozen: these helpers score held-out
test windows and stack the per-model metrics. No model choice happens here.
"""

import numpy as np
import pandas as pd

from cheatdetect.eval import compare_models, evaluate_model


def evaluate_scores(
    name: str, scores: np.ndarray, threshold: float, y_test: np.ndarray
) -> dict:
    """Evaluate one score vector against held-out test labels."""
    return evaluate_model(name, scores, threshold, y_test)


def evaluate_detectors(
    detectors: dict,
    thresholds: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> list[dict]:
    """Evaluate each fitted detector on the held-out test set."""
    return [
        evaluate_scores(name, detector.decision_function(X_test), thresholds[name], y_test)
        for name, detector in detectors.items()
    ]


def summarize(results: list[dict]) -> pd.DataFrame:
    """Stack per-model test results into the comparison table."""
    return compare_models(results)

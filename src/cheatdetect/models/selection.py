"""Ordered candidate sweep shared by production grid searches and the study runner.

Selection is validation ROC-AUC with a deterministic first-candidate tie-break:
candidates are evaluated in declaration order, results are sorted by descending
ROC-AUC with a stable sort, and the winner is the fitted detector from the first
row. PR-AUC is reported alongside. Failed fits are recorded (never silently
replaced); if every candidate fails, the caller receives ``None`` and can
report the failure.

This module lives in the models layer so detector grid searches and the
experiment runner can share it without importing each other.
"""

import logging
import time
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .base import validate_selection_labels

logger = logging.getLogger(__name__)


def sweep_candidates(
    candidates: Iterable[dict],
    fit_and_score: Callable[[dict], tuple[object, np.ndarray, dict]],
    y_val: np.ndarray,
) -> tuple[object | None, pd.DataFrame]:
    """Evaluate candidates in declaration order and keep the fitted winner.

    Args:
        candidates: Candidate parameter dicts, in tie-break order.
        fit_and_score: ``candidate -> (fitted_detector, val_scores, diagnostics)``.
            ``diagnostics`` rows are merged into the results table (e.g.
            convergence status or training epochs).
        y_val: Binary validation labels (1 = anomalous).

    Returns:
        ``(best_detector, results_df)`` where ``results_df`` is sorted by
        descending ROC-AUC (stable) and carries ``roc_auc``, ``pr_auc``,
        ``pr_auc_baseline``, ``seconds``, ``status``, and ``error`` columns.
        ``best_detector`` is ``None`` when no candidate could be fitted.
    """
    validate_selection_labels(y_val)

    rows: list[dict] = []
    baseline = float(np.mean(y_val))
    best_detector: object | None = None
    best_score = -np.inf
    for candidate in candidates:
        row = {**candidate, "pr_auc_baseline": baseline}
        start = time.perf_counter()
        try:
            fitted, scores, diagnostics = fit_and_score(candidate)
        except Exception as exc:  # keep searching; the failure must stay visible
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("Candidate %s failed: %s", candidate, message)
            row.update(
                {
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "seconds": time.perf_counter() - start,
                    "status": "failed",
                    "error": message,
                }
            )
            rows.append(row)
            continue

        roc_auc = roc_auc_score(y_val, scores)
        row.update(diagnostics or {})
        row.update(
            {
                "roc_auc": roc_auc,
                "pr_auc": average_precision_score(y_val, scores),
                "seconds": time.perf_counter() - start,
                "status": "ok",
                "error": "",
            }
        )
        rows.append(row)
        if roc_auc > best_score:
            best_score = roc_auc
            best_detector = fitted

    results = pd.DataFrame(rows).sort_values(
        "roc_auc", ascending=False, kind="stable", na_position="last"
    )
    return best_detector, results.reset_index(drop=True)

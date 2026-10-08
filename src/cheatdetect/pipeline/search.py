"""Validation-only model search.

Selection never reads test partitions: the flat candidates compete on the
combined validation set by ROC-AUC, and the research-only LSTM grid is scored
on validation only. Test evaluation lives in ``pipeline.report``.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cheatdetect.config import ExperimentConfig
from cheatdetect.models import IsolationForestDetector, OCSVMDetector, tune_threshold
from cheatdetect.models.base import validate_selection_labels
from cheatdetect.models.ensemble import grid_search as ensemble_grid_search
from cheatdetect.models.selection import sweep_candidates

__all__ = ["search_validation", "search_lstm", "resolve_lstm_es", "sweep_candidates"]

logger = logging.getLogger(__name__)

SELECTION_METRIC = "roc_auc"


def search_validation(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_val_normal: pd.DataFrame,
    y_val: np.ndarray,
    config: ExperimentConfig,
) -> dict:
    """Search flat candidates on validation only and pick the production winner.

    Args:
        X_train, X_val: raw 25-feature training / combined validation matrices.
        X_val_normal: normal-only validation rows used to calibrate the
            ensemble's fixed score normalization.
        y_val: binary validation labels (normal + anomalous).
        config: frozen hyperparameter configuration.

    Returns:
        Dict with the fitted ``detectors``, their validation ``val_scores`` and
        ``thresholds``, per-model grid results, and the ``best_*`` winner by
        validation ROC-AUC (first-candidate ties).
    """
    validate_selection_labels(y_val)

    best_if, if_results = IsolationForestDetector.grid_search(
        X_train,
        X_val,
        y_val,
        {
            "n_estimators": list(config.if_n_estimators),
            "max_samples": list(config.if_max_samples),
            "contamination": list(config.if_contamination),
        },
        random_state=config.random_state,
    )
    if best_if is None:
        raise ValueError("No Isolation Forest candidate could be fitted")

    best_ocsvm, ocsvm_results = OCSVMDetector.grid_search(
        X_train,
        X_val,
        y_val,
        {
            "nu": list(config.ocsvm_nu),
            "gamma": list(config.ocsvm_gamma),
            "kernel": list(config.ocsvm_kernel),
        },
        random_state=config.random_state,
    )
    if best_ocsvm is None:
        raise ValueError("No One-Class SVM candidate could be fitted")

    best_ensemble, ensemble_results = ensemble_grid_search(
        best_if,
        best_ocsvm,
        X_val,
        y_val,
        config.ensemble_weights,
        X_ref=X_val_normal,
    )

    detectors = {"IF": best_if, "OCSVM": best_ocsvm, "Ensemble": best_ensemble}
    val_scores = {name: det.decision_function(X_val) for name, det in detectors.items()}
    thresholds = {
        name: tune_threshold(
            val_scores[name], y_val, precision_floor=config.precision_floor
        )["threshold"]
        for name in detectors
    }
    selection_scores = {
        name: roc_auc_score(y_val, scores) for name, scores in val_scores.items()
    }
    best_name = max(selection_scores, key=selection_scores.get)

    logger.info("Validation selection complete. Best model: %s", best_name)
    return {
        "detectors": detectors,
        "val_scores": val_scores,
        "thresholds": thresholds,
        "selection_metric": SELECTION_METRIC,
        "selection_scores": selection_scores,
        "best_name": best_name,
        "best_detector": detectors[best_name],
        "best_threshold": thresholds[best_name],
        "if_grid_results": if_results,
        "ocsvm_grid_results": ocsvm_results,
        "ensemble_grid_results": ensemble_results,
    }


def resolve_lstm_es(config: ExperimentConfig, lstm_data: dict) -> np.ndarray | None:
    """Select the sequence set used for early stopping.

    ``normal_val`` (default) uses held-out pure-normal sequences, which match
    the training distribution. ``mixed_val`` reproduces the old behavior
    (normal + anomalies). ``none`` disables early stopping.
    """
    source = config.lstm_es_source
    if source == "none":
        return None
    if source == "normal_val":
        if lstm_data["y_es"].any():
            raise ValueError("LSTM early-stopping set must contain only normals")
        return lstm_data["X_es"]
    if source == "mixed_val":
        return lstm_data["X_val"]
    raise ValueError(
        f"Unknown lstm_es_source '{source}'; expected normal_val, mixed_val, none"
    )


def search_lstm(config: ExperimentConfig, lstm_data: dict) -> dict:
    """Train the research-only LSTM AE and score it on validation only.

    The detector is imported lazily so importing the pipeline does not require
    torch (a dev-only dependency). The result is never eligible to be the
    serialized production model.

    Returns:
        Dict with the fitted ``detector``, its validation ``val_scores``,
        validation ``threshold``, and ``grid_results``.
    """
    from cheatdetect.models.lstm_ae import LSTMAutoencoderDetector

    X_train = lstm_data["X_train"]
    X_val = lstm_data["X_val"]
    y_val = lstm_data["y_val"]
    X_es = resolve_lstm_es(config, lstm_data)

    best_detector, grid_results = LSTMAutoencoderDetector.grid_search(
        X_train,
        X_val,
        y_val,
        X_es,
        lstm_data["feature_names"],
        {
            "hidden_dim": list(config.lstm_hidden_dims),
            "num_layers": list(config.lstm_num_layers),
            "dropout": list(config.lstm_dropouts),
            "lr": list(config.lstm_lrs),
            "batch_size": list(config.lstm_batch_sizes),
        },
        fixed_kwargs={
            "input_dim": X_train.shape[2],
            "seq_len": lstm_data["seq_len"],
            "epochs": config.lstm_epochs,
            "patience": config.lstm_patience,
            "recipe": config.lstm_recipe,
        },
        random_state=config.random_state,
    )
    if best_detector is None:
        raise ValueError("No LSTM autoencoder candidate could be fitted")

    val_scores = best_detector.decision_function(X_val)
    threshold = tune_threshold(
        val_scores, y_val, precision_floor=config.precision_floor
    )["threshold"]
    return {
        "detector": best_detector,
        "val_scores": val_scores,
        "threshold": threshold,
        "grid_results": grid_results,
    }

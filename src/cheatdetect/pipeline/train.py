"""End-to-end training orchestration.

``train_pipeline`` runs the full experiment: session split, paired data
construction, model grid search, threshold tuning, and final evaluation — then
serializes the best detector plus an inference-ready config. The
manifest-guarded builder in ``cheatdetect.data.dataset`` is the single source of
truth; no pickle/npz caches are involved.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cheatdetect.config import (
    EDA_DIR,
    MODELS_DIR,
    MIXED_DIR,
    NORMAL_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLIT_INFO_PATH,
    TEST_DIR,
    VAL_DIR,
    ExperimentConfig,
)
from cheatdetect.data import prepare_study
from cheatdetect.data.feature_schema import SOURCE_FEATURES
from cheatdetect.eval import compare_models, evaluate_model
from cheatdetect.models import IsolationForestDetector, OCSVMDetector, tune_threshold
from cheatdetect.models.base import validate_selection_labels
from cheatdetect.models.ensemble import grid_search as ensemble_grid_search

logger = logging.getLogger(__name__)


def _ensure_directories() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, EDA_DIR, VAL_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _split_sessions(config: ExperimentConfig) -> dict:
    """Split sessions into train/val/test and persist split info."""
    normal_files = sorted(NORMAL_DIR.glob("*.csv"))
    mixed_files = sorted(MIXED_DIR.glob("*.csv"))

    normal_train, normal_val = train_test_split(
        normal_files,
        test_size=config.normal_val_size,
        random_state=config.random_state,
    )
    mixed_val, mixed_test = train_test_split(
        mixed_files,
        test_size=1 - config.mixed_val_size,
        random_state=config.random_state,
    )

    split_info = {
        "normal_train": [f.name for f in normal_train],
        "normal_val": [f.name for f in normal_val],
        "mixed_val": [f.name for f in mixed_val],
        "mixed_test": [f.name for f in mixed_test],
        "random_state": config.random_state,
        "normal_val_size": config.normal_val_size,
        "mixed_val_size": config.mixed_val_size,
    }
    SPLIT_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_INFO_PATH, "w") as f:
        json.dump(split_info, f, indent=2)

    return {
        "normal_train": [f.name for f in normal_train],
        "normal_val": [f.name for f in normal_val],
        "mixed_val": [f.name for f in mixed_val],
        "mixed_test": [f.name for f in mixed_test],
    }


def prepare_data(config: ExperimentConfig) -> dict:
    """Split sessions and build the fixed-schema paired representations.

    Single production data path: delegates to
    :func:`cheatdetect.data.prepare_study`, which extracts the flat and
    sequence views from the same realized parent windows, so the flat
    matrices and the LSTM sequences are aligned by construction. Recipes are
    not fitted here because detectors own preprocessing; the study runner
    requests them explicitly.

    Args:
        config: Frozen hyperparameter configuration for the run.

    Returns:
        Dict with the ``study``, the session ``split``, the raw 25-feature
        matrices (``X_train``, ``X_val_normal``, ``X_val``, ``X_test``), their
        labels, and ``features_to_keep``.
    """
    _ensure_directories()
    split = _split_sessions(config)
    study = prepare_study(
        split=split,
        normal_dir=NORMAL_DIR,
        mixed_dir=MIXED_DIR,
        n_copies=config.aug_n_copies,
        sigma_range=(config.aug_sigma_min, config.aug_sigma_max),
        chunk_size=config.chunk_size,
        step_size=config.step_size,
        sub_chunk=config.lstm_sub_chunk,
        sub_step=config.lstm_sub_step,
        label_fraction=config.cheating_threshold,
        recipes=(),
        random_state=config.random_state,
    )
    raw = study.raw
    X_train = study.training_raw(config.aug_enabled)
    X_val_normal = raw["normal_val"].parent
    X_val_mixed = raw["mixed_val"].parent
    X_test = raw["mixed_test"].parent
    y_val = np.concatenate([raw["normal_val"].labels, raw["mixed_val"].labels])
    y_test = raw["mixed_test"].labels
    X_val = pd.concat([X_val_normal, X_val_mixed], ignore_index=True)

    return {
        "study": study,
        "split": split,
        "df_train_normal": X_train,
        "df_val_normal": X_val_normal,
        "df_val_mixed": X_val_mixed,
        "df_test_mixed": X_test,
        "X_train": X_train,
        "X_val_normal": X_val_normal,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "features_to_keep": list(SOURCE_FEATURES),
    }


def prepare_lstm_data(config: ExperimentConfig, data: dict) -> dict:
    """Return raw micro-chunk sequences aligned 1:1 with the flat matrices.

    Built from the same :class:`~cheatdetect.data.PreparedStudy` as
    :func:`prepare_data`: training sequences follow the augmentation setting,
    validation combines normal then mixed in the flat order, and ``X_es`` is
    the held-out pure-normal set used for early stopping.

    Args:
        config: Frozen experiment configuration.
        data: The dict returned by :func:`prepare_data` (for its ``study``).

    Returns:
        Dict with aligned ``X_train``, ``X_val``, ``X_es``, ``X_test`` (3D
        arrays), their labels, the source feature names, and the sequence length.
    """
    study = data["study"]
    raw = study.raw
    X_train = study.training_sequences(config.aug_enabled)
    y_train = study.training_labels(config.aug_enabled)
    X_val = np.concatenate(
        [raw["normal_val"].sequences, raw["mixed_val"].sequences], axis=0
    )
    y_val = np.concatenate([raw["normal_val"].labels, raw["mixed_val"].labels])
    X_es = raw["normal_val"].sequences
    y_es = raw["normal_val"].labels
    X_test = raw["mixed_test"].sequences
    y_test = raw["mixed_test"].labels

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_es": X_es,
        "y_es": y_es,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(SOURCE_FEATURES),
        "seq_len": X_train.shape[1],
    }


def _serialize_model(
    best_detector,
    best_name: str,
    best_threshold: float,
    features_to_keep: list[str],
    config: ExperimentConfig,
    pr_auc_test: float,
    *,
    selection_metric: str,
    selection_score_val: float,
) -> None:
    """Persist the best detector and the inference-ready config."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_detector, MODELS_DIR / "best_model.joblib")

    model_config = {
        "model": best_name,
        "features_to_keep": features_to_keep,
        "threshold": float(best_threshold),
        "chunk_size": config.chunk_size,
        "step_size": config.step_size,
        "cheating_threshold": config.cheating_threshold,
        "pr_auc_test": pr_auc_test,
        "selection_metric": selection_metric,
        "selection_score_val": float(selection_score_val),
        "random_state": config.random_state,
    }
    with open(MODELS_DIR / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)


def _resolve_lstm_es(config: ExperimentConfig, lstm_data: dict) -> np.ndarray | None:
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


def _run_lstm(
    config: ExperimentConfig,
    data: dict,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Train the research-only LSTM AE and return its val/test scores.

    The detector is imported lazily so importing the pipeline does not
    require torch (a dev-only dependency).
    """
    from cheatdetect.models.lstm_ae import LSTMAutoencoderDetector

    lstm_data = prepare_lstm_data(config, data)
    X_train = lstm_data["X_train"]
    X_val = lstm_data["X_val"]
    y_val = lstm_data["y_val"]
    X_test = lstm_data["X_test"]
    y_test = lstm_data["y_test"]
    X_es = _resolve_lstm_es(config, lstm_data)

    # The LSTM samples must map 1:1 onto the flat chunks for a fair comparison.
    if len(y_val) != len(data["y_val"]) or not np.array_equal(y_val, data["y_val"]):
        raise ValueError(
            "LSTM validation labels are not aligned with the flat pipeline; "
            "the sequence extraction order/count drifted."
        )
    if len(y_test) != len(data["y_test"]) or not np.array_equal(
        y_test, data["y_test"]
    ):
        raise ValueError(
            "LSTM test labels are not aligned with the flat pipeline; "
            "the sequence extraction order/count drifted."
        )

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

    return (
        best_detector.decision_function(X_val),
        best_detector.decision_function(X_test),
        grid_results,
    )


def train_pipeline(config: ExperimentConfig) -> dict:
    """Run the full experiment and return the results dict.

    Args:
        config: Frozen hyperparameter configuration for the run.

    Returns:
        Dict with the best detector, its name/threshold, per-model test
        results, a summary metrics DataFrame, the feature lists, and the
        validation scores/labels for plotting.
    """
    data = prepare_data(config)
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    validate_selection_labels(y_val)
    X_test = data["X_test"]
    y_test = data["y_test"]
    features_to_keep = data["features_to_keep"]

    # ---- Modeling ---------------------------------------------------------
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

    best_ensemble, ensemble_results = ensemble_grid_search(
        best_if,
        best_ocsvm,
        X_val,
        y_val,
        config.ensemble_weights,
        X_ref=data["X_val_normal"],
    )

    detectors = {"IF": best_if, "OCSVM": best_ocsvm, "Ensemble": best_ensemble}

    val_scores = {name: det.decision_function(X_val) for name, det in detectors.items()}
    thresholds = {
        name: tune_threshold(
            val_scores[name], y_val, precision_floor=config.precision_floor
        )["threshold"]
        for name in detectors
    }

    selection_metric = "roc_auc"
    selection_scores = {
        name: roc_auc_score(y_val, scores)
        for name, scores in val_scores.items()
    }
    best_name = max(selection_scores, key=selection_scores.get)
    best_detector = detectors[best_name]
    best_threshold = thresholds[best_name]

    # ---- Evaluation -------------------------------------------------------
    flat_test_results = [
        evaluate_model(name, det.decision_function(X_test), thresholds[name], y_test)
        for name, det in detectors.items()
    ]

    _serialize_model(
        best_detector,
        best_name,
        best_threshold,
        features_to_keep,
        config,
        next(result["pr_auc"] for result in flat_test_results if result["model"] == best_name),
        selection_metric=selection_metric,
        selection_score_val=selection_scores[best_name],
    )

    # Research-only LSTM comparison — surfaced in the table/plots but never
    # eligible to be the serialized production model.
    test_results = list(flat_test_results)
    lstm_grid_results = None
    if config.lstm_enabled:
        lstm_val_scores, lstm_test_scores, lstm_grid_results = _run_lstm(
            config, data
        )
        lstm_threshold = tune_threshold(
            lstm_val_scores, y_val, precision_floor=config.precision_floor
        )["threshold"]
        val_scores["LSTM-AE"] = lstm_val_scores
        test_results.append(
            evaluate_model("LSTM-AE", lstm_test_scores, lstm_threshold, y_test)
        )

    metrics_df = compare_models(test_results)

    logger.info("Training complete. Best model: %s", best_name)
    return {
        "best_detector": best_detector,
        "best_name": best_name,
        "best_threshold": best_threshold,
        "selection_metric": selection_metric,
        "selection_score_val": selection_scores[best_name],
        "test_results": test_results,
        "metrics_df": metrics_df,
        "features_to_keep": features_to_keep,
        "val_scores": val_scores,
        "y_val": y_val,
        "y_test": y_test,
        "if_grid_results": if_results,
        "ocsvm_grid_results": ocsvm_results,
        "ensemble_grid_results": ensemble_results,
        "lstm_grid_results": lstm_grid_results,
    }

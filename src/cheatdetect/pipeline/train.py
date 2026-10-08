"""End-to-end training orchestration.

``train_pipeline`` runs the full experiment: session split, paired data
construction, validation-only model search, held-out test reporting, and
serialization of the best detector plus an inference-ready config. The
manifest-guarded builder in ``cheatdetect.data.dataset`` is the single source of
truth; no pickle/npz caches are involved.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
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

from .report import evaluate_detectors, evaluate_scores, summarize
from .search import search_lstm, search_validation

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


def _assert_lstm_alignment(lstm_data: dict, data: dict) -> None:
    """The LSTM samples must map 1:1 onto the flat chunks for a fair comparison."""
    for key in ("val", "test"):
        lstm_labels = lstm_data[f"y_{key}"]
        flat_labels = data[f"y_{key}"]
        if len(lstm_labels) != len(flat_labels) or not np.array_equal(
            lstm_labels, flat_labels
        ):
            raise ValueError(
                f"LSTM {key} labels are not aligned with the flat pipeline; "
                "the sequence extraction order/count drifted."
            )


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
    y_val = data["y_val"]
    y_test = data["y_test"]
    features_to_keep = data["features_to_keep"]

    # ---- Validation-only search (never reads test) ------------------------
    search = search_validation(
        data["X_train"],
        data["X_val"],
        data["X_val_normal"],
        y_val,
        config,
    )
    best_name = search["best_name"]

    # ---- Frozen test reporting --------------------------------------------
    flat_test_results = evaluate_detectors(
        search["detectors"], search["thresholds"], data["X_test"], y_test
    )
    _serialize_model(
        search["best_detector"],
        best_name,
        search["best_threshold"],
        features_to_keep,
        config,
        next(result["pr_auc"] for result in flat_test_results if result["model"] == best_name),
        selection_metric=search["selection_metric"],
        selection_score_val=search["selection_scores"][best_name],
    )

    # Research-only LSTM comparison — surfaced in the table/plots but never
    # eligible to be the serialized production model.
    test_results = list(flat_test_results)
    val_scores = dict(search["val_scores"])
    lstm_grid_results = None
    if config.lstm_enabled:
        lstm_data = prepare_lstm_data(config, data)
        _assert_lstm_alignment(lstm_data, data)
        lstm = search_lstm(config, lstm_data)
        val_scores["LSTM-AE"] = lstm["val_scores"]
        test_results.append(
            evaluate_scores(
                "LSTM-AE",
                lstm["detector"].decision_function(lstm_data["X_test"]),
                lstm["threshold"],
                y_test,
            )
        )
        lstm_grid_results = lstm["grid_results"]

    metrics_df = summarize(test_results)

    logger.info("Training complete. Best model: %s", best_name)
    return {
        "best_detector": search["best_detector"],
        "best_name": best_name,
        "best_threshold": search["best_threshold"],
        "selection_metric": search["selection_metric"],
        "selection_score_val": search["selection_scores"][best_name],
        "test_results": test_results,
        "metrics_df": metrics_df,
        "features_to_keep": features_to_keep,
        "val_scores": val_scores,
        "y_val": y_val,
        "y_test": y_test,
        "if_grid_results": search["if_grid_results"],
        "ocsvm_grid_results": search["ocsvm_grid_results"],
        "ensemble_grid_results": search["ensemble_grid_results"],
        "lstm_grid_results": lstm_grid_results,
    }

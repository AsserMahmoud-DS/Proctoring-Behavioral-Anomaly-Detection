"""End-to-end training orchestration.

``train_pipeline`` runs the full experiment: session split, feature
extraction (with caching), augmentation, feature selection, model grid
search, threshold tuning, and final evaluation — then serializes the best
detector plus an inference-ready config.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from cheatdetect.config import (
    EDA_DIR,
    FEATURE_LISTS_PATH,
    LSTM_ES_PATH,
    LSTM_TEST_PATH,
    LSTM_TRAIN_PATH,
    LSTM_VAL_PATH,
    MODELS_DIR,
    MIXED_DIR,
    NORMAL_DIR,
    PROCESSED_DIR,
    SPLIT_INFO_PATH,
    TEST_DIR,
    TEST_MIXED_PATH,
    TRAIN_AUGMENTED_PATH,
    TRAIN_NORMAL_PATH,
    VAL_DIR,
    VAL_MIXED_PATH,
    VAL_NORMAL_PATH,
    REPORTS_DIR,
    ExperimentConfig,
)
from cheatdetect.data import (
    align_sequence_features,
    augment_sequences,
    augment_session_data,
    clean_features,
    clean_session_data,
    extract_sequences_from_sessions,
    find_skewed_features,
    load_sessions,
    load_single_session,
    merge_window_switch_events,
    process_sessions,
    select_features,
    sequence_length,
)
from cheatdetect.eval import compare_models, evaluate_model
from cheatdetect.models import IsolationForestDetector, OCSVMDetector, tune_threshold
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
        "normal_train": list(normal_train),
        "normal_val": list(normal_val),
        "mixed_val": list(mixed_val),
        "mixed_test": list(mixed_test),
    }


def _load_or_process(
    files: list[Path], cache_path: Path, config: ExperimentConfig
) -> pd.DataFrame:
    """Return cached features if present, otherwise extract and cache them."""
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        return pd.read_pickle(cache_path)

    logger.info("Processing %d sessions -> %s", len(files), cache_path.name)
    df = process_sessions(
        [str(f) for f in files],
        chunk_size=config.chunk_size,
        step_size=config.step_size,
        cheating_threshold=config.cheating_threshold,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return df


def _augment_train_normal(
    df_train_normal: pd.DataFrame,
    normal_train_files: list[Path],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Augment the train-normal data and concatenate it with the original."""
    if TRAIN_AUGMENTED_PATH.exists():
        logger.info("Loading cached augmented features from %s", TRAIN_AUGMENTED_PATH)
        df_aug = pd.read_pickle(TRAIN_AUGMENTED_PATH)
    else:
        sessions_raw = load_sessions(str(NORMAL_DIR))
        train_basenames = {f.name for f in normal_train_files}
        sessions_raw = {
            k: v for k, v in sessions_raw.items() if k in train_basenames
        }
        sessions_clean = {
            Path(name).stem: clean_session_data(df)
            for name, df in sessions_raw.items()
        }
        df_aug = augment_session_data(
            sessions_clean,
            chunk_size=config.chunk_size,
            step_size=config.step_size,
            n_copies=config.aug_n_copies,
            sigma_range=(config.aug_sigma_min, config.aug_sigma_max),
            random_state=config.random_state,
        )
        df_aug.to_pickle(TRAIN_AUGMENTED_PATH)

    return pd.concat([df_train_normal, df_aug], ignore_index=True)


def _build_matrices(
    df_train_normal: pd.DataFrame,
    df_val_normal: pd.DataFrame,
    df_val_mixed: pd.DataFrame,
    df_test_mixed: pd.DataFrame,
    config: ExperimentConfig,
) -> dict:
    """Select and clean already-merged features into model-ready matrices.

    Expects the DataFrames to already have been passed through
    :func:`merge_window_switch_events` (done in :func:`prepare_data`).
    """
    # Feature selection on training data only
    X_train_raw, _ = clean_features(df_train_normal)
    features_to_keep, features_to_drop = select_features(
        X_train_raw, corr_threshold=config.high_corr_threshold
    )

    X_train, _ = clean_features(df_train_normal)
    X_train = X_train[features_to_keep]

    X_val_normal, _ = clean_features(df_val_normal)
    X_val_normal = X_val_normal[features_to_keep]

    X_val_mixed, _ = clean_features(df_val_mixed)
    X_val_mixed = X_val_mixed[features_to_keep]
    y_val_mixed = df_val_mixed["is_cheating"].values

    X_test, _ = clean_features(df_test_mixed)
    X_test = X_test[features_to_keep]
    y_test = df_test_mixed["is_cheating"].values

    # Combine validation sets (normal + mixed) for model selection
    X_val = pd.concat([X_val_normal, X_val_mixed], ignore_index=True)
    y_val = np.concatenate([np.zeros(len(X_val_normal)), y_val_mixed])

    # Persist the keep/drop decision for the notebook's EDA
    FEATURE_LISTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_LISTS_PATH, "w") as f:
        json.dump(
            {"features_to_keep": features_to_keep, "features_to_drop": features_to_drop},
            f,
            indent=2,
        )

    return {
        "X_train": X_train,
        "X_val_normal": X_val_normal,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "features_to_keep": features_to_keep,
    }


def _serialize_model(
    best_detector,
    best_name: str,
    best_threshold: float,
    features_to_keep: list[str],
    skewed_features: list[str],
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
        "skewed_features": skewed_features,
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


def prepare_data(config: ExperimentConfig) -> dict:
    """Split, load, augment, merge, and select features into model-ready matrices.

    Shared by :func:`train_pipeline` and the EDA notebook so both observe the
    exact same split and processed data (reproducibility). Intermediate
    DataFrames are returned for visualization.

    Args:
        config: Frozen hyperparameter configuration for the run.

    Returns:
        Dict with the processed DataFrames (post-merge), the session split,
        and the model-ready matrices (``X_train``, ``X_val_normal``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``, ``features_to_keep``).
    """
    _ensure_directories()

    split = _split_sessions(config)

    df_train_normal = _load_or_process(split["normal_train"], TRAIN_NORMAL_PATH, config)
    df_val_normal = _load_or_process(split["normal_val"], VAL_NORMAL_PATH, config)
    df_val_mixed = _load_or_process(split["mixed_val"], VAL_MIXED_PATH, config)
    df_test_mixed = _load_or_process(split["mixed_test"], TEST_MIXED_PATH, config)

    if config.aug_enabled:
        df_train_normal = _augment_train_normal(
            df_train_normal, split["normal_train"], config
        )

    # Merge redundant action columns (post-augmentation, as in the notebook)
    df_train_normal = merge_window_switch_events(df_train_normal)
    df_val_normal = merge_window_switch_events(df_val_normal)
    df_val_mixed = merge_window_switch_events(df_val_mixed)
    df_test_mixed = merge_window_switch_events(df_test_mixed)

    matrices = _build_matrices(
        df_train_normal, df_val_normal, df_val_mixed, df_test_mixed, config
    )

    return {
        "df_train_normal": df_train_normal,
        "df_val_normal": df_val_normal,
        "df_val_mixed": df_val_mixed,
        "df_test_mixed": df_test_mixed,
        "split": split,
        **matrices,
    }


def _load_or_build_sequences(cache_path: Path, builder):
    """Load cached sequences or build + cache them as a compressed ``.npz``."""
    if cache_path.exists():
        logger.info("Loading cached sequences from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], [str(n) for n in data["feature_names"]]

    X, y, feature_names = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, X=X, y=y, feature_names=np.array(feature_names, dtype=object)
    )
    return X, y, feature_names


def prepare_lstm_data(
    config: ExperimentConfig, data: dict, features_to_keep: list[str]
) -> dict:
    """Build (and cache) aligned micro-chunk sequences for the LSTM AE.

    Reuses the exact session split from :func:`prepare_data`. Training
    sequences are the augmented normal set (matching the flat models);
    validation is normal + mixed in the same order as the flat matrices.

    Args:
        config: Frozen experiment configuration.
        data: The dict returned by :func:`prepare_data` (for its ``split``).
        features_to_keep: Curated feature list from the flat pipeline.

    Returns:
        Dict with aligned ``X_train``, ``X_val``, ``X_test`` (3D arrays),
        their labels, the feature names, and the sequence length.
    """
    split = data["split"]
    # Key by full path: normal and mixed directories may contain files with
    # identical basenames, so a basename-keyed dict would collide.
    sessions_clean: dict[Path, pd.DataFrame] = {}
    for directory in (NORMAL_DIR, MIXED_DIR):
        for path in sorted(directory.glob("*.csv")):
            sessions_clean[path] = clean_session_data(load_single_session(path))

    def ordered(files: list[Path]) -> list[pd.DataFrame]:
        return [sessions_clean[f] for f in files if f in sessions_clean]

    def geometry() -> dict:
        return {
            "chunk_size": config.chunk_size,
            "step_size": config.step_size,
            "sub_chunk": config.lstm_sub_chunk,
            "sub_step": config.lstm_sub_step,
        }

    def build_train() -> tuple[np.ndarray, np.ndarray, list[str]]:
        files = ordered(split["normal_train"])
        X_orig, y_orig, names = extract_sequences_from_sessions(
            files, cheating_threshold=config.cheating_threshold, **geometry()
        )
        if not config.aug_enabled:
            return X_orig, y_orig, names
        X_aug, y_aug, _ = augment_sequences(
            files,
            n_copies=config.aug_n_copies,
            sigma_range=(config.aug_sigma_min, config.aug_sigma_max),
            cheating_threshold=config.cheating_threshold,
            random_state=config.random_state,
            **geometry(),
        )
        return (
            np.concatenate([X_orig, X_aug], axis=0),
            np.concatenate([y_orig, y_aug]),
            names,
        )

    def build_val() -> tuple[np.ndarray, np.ndarray, list[str]]:
        X_normal, y_normal, names = extract_sequences_from_sessions(
            ordered(split["normal_val"]),
            cheating_threshold=config.cheating_threshold,
            **geometry(),
        )
        X_mixed, y_mixed, _ = extract_sequences_from_sessions(
            ordered(split["mixed_val"]),
            cheating_threshold=config.cheating_threshold,
            **geometry(),
        )
        return (
            np.concatenate([X_normal, X_mixed], axis=0),
            np.concatenate([y_normal, y_mixed]),
            names,
        )

    def build_es() -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Held-out pure-normal sequences for early stopping."""
        return extract_sequences_from_sessions(
            ordered(split["normal_val"]),
            cheating_threshold=config.cheating_threshold,
            **geometry(),
        )

    def build_test() -> tuple[np.ndarray, np.ndarray, list[str]]:
        return extract_sequences_from_sessions(
            ordered(split["mixed_test"]),
            cheating_threshold=config.cheating_threshold,
            **geometry(),
        )

    X_train, y_train, raw_names = _load_or_build_sequences(
        LSTM_TRAIN_PATH, build_train
    )
    X_val, y_val, _ = _load_or_build_sequences(LSTM_VAL_PATH, build_val)
    X_es, y_es, _ = _load_or_build_sequences(LSTM_ES_PATH, build_es)
    X_test, y_test, _ = _load_or_build_sequences(LSTM_TEST_PATH, build_test)

    return {
        "X_train": align_sequence_features(X_train, raw_names, features_to_keep),
        "y_train": y_train,
        "X_val": align_sequence_features(X_val, raw_names, features_to_keep),
        "y_val": y_val,
        "X_es": align_sequence_features(X_es, raw_names, features_to_keep),
        "y_es": y_es,
        "X_test": align_sequence_features(X_test, raw_names, features_to_keep),
        "y_test": y_test,
        "feature_names": features_to_keep,
        "seq_len": sequence_length(
            config.chunk_size, config.lstm_sub_chunk, config.lstm_sub_step
        ),
    }


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
    features_to_keep: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Train the research-only LSTM AE and return its val/test scores.

    The detector is imported lazily so importing the pipeline does not
    require torch (a dev-only dependency).
    """
    from cheatdetect.models.lstm_ae import LSTMAutoencoderDetector

    lstm_data = prepare_lstm_data(config, data, features_to_keep)
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
            "input_dim": len(features_to_keep),
            "seq_len": lstm_data["seq_len"],
            "epochs": config.lstm_epochs,
            "patience": config.lstm_patience,
            "preprocessing": config.lstm_preprocessing,
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
    X_test = data["X_test"]
    y_test = data["y_test"]
    features_to_keep = data["features_to_keep"]

    # ---- Modeling ---------------------------------------------------------
    skewed_features = find_skewed_features(X_train, skew_threshold=config.skew_threshold)

    best_if, if_results = IsolationForestDetector.grid_search(
        X_train,
        X_val,
        y_val,
        skewed_features,
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
        skewed_features,
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

    selection_metric = "pr_auc"
    selection_scores = {
        name: average_precision_score(y_val, scores)
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
        skewed_features,
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
            config, data, features_to_keep
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
        "skewed_features": skewed_features,
        "val_scores": val_scores,
        "y_val": y_val,
        "y_test": y_test,
        "if_grid_results": if_results,
        "ocsvm_grid_results": ocsvm_results,
        "ensemble_grid_results": ensemble_results,
        "lstm_grid_results": lstm_grid_results,
    }

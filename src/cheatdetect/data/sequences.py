"""Sequence construction for temporal (sequence) anomaly detectors.

Flat models (Isolation Forest, One-Class SVM) consume one aggregated
feature vector per event-count window. The LSTM autoencoder instead needs a
*sequence* of feature vectors per window so it can model temporal dynamics.

To stay directly comparable to the flat models, each sequence maps 1:1 to a
flat chunk: the same parent window (``chunk_size`` events, ``step_size``
stride) is subdivided into overlapping micro-chunks
(``sub_chunk`` events, ``sub_step`` stride) whose feature vectors form the
sequence. The parent chunk's cheating label is the sequence label.

Public API:
    - :func:`extract_sequences_from_sessions` — build ``(n, seq_len, k)``.
    - :func:`augment_sequences` — noisy copies of normal sequences.
    - :func:`align_sequence_features` — project raw sequences onto the
      selected feature set.
"""

import numpy as np
import pandas as pd

from .augment import add_coordinate_noise
from .build import merge_window_switch_events
from .cleaning import clean_features
from .features.extract import extract_features_from_chunk


def sequence_length(chunk_size: int, sub_chunk: int, sub_step: int) -> int:
    """Number of micro-chunks produced inside a parent window."""
    return (chunk_size - sub_chunk) // sub_step + 1


def _iter_parent_chunks(
    df_clean: pd.DataFrame, chunk_size: int, step_size: int
):
    """Yield overlapping parent windows of ``chunk_size`` events."""
    for start in range(0, len(df_clean) - chunk_size + 1, step_size):
        yield df_clean.iloc[start : start + chunk_size]


def _parent_label(chunk: pd.DataFrame, cheating_threshold: float) -> int:
    """Label a parent window from the fraction of cheating-flagged events."""
    if "Is Cheating" not in chunk.columns:
        return 0
    return int(chunk["Is Cheating"].mean() >= cheating_threshold)


def _sequence_for_chunk(
    chunk: pd.DataFrame,
    sub_chunk: int,
    sub_step: int,
    seq_len: int,
    cheating_threshold: float,
) -> pd.DataFrame | None:
    """Build the micro-chunk feature sequence for a single parent window."""
    sub_rows = [
        extract_features_from_chunk(
            chunk.iloc[sub_start : sub_start + sub_chunk], cheating_threshold
        )
        for sub_start in range(0, len(chunk) - sub_chunk + 1, sub_step)
    ]
    if len(sub_rows) != seq_len:
        return None
    # Cast to float so the stacked array is numeric (the raw dict mixes bool
    # and integer columns, which would otherwise yield an object array).
    return pd.DataFrame(sub_rows).astype(float)


def _stack(sequences: list[np.ndarray]) -> np.ndarray:
    return np.stack(sequences, axis=0) if sequences else np.empty((0, 0, 0))


def extract_sequences_from_sessions(
    sessions: list[pd.DataFrame],
    chunk_size: int,
    step_size: int,
    sub_chunk: int,
    sub_step: int,
    cheating_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build 3D feature sequences from cleaned sessions.

    Sessions are processed in the given order so that the resulting rows
    align 1:1 with the flat pipeline's output (same chunk, same label).

    Args:
        sessions: List of cleaned session DataFrames (output of
            ``load_sessions`` + ``clean_session_data``), in the desired order.
        chunk_size: Events per parent window (must match the flat pipeline).
        step_size: Parent-window stride (must match the flat pipeline).
        sub_chunk: Events per micro-chunk.
        sub_step: Micro-chunk stride.
        cheating_threshold: Fraction of cheating events required to label a
            parent window as cheating.

    Returns:
        ``(X, y, feature_names)`` where ``X`` is
        ``(n_samples, seq_len, n_raw_features)``, ``y`` is the binary parent
        label, and ``feature_names`` are the raw micro-chunk feature columns.
    """
    seq_len = sequence_length(chunk_size, sub_chunk, sub_step)

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    feature_names: list[str] | None = None

    for df_clean in sessions:
        if len(df_clean) < chunk_size:
            continue
        for chunk in _iter_parent_chunks(df_clean, chunk_size, step_size):
            seq = _sequence_for_chunk(
                chunk, sub_chunk, sub_step, seq_len, cheating_threshold
            )
            if seq is None:
                continue
            if feature_names is None:
                feature_names = list(seq.columns)
            sequences.append(seq.values)
            labels.append(_parent_label(chunk, cheating_threshold))

    if not sequences:
        return (
            np.empty((0, seq_len, 0)),
            np.array([], dtype=int),
            feature_names or [],
        )

    return (
        _stack(sequences),
        np.array(labels, dtype=int),
        feature_names,
    )


def augment_sequences(
    sessions: list[pd.DataFrame],
    chunk_size: int,
    step_size: int,
    sub_chunk: int,
    sub_step: int,
    n_copies: int = 2,
    sigma_range: tuple[float, float] = (2.0, 5.0),
    screen_bounds: tuple[int, int] = (1920, 1080),
    cheating_threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create noisy copies of normal sequences.

    Mirrors :func:`augment_session_data`, but produces micro-chunk sequences:
    coordinate noise is applied to each raw parent window *before* the
    micro-chunks are extracted, so the derived kinematic features naturally
    reflect the noise.

    All produced sequences are normal (``y = 0``).

    Args:
        sessions: Cleaned normal training sessions.
        chunk_size, step_size: Parent-window geometry (must match pipeline).
        sub_chunk, sub_step: Micro-chunk geometry.
        n_copies: Number of noisy copies per parent window.
        sigma_range: ``(min, max)`` Gaussian noise sigma, drawn uniformly per
            copy.
        screen_bounds: ``(width, height)`` for coordinate clipping.
        cheating_threshold: Passed to micro-chunk extraction.
        random_state: Seed for reproducibility.

    Returns:
        ``(X, y, feature_names)`` with the same shape convention as
        :func:`extract_sequences_from_sessions`.
    """
    rng = np.random.default_rng(random_state)
    seq_len = sequence_length(chunk_size, sub_chunk, sub_step)

    sequences: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for df_clean in sessions:
        if len(df_clean) < chunk_size:
            continue
        for chunk in _iter_parent_chunks(df_clean, chunk_size, step_size):
            for _ in range(n_copies):
                sigma = rng.uniform(sigma_range[0], sigma_range[1])
                noisy = add_coordinate_noise(
                    chunk,
                    sigma=sigma,
                    screen_bounds=screen_bounds,
                    random_state=rng,
                )
                seq = _sequence_for_chunk(
                    noisy, sub_chunk, sub_step, seq_len, cheating_threshold
                )
                if seq is None:
                    continue
                if feature_names is None:
                    feature_names = list(seq.columns)
                sequences.append(seq.values)

    if not sequences:
        return (
            np.empty((0, seq_len, 0)),
            np.array([], dtype=int),
            feature_names or [],
        )

    n = len(sequences)
    return _stack(sequences), np.zeros(n, dtype=int), feature_names


def align_sequence_features(
    X_3d: np.ndarray,
    raw_feature_names: list[str],
    features_to_keep: list[str],
) -> np.ndarray:
    """Project raw sequences onto the curated feature set.

    Flattens the sequences, merges redundant window-switch columns, selects
    ``features_to_keep``, applies the standard feature cleaning (inf → NaN →
    median), then restores the 3D shape.

    Args:
        X_3d: ``(n_samples, seq_len, n_raw_features)`` raw sequences.
        raw_feature_names: Column names for the raw feature axis.
        features_to_keep: Curated feature list from the flat pipeline.

    Returns:
        ``(n_samples, seq_len, len(features_to_keep))`` aligned sequences.

    Raises:
        ValueError: If any curated feature is missing from the raw features.
    """
    if X_3d.size == 0:
        return np.empty((0, X_3d.shape[1] if X_3d.ndim == 3 else 0, 0))

    n_samples, seq_len, n_raw = X_3d.shape
    flat = pd.DataFrame(X_3d.reshape(-1, n_raw), columns=raw_feature_names)
    flat = merge_window_switch_events(flat)

    missing = [c for c in features_to_keep if c not in flat.columns]
    if missing:
        raise ValueError(
            f"LSTM raw features are missing curated columns: {missing}"
        )

    flat = flat[features_to_keep]
    flat, _ = clean_features(flat)
    return flat.values.reshape(n_samples, seq_len, len(features_to_keep))

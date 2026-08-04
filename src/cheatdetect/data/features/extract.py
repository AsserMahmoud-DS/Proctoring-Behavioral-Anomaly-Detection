"""Feature extraction orchestration.

Splits a cleaned session into overlapping event-count windows (chunks)
and delegates per-chunk feature computation to the domain-specific
modules (mouse, keyboard, actions).
"""

import logging
import pandas as pd

from .mouse import extract_mouse_features
from .keyboard import extract_keyboard_features
from .actions import extract_action_features

logger = logging.getLogger(__name__)


def extract_features_from_chunk(
    chunk: pd.DataFrame, cheating_threshold: float = 0.5
) -> dict:
    """Compute all features for a single event-count window.

    Separates the chunk into mouse and keyboard sub-frames, then
    aggregates action counts and computes the cheating label.

    Args:
        chunk: A DataFrame slice of consecutive events from a cleaned
            session (must include the flag columns added by
            ``clean_session_data``).
        cheating_threshold: If the fraction of cheating-flagged events
            in the chunk meets or exceeds this value, the chunk label
            is True.

    Returns:
        A flat dict with keys like ``mouse_*``, ``keyboard_*``,
        ``*_events``, ``elapsed_time``, and ``is_cheating``.
    """
    # Determine chunk-level cheating label from per-event annotations
    if "Is Cheating" in chunk.columns:
        cheating_ratio = chunk["Is Cheating"].mean()
        chunk_is_cheating = cheating_ratio >= cheating_threshold
    else:
        chunk_is_cheating = False

    # Degenerate chunks (all events at the same timestamp) get elapsed_time=0.0
    elapsed_time = float(
        chunk.iloc[-1]["Time (seconds)"] - chunk.iloc[0]["Time (seconds)"]
    )

    # Subset events by category
    mouse_chunk = chunk[chunk["is_mouse_event"]].copy()
    keyboard_chunk = chunk[chunk["is_keyboard_event"]].copy()

    mouse_features = extract_mouse_features(mouse_chunk)
    keyboard_features = extract_keyboard_features(keyboard_chunk)
    action_features = extract_action_features(chunk, elapsed_time)

    return {
        **action_features,
        **mouse_features,
        **keyboard_features,
        "is_cheating": chunk_is_cheating,
    }


def extract_features_from_session(
    df_session: pd.DataFrame,
    chunk_size: int = 5,
    step_size: int = 1,
    cheating_threshold: float = 0.5,
    session_name: str = "",
) -> pd.DataFrame:
    """Slide an event-count window over a session and extract features.

    Args:
        df_session: Cleaned session DataFrame (output of
            ``clean_session_data``).
        chunk_size: Number of consecutive events in each window.
        step_size: Number of events to advance the window each step.
            A value of 1 gives maximum overlap; equal to chunk_size
            gives non-overlapping windows.
        cheating_threshold: Fraction threshold for labeling a chunk
            as cheating.
        session_name: Label attached to every row for traceability.

    Returns:
        DataFrame with one row per window and a ``session_name`` column.
        Returns an empty DataFrame if the session is too short.
    """
    logger.info("Processing session: %s", session_name)

    if len(df_session) < chunk_size:
        logger.warning(
            "Insufficient data for session %s (%d rows, need %d)",
            session_name,
            len(df_session),
            chunk_size,
        )
        return pd.DataFrame()

    # Sliding window over events
    chunks = []
    for start in range(0, len(df_session) - chunk_size + 1, step_size):
        chunk = df_session.iloc[start : start + chunk_size]
        features = extract_features_from_chunk(chunk, cheating_threshold)
        chunks.append(features)

    features_df = pd.DataFrame(chunks)

    if not features_df.empty:
        features_df["session_name"] = session_name

    cheating_count = int(features_df["is_cheating"].sum())
    total = len(features_df)
    logger.info(
        "Session %s: %d chunks, %d cheating (%.1f%%)",
        session_name,
        total,
        cheating_count,
        cheating_count / total * 100 if total > 0 else 0,
    )

    return features_df
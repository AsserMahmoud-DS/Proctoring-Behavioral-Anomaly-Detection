"""Action-event feature extraction.

Counts browser-level signaling events (copy, paste, blur, focus,
tab-switch) that indicate potential cheating behavior such as
window-switching or clipboard use during an exam.
"""

import pandas as pd


def extract_action_features(chunk: pd.DataFrame, elapsed_time: float) -> dict:
    """Extract action-event counts and elapsed time for a chunk.

    Args:
        chunk: A DataFrame slice of consecutive events from a cleaned
            session (must include the ``is_copy``, ``is_paste``,
            ``is_blur``, ``is_focus``, and ``is_tab_switch`` flag
            columns created by ``clean_session_data``).
        elapsed_time: Time span of the chunk in seconds.

    Returns:
        Dictionary with ``elapsed_time`` and integer counts for each
        action type.
    """
    return {
        "elapsed_time": float(elapsed_time),
        "copy_events": int(chunk["is_copy"].sum()),
        "paste_events": int(chunk["is_paste"].sum()),
        "blur_events": int(chunk["is_blur"].sum()),
        "focus_events": int(chunk["is_focus"].sum()),
        "tab_switch_events": int(chunk["is_tab_switch"].sum()),
    }


def merge_window_switch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Merge blur, focus, and tab_switch events into window_switch_events."""
    df = df.copy()
    cols_available = [c for c in ["blur_events", "focus_events", "tab_switch_events"]
                      if c in df.columns]
    if len(cols_available) >= 2:
        df["window_switch_events"] = df[cols_available].sum(axis=1)
        for col in cols_available:
            df.drop(columns=[col], inplace=True)
    elif len(cols_available) == 1:
        df["window_switch_events"] = df[cols_available[0]]
        df.drop(columns=cols_available, inplace=True)
    return df
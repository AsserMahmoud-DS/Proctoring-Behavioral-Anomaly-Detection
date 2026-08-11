"""Keyboard typing behavior feature extraction.

Produces features that capture typing tempo from keydown events:
typing rate, burst count (rapid keypresses), and pause count
(keypress gaps indicating hesitation or context-switching).
"""

import pandas as pd


def extract_keyboard_features(keyboard_chunk: pd.DataFrame) -> dict:
    """Extract typing-rate features from keyboard events in a chunk.

    Args:
        keyboard_chunk: DataFrame filtered to keyboard events (must
            contain ``Time (seconds)``).

    Returns:
        Dictionary with keys:
            - ``keyboard_typing_rate``: keystrokes per second.
            - ``keyboard_burst_count``: inter-key gaps < 0.2s (fast typing).
            - ``keyboard_pause_count``: inter-key gaps > 1.0s (hesitation).
    """
    if len(keyboard_chunk) == 0:
        return {
            "keyboard_typing_rate": 0.0,
            "keyboard_burst_count": 0,
            "keyboard_pause_count": 0,
        }

    chunk = keyboard_chunk.sort_values("Time (seconds)")

    if len(chunk) >= 2:
        time_span = chunk.iloc[-1]["Time (seconds)"] - chunk.iloc[0]["Time (seconds)"]
        typing_rate = len(chunk) / time_span if time_span > 0 else 0.0

        time_diffs = chunk["Time (seconds)"].diff().dropna()
        burst_count = int((time_diffs < 0.2).sum())   # rapid succession
        pause_count = int((time_diffs > 1.0).sum())   # noticeable gap
    else:
        typing_rate = 0.0
        burst_count = 0
        pause_count = 0

    return {
        "keyboard_typing_rate": float(typing_rate),
        "keyboard_burst_count": burst_count,
        "keyboard_pause_count": pause_count,
    }
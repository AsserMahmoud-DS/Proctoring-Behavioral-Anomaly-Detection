"""Raw session data cleaning.

Transforms raw CSV data into a standardized format ready for feature
extraction: removes summary rows, coerces types, forward-fills missing
coordinates, and creates boolean event-category and action flags.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Event-type categories used to separate mouse, keyboard, and tab-switch
# events during feature extraction.
MOUSE_EVENTS = {"mousemove", "click"}
KEYBOARD_EVENTS = {"keydown", "copy", "paste"}
ACTION_EVENTS = {"tab-switch"}

# Columns kept from the raw CSV; all others are discarded.
USEFUL_COLUMNS = [
    "Time (seconds)",
    "Event Type",
    "X Coordinate",
    "Y Coordinate",
    "Action",
    "Is Cheating",
]

EXCLUDE_COLUMNS = {"is_cheating", "session_name", "chunk_start_time"}


def clean_session_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize a raw session DataFrame.

    Performs the following steps:
        1. Selects only useful columns (discards metadata).
        2. Removes rows after a "SESSION SUMMARY" marker.
        3. Coerces Time to numeric and drops rows with invalid times.
        4. Creates boolean flags for event categories (mouse/keyboard/action).
        5. Forward-fills missing X/Y coordinates and drops remaining NaNs.
        6. Creates binary action flags (copy, paste, blur, focus, tab-switch).
        7. Casts Is Cheating to bool (defaults to False if absent).

    Args:
        df: Raw session DataFrame as loaded from CSV.

    Returns:
        Cleaned DataFrame with added flag columns and no summary rows.
    """
    logger.info("Input shape: %s", df.shape)

    # Keep only columns that exist in the data
    available_columns = [col for col in USEFUL_COLUMNS if col in df.columns]
    df_clean = df[available_columns].copy()

    # Strip the session summary footer if present
    summary_idx = df_clean[df_clean.iloc[:, 0] == "SESSION SUMMARY"].index
    if len(summary_idx) > 0:
        df_clean = df_clean.iloc[: summary_idx[0]]

    # Ensure time is numeric; drop rows where it isn't
    df_clean["Time (seconds)"] = pd.to_numeric(
        df_clean["Time (seconds)"], errors="coerce"
    )
    df_clean = df_clean.dropna(subset=["Time (seconds)"])

    # Tag each row with its event category
    df_clean["is_mouse_event"] = df_clean["Event Type"].isin(MOUSE_EVENTS)
    df_clean["is_keyboard_event"] = df_clean["Event Type"].isin(KEYBOARD_EVENTS)
    df_clean["is_action_event"] = df_clean["Event Type"].isin(ACTION_EVENTS)

    # Forward-fill coordinates (mousemove carries position; keydown/paste do not)
    if "X Coordinate" in df_clean.columns and "Y Coordinate" in df_clean.columns:
        df_clean["X Coordinate"] = pd.to_numeric(
            df_clean["X Coordinate"], errors="coerce"
        )
        df_clean["Y Coordinate"] = pd.to_numeric(
            df_clean["Y Coordinate"], errors="coerce"
        )
        df_clean["X Coordinate"] = df_clean["X Coordinate"].ffill()
        df_clean["Y Coordinate"] = df_clean["Y Coordinate"].ffill()
        df_clean = df_clean.dropna(subset=["X Coordinate", "Y Coordinate"])

    # Binary action flags
    if "Action" in df_clean.columns:
        df_clean["is_copy"] = (
            (df_clean["Action"] == "copy") | (df_clean["Event Type"] == "copy")
        ).astype(int)
        df_clean["is_paste"] = (
            (df_clean["Action"] == "paste") | (df_clean["Event Type"] == "paste")
        ).astype(int)
        df_clean["is_blur"] = (df_clean["Action"] == "blur").astype(int)
        df_clean["is_focus"] = (df_clean["Action"] == "focus").astype(int)
    else:
        for col in ["is_copy", "is_paste", "is_blur", "is_focus"]:
            df_clean[col] = 0

    df_clean["is_tab_switch"] = (df_clean["Event Type"] == "tab-switch").astype(int)

    # Cheating label (used during supervised training; ignored for unsupervised)
    # .astype(bool) would make any non-empty string True (e.g. "FALSE" → True),
    # so we compare to the literal "TRUE" instead.
    if "Is Cheating" in df_clean.columns:
        df_clean["Is Cheating"] = (
            df_clean["Is Cheating"].astype(str).str.strip().str.upper() == "TRUE"
        )
    else:
        df_clean["Is Cheating"] = False

    logger.info("Cleaned shape: %s", df_clean.shape)
    return df_clean


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a feature DataFrame for model training.

    Strips label/metadata columns, replaces ±inf with NaN, then
    fills all NaNs with column medians.

    Args:
        df: DataFrame as produced by ``extract_features_from_session``
            (includes ``is_cheating`` and ``session_name`` columns).

    Returns:
        A tuple ``(X, feature_cols)`` where:
            - **X** is a clean numeric DataFrame ready for an estimator.
            - **feature_cols** is the list of column names used.
    """
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    return X, feature_cols
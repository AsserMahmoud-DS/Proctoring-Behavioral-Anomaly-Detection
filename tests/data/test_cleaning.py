import numpy as np
import pandas as pd

from cheatdetect.data import clean_features, clean_session_data


def test_clean_session_data_drops_extra_columns_and_invalid_rows():
    raw = pd.DataFrame({
        "Time (seconds)": ["0.0", "invalid", "1.5", "SESSION SUMMARY", "9.0"],
        "Event Type": ["mousemove", "click", "keydown", None, "click"],
        "X Coordinate": [10, 10, 10, None, 20],
        "Y Coordinate": [20, 20, 20, None, 30],
        "Element Tag": ["div"] * 5,
    })

    cleaned = clean_session_data(raw)

    assert "Element Tag" not in cleaned.columns
    assert cleaned["Time (seconds)"].tolist() == [0.0, 1.5]
    assert cleaned["Event Type"].tolist() == ["mousemove", "keydown"]
    assert pd.api.types.is_numeric_dtype(cleaned["Time (seconds)"])


def test_clean_session_data_forward_fills_coordinates_after_first_position():
    raw = pd.DataFrame({
        "Time (seconds)": [0.0, 0.5, 1.0],
        "Event Type": ["keydown", "mousemove", "keydown"],
        "X Coordinate": [None, "10", None],
        "Y Coordinate": [None, "20", None],
    })

    cleaned = clean_session_data(raw)

    assert cleaned["Time (seconds)"].tolist() == [0.5, 1.0]
    assert cleaned["X Coordinate"].tolist() == [10.0, 10.0]
    assert cleaned["Y Coordinate"].tolist() == [20.0, 20.0]
    assert pd.api.types.is_numeric_dtype(cleaned["X Coordinate"])
    assert pd.api.types.is_numeric_dtype(cleaned["Y Coordinate"])


def test_clean_session_data_sets_event_action_and_cheating_flags():
    raw = pd.DataFrame({
        "Time (seconds)": list(range(8)),
        "Event Type": [
            "mousemove", "click", "keydown", "copy", "paste",
            "tab-switch", "click", "click",
        ],
        "Action": [None, "copy", None, None, None, None, "blur", "focus"],
        "Is Cheating": ["FALSE", " TRUE ", "FALSE", "FALSE", "FALSE", "FALSE", "FALSE", "FALSE"],
    })

    cleaned = clean_session_data(raw)

    assert cleaned["is_mouse_event"].tolist() == [True, True, False, False, False, False, True, True]
    assert cleaned["is_keyboard_event"].tolist() == [False, False, True, True, True, False, False, False]
    assert cleaned["is_action_event"].tolist() == [False, False, False, False, False, True, False, False]
    assert cleaned["is_copy"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]
    assert cleaned["is_paste"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    assert cleaned["is_blur"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0]
    assert cleaned["is_focus"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert cleaned["is_tab_switch"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert cleaned["Is Cheating"].tolist() == [False, True, False, False, False, False, False, False]
    assert pd.api.types.is_bool_dtype(cleaned["Is Cheating"])


def test_clean_features_excludes_labels_and_fills_missing_values():
    extracted = pd.DataFrame({
        "speed": [1.0, np.nan, 3.0],
        "action_count": [0.0, np.inf, 2.0],
        "is_cheating": [False, False, True],
        "session_name": ["s1"] * 3,
        "chunk_start_time": [0.0, 1.0, 2.0],
    })

    X, columns = clean_features(extracted)

    assert columns == ["speed", "action_count"]
    assert X.columns.tolist() == columns
    assert X["speed"].tolist() == [1.0, 2.0, 3.0]
    assert X["action_count"].tolist() == [0.0, 2.0, 2.0]
    assert np.isfinite(X.to_numpy()).all()


import pandas as pd

from cheatdetect.data import (
    clean_session_data,
    extract_features_from_chunk,
    extract_features_from_session,
)


def test_chunk_label_uses_cheating_event_ratio():
    raw = pd.DataFrame(
        {
            "Time (seconds)": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Event Type": ["mousemove"] * 5,
            "X Coordinate": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Y Coordinate": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Is Cheating": ["FALSE"] * 5,
        }
    )
    normal = clean_session_data(raw)
    raw.loc[1:, "Is Cheating"] = "TRUE"
    cheating = clean_session_data(raw)

    assert not extract_features_from_chunk(normal)["is_cheating"]
    assert extract_features_from_chunk(cheating)["is_cheating"]


def test_session_extracts_overlapping_windows_in_order():
    raw = pd.DataFrame(
        {
            "Time (seconds)": [0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0],
            "Event Type": ["mousemove"] * 7,
            "X Coordinate": [float(i) for i in range(7)],
            "Y Coordinate": [float(i) for i in range(7)],
            "Is Cheating": ["FALSE"] * 7,
        }
    )

    extracted = extract_features_from_session(
        clean_session_data(raw), chunk_size=5, step_size=1, session_name="student_1"
    )

    assert len(extracted) == 3
    assert extracted["elapsed_time"].tolist() == [7.0, 10.0, 14.0]
    assert extracted["is_cheating"].tolist() == [False] * 3
    assert extracted["session_name"].tolist() == ["student_1"] * 3

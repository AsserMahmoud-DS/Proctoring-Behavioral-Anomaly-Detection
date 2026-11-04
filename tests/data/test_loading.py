"""Tests for loading raw session CSV files."""

import pandas as pd

from cheatdetect.data import load_sessions, load_single_session


def test_load_single_session(tmp_path):
    expected = pd.DataFrame(
        {
            "Time (seconds)": [0.0, 0.5],
            "Event Type": ["mousemove", "keydown"],
        }
    )
    path = tmp_path / "student_1.csv"
    expected.to_csv(path, index=False)

    actual = load_single_session(path)

    pd.testing.assert_frame_equal(actual, expected)


def test_load_sessions_from_directory(tmp_path):
    first = pd.DataFrame({"Time (seconds)": [0.0], "Event Type": ["click"]})
    second = pd.DataFrame({"Time (seconds)": [1.0], "Event Type": ["keydown"]})
    first.to_csv(tmp_path / "student_1.csv", index=False)
    second.to_csv(tmp_path / "student_2.csv", index=False)

    sessions = load_sessions(tmp_path)

    assert set(sessions) == {"student_1.csv", "student_2.csv"}
    pd.testing.assert_frame_equal(sessions["student_1.csv"], first)
    pd.testing.assert_frame_equal(sessions["student_2.csv"], second)


def test_load_single_missing_session_returns_empty_frame(tmp_path):
    actual = load_single_session(tmp_path / "missing.csv")

    assert actual.empty

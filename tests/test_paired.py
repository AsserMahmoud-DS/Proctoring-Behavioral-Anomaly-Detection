import numpy as np
import pandas as pd
import pytest

from cheatdetect.data import (
    build_paired_representation,
    extract_paired_features,
    load_parent_sessions,
)
from cheatdetect.data.augment import add_coordinate_noise
from cheatdetect.data.feature_schema import SOURCE_FEATURES


def test_shared_realized_noise_and_originals_unchanged(make_clean_session):
    cleaned = make_clean_session(n_events=75, seed=42)
    before = cleaned.copy(deep=True)
    rng = np.random.default_rng(42)
    sigma = rng.uniform(2, 5)
    noisy = add_coordinate_noise(cleaned.iloc[:50], sigma, random_state=rng)
    parent, micro, label = extract_paired_features(noisy)
    result = build_paired_representation({"session.csv": cleaned}, synthetic=True)
    np.testing.assert_allclose(result.parent.iloc[0], parent)
    np.testing.assert_allclose(result.sequences[0], micro)
    assert result.labels[0] == label == 0
    assert result.sequences.shape == (4, 9, 25)
    assert result.identities["copy_index"].tolist() == [1, 2, 1, 2]
    assert result.identities["parent_start_event"].tolist() == [0, 0, 25, 25]
    pd.testing.assert_frame_equal(cleaned, before)
    repeated = build_paired_representation({"session.csv": cleaned}, synthetic=True)
    pd.testing.assert_frame_equal(result.parent, repeated.parent)
    np.testing.assert_array_equal(result.sequences, repeated.sequences)


def test_custom_geometry_and_parent_labels(make_clean_session):
    cleaned = make_clean_session(n_events=40, cheating=True, seed=42)
    result = build_paired_representation(
        {"session.csv": cleaned}, chunk_size=20, step_size=10, sub_chunk=5, sub_step=5,
    )
    assert result.parent.shape == (3, 25)
    assert result.sequences.shape == (3, 4, 25)
    assert list(result.parent.columns) == list(SOURCE_FEATURES)
    assert result.labels.tolist() == [1, 1, 1]
    assert result.identities["parent_start_event"].tolist() == [0, 10, 20]


def test_empty_representation_preserves_schema_and_geometry():
    result = build_paired_representation({}, chunk_size=20, sub_chunk=5, sub_step=5)
    assert result.parent.shape == (0, 25)
    assert result.sequences.shape == (0, 4, 25)
    assert len(result.labels) == len(result.identities) == 0


@pytest.mark.parametrize("parameters", [
    {"chunk_size": 0}, {"step_size": 0}, {"sub_chunk": 60}, {"sub_step": 0},
    {"n_copies": 0}, {"sigma_range": (-1, 2)}, {"screen_bounds": (0, 1080)},
    {"cheating_threshold": 1.5},
])
def test_invalid_parameters_fail(parameters):
    with pytest.raises(ValueError):
        build_paired_representation({}, **parameters)


def test_anomalous_windows_cannot_be_augmented(make_clean_session):
    with pytest.raises(ValueError, match="anomalous"):
        build_paired_representation(
            {"session.csv": make_clean_session(n_events=75, cheating=True)}, synthetic=True,
        )


def test_loader_reports_exclusions_and_keeps_contributing_order(tmp_path, make_clean_session):
    paths = [tmp_path / name for name in ("first.csv", "short.csv", "summary.csv", "last.csv")]
    make_clean_session(n_events=75).to_csv(paths[0], index=False)
    make_clean_session(n_events=49).to_csv(paths[1], index=False)
    pd.DataFrame([{
        "Time (seconds)": "SESSION SUMMARY", "Event Type": None,
        "X Coordinate": None, "Y Coordinate": None,
    }]).to_csv(paths[2], index=False)
    make_clean_session(n_events=50).to_csv(paths[3], index=False)
    sessions, report = load_parent_sessions(paths, partition="test")
    assert list(sessions) == ["first.csv", "last.csv"]
    assert report["included"].tolist() == [True, False, False, True]
    assert report["parent_windows"].tolist() == [2, 0, 0, 1]
    assert report["exclusion_reason"].iloc[1] == "insufficient_events_for_50_event_window"
    assert report["exclusion_reason"].iloc[2] == "no_cleaned_events"


def test_loader_rejects_empty_partition_duplicates_and_anomalous_normal_inputs(tmp_path, make_clean_session):
    path = tmp_path / "session.csv"
    make_clean_session(n_events=49).to_csv(path, index=False)
    with pytest.raises(ValueError, match="No usable parent windows"):
        load_parent_sessions([path], partition="train")
    with pytest.raises(ValueError, match="Duplicate"):
        load_parent_sessions([path, path])
    make_clean_session(n_events=75, cheating=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Anomalous"):
        load_parent_sessions([path], normal_only=True)

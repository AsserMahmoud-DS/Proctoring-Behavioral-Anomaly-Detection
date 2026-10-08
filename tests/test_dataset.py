import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from cheatdetect.data.augment import add_coordinate_noise
from cheatdetect.data.cleaning import clean_session_data
from cheatdetect.data import dataset as data
from cheatdetect.data.feature_schema import SOURCE_FEATURES


def raw_session(seed=42, cheating=False):
    rng = np.random.default_rng(seed)
    coordinates = 100 + np.cumsum(rng.normal(0, 2, (75, 2)), axis=0)
    return pd.DataFrame({
        "Time (seconds)": np.arange(75) * 0.1,
        "Event Type": np.where(np.arange(75) % 2, "keydown", "mousemove"),
        "X Coordinate": coordinates[:, 0], "Y Coordinate": coordinates[:, 1],
        "Action": "", "Is Cheating": cheating,
    })


@pytest.fixture
def inputs(tmp_path):
    normal = tmp_path / "normal"
    mixed = tmp_path / "mixed"
    normal.mkdir()
    mixed.mkdir()
    split = {"normal_train": ["train.csv"], "normal_val": ["normal_val.csv"],
             "mixed_val": ["mixed_val.csv"], "mixed_test": ["test.csv"]}
    for index, (key, names) in enumerate(split.items()):
        directory = normal if key.startswith("normal") else mixed
        raw_session(index, key.startswith("mixed")).to_csv(directory / names[0], index=False)
    return split, normal, mixed


def test_realized_noise_is_shared_by_parent_and_micro_features():
    cleaned = clean_session_data(raw_session())
    original = cleaned.copy(deep=True)
    rng = np.random.default_rng(42)
    sigma = rng.uniform(2, 5)
    noisy = add_coordinate_noise(cleaned.iloc[:50], sigma, (1920, 1080), rng)
    expected_parent, expected_micro, expected_label = data._paired_features(noisy)
    result = data.build_representation({"session.csv": cleaned}, synthetic=True)
    assert result.parent.shape == (4, 25)
    assert result.sequences.shape == (4, 9, 25)
    np.testing.assert_allclose(result.parent.iloc[0], expected_parent)
    np.testing.assert_allclose(result.sequences[0], expected_micro)
    assert result.labels[0] == expected_label == 0
    assert result.identities["copy_index"].tolist() == [1, 2, 1, 2]
    pd.testing.assert_frame_equal(result.parent, data.build_representation({"session.csv": cleaned}, synthetic=True).parent)
    pd.testing.assert_frame_equal(cleaned, original)


def test_preparation_fits_originals_once_and_reuses_held_out_views(inputs):
    split, normal, mixed = inputs
    prepared = data.prepare_study(split, normal, mixed)
    assert set(prepared.transformed) == {"IF-raw", "IF-log", "SVM-base", "SVM-log", "SVM-YJ", "SVM-quantile", "AE-base", "AE-log", "AE-YJ"}
    before = joblib.hash(prepared.preprocessors)
    for recipe, processor in prepared.preprocessors.items():
        held_out = {key: prepared.transformed[recipe][key] for key in ("normal_val", "mixed_val", "mixed_test")}
        assert processor.n_original_rows_ == (18 if recipe.startswith("AE-") else 2)
        assert len(prepared.training(recipe, False)) == 2
        assert len(prepared.training(recipe, True)) == 6
        originals = prepared.transformed[recipe]["train_original"]
        np.testing.assert_array_equal(np.asarray(prepared.training(recipe, True))[:2], np.asarray(originals))
        for key, view in held_out.items():
            assert prepared.transformed[recipe][key] is view
    assert joblib.hash(prepared.preprocessors) == before
    assert len(prepared.training_identities(False)) == 2
    assert len(prepared.training_identities(True)) == 6
    assert prepared.training_identities(True)["session_file"].nunique() == 1
    parent_mean = prepared.preprocessors["SVM-base"].scaler_.mean_[0]
    micro_mean = prepared.preprocessors["AE-base"].scaler_.mean_[0]
    assert parent_mean != micro_mean
    np.testing.assert_array_equal(prepared.raw["mixed_test"].labels, 1)
    for view in prepared.raw.values():
        assert len(view.parent) == len(view.sequences) == len(view.labels) == len(view.identities)


def test_source_changes_and_augmentation_invalidate_manifest(inputs):
    split, normal, mixed = inputs
    original = data.input_manifest(split, normal, mixed)
    assert original != data.input_manifest(split, normal, mixed, n_copies=3)
    with (normal / "train.csv").open("a") as stream:
        stream.write("\n")
    assert original != data.input_manifest(split, normal, mixed)


def test_persistence_rejects_stale_manifest_and_production_paths(inputs, tmp_path, monkeypatch):
    split, normal, mixed = inputs
    prepared = data.prepare_study(split, normal, mixed)
    monkeypatch.setattr(data, "ARTIFACT_ROOT", tmp_path / "artifacts")
    directory = data.ARTIFACT_ROOT / "phase_02_readiness"
    data.save_prepared(prepared, directory)
    restored = data.load_prepared(directory, data.input_manifest(split, normal, mixed))
    assert restored.gamma_centers == prepared.gamma_centers
    np.testing.assert_allclose(restored.training("AE-log", True), prepared.training("AE-log", True))
    metadata = json.loads((directory / "preprocessing.json").read_text())
    assert len(metadata["output_features"]) == 33
    pd.testing.assert_frame_equal(restored.session_report, prepared.session_report)
    assert (directory / "session_readiness.csv").is_file()
    stale = copy.deepcopy(prepared.manifest)
    stale["augmentation"]["seed"] = 123
    with pytest.raises(ValueError, match="manifest"):
        data.load_prepared(directory, stale)
    with pytest.raises(ValueError, match="study/artifacts"):
        data.save_prepared(prepared, tmp_path / "production")


def test_overlapping_sessions_and_anomalous_normal_training_fail(inputs):
    split, normal, mixed = inputs
    overlap = {**split, "normal_val": split["normal_train"]}
    with pytest.raises(ValueError, match="overlap"):
        data.prepare_study(overlap, normal, mixed)
    raw_session(cheating=True).to_csv(normal / "train.csv", index=False)
    with pytest.raises(ValueError, match="Anomalous"):
        data.prepare_study(split, normal, mixed)


def test_held_out_extremes_do_not_change_fitted_state(inputs):
    split, normal, mixed = inputs
    first = data.prepare_study(split, normal, mixed)
    extreme = raw_session(seed=13)
    extreme["Time (seconds)"] *= 1000
    extreme.to_csv(normal / "normal_val.csv", index=False)
    second = data.prepare_study(split, normal, mixed, n_copies=3)
    assert joblib.hash(first.preprocessors) == joblib.hash(second.preprocessors)
    assert first.gamma_centers == second.gamma_centers
    assert first.manifest != second.manifest


def test_default_preparation_checks_frozen_training_hashes(inputs, tmp_path, monkeypatch):
    split, normal, mixed = inputs
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "manifest.json").write_text(json.dumps({"split": split}))
    pd.DataFrame([{
        "session_file": "train.csv",
        "sha256": hashlib.sha256((normal / "train.csv").read_bytes()).hexdigest(),
    }]).to_csv(audit_dir / "sessions.csv", index=False)
    monkeypatch.setattr(data, "AUDIT_DIR", audit_dir)
    with (normal / "train.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="frozen Phase 1"):
        data.prepare_study(normal_dir=normal, mixed_dir=mixed)


@pytest.mark.parametrize("partition", data.SPLIT_KEYS)
def test_empty_required_partition_fails(inputs, partition):
    split, normal, mixed = inputs
    directory = normal if partition.startswith("normal") else mixed
    raw_session().iloc[:49].to_csv(directory / split[partition][0], index=False)
    with pytest.raises(ValueError, match=f"No usable parent windows in partition {partition}"):
        data.prepare_study(split, normal, mixed)


def test_zero_window_files_are_reported_without_changing_split(inputs):
    split, normal, mixed = inputs
    split = {**split, "mixed_test": ["test.csv", "summary.csv", "short.csv"]}
    frozen_split = copy.deepcopy(split)
    raw_session().iloc[:49].to_csv(mixed / "short.csv", index=False)
    pd.DataFrame([{
        "Time (seconds)": "SESSION SUMMARY", "Event Type": None,
        "X Coordinate": None, "Y Coordinate": None, "Action": None,
        "Is Cheating": None,
    }]).to_csv(mixed / "summary.csv", index=False)
    prepared = data.prepare_study(split, normal, mixed)
    assert split == frozen_split
    assert prepared.manifest["split"]["mixed_test"] == frozen_split["mixed_test"]
    report = prepared.session_report.query("partition == 'mixed_test'").set_index("session_file")
    assert len(report) == 3 and report["included"].sum() == 1
    assert report.loc["summary.csv", "cleaned_events"] == 0
    assert report.loc["summary.csv", "exclusion_reason"] == "no_cleaned_events"
    assert report.loc["short.csv", "cleaned_events"] == 49
    assert report.loc["short.csv", "exclusion_reason"] == "insufficient_events_for_50_event_window"
    assert report.loc[["summary.csv", "short.csv"], "parent_windows"].eq(0).all()
    test_view = prepared.raw["mixed_test"]
    assert test_view.identities["session_file"].unique().tolist() == ["test.csv"]
    assert len(test_view.parent) == len(test_view.sequences) == len(test_view.labels) == 2

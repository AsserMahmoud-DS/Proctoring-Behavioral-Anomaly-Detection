import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from cheatdetect.experiments.experiment_1 import audit
from cheatdetect.experiments.experiment_1.audit import (
    SOURCE_FEATURES,
    audit_recipe,
    feature_statistics,
    kernel_geometry,
)


@pytest.fixture
def original_features():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(0.1, 4, (30, 25)), columns=SOURCE_FEATURES)
    frame["mouse_direction_class"] = np.arange(30) % 9
    frame["copy_events"] = 0
    frame["mouse_jerk_mean"] = np.linspace(-100, 100, 30)
    frame["mouse_curvature_min"] = 0
    return frame


@pytest.mark.parametrize("recipe", ["base", "log", "yj", "quantile"])
def test_recipes_keep_constants_and_fixed_direction_columns(original_features, recipe):
    transformed, _ = audit_recipe(original_features, recipe, scale=True)
    assert transformed.shape == (30, 33)
    assert np.isfinite(transformed.to_numpy()).all()
    assert transformed["copy_events"].eq(0).all()
    assert transformed["mouse_curvature_min"].eq(0).all()
    indicators = transformed.filter(like="mouse_direction_class_")
    assert indicators.shape[1] == 9
    assert indicators.sum(axis=1).eq(1).all()


def test_signed_jerk_uses_signed_log(original_features):
    transformed, _ = audit_recipe(original_features, "log", scale=False)
    values = original_features["mouse_jerk_mean"]
    np.testing.assert_allclose(
        transformed["mouse_jerk_mean"], np.sign(values) * np.log1p(np.abs(values)),
    )


def test_statistics_do_not_hide_nonfinite_values():
    statistics = feature_statistics(pd.DataFrame({"feature": [0, np.inf, np.nan, -1]}))
    assert statistics.loc[0, "nonfinite_count"] == 2
    assert statistics.loc[0, "negative_fraction"] == 0.25


def test_gamma_uses_all_entry_variance_and_is_repeatable(original_features):
    transformed, _ = audit_recipe(original_features, "base", scale=True)
    rows = kernel_geometry(transformed)
    assert rows == kernel_geometry(transformed)
    expected = 1 / (33 * transformed.to_numpy().var(ddof=0))
    assert rows[1]["gamma"] == pytest.approx(expected)
    assert rows[0]["kernel_median"] >= rows[1]["kernel_median"] >= rows[2]["kernel_median"]


def test_recipe_rejects_unknown_direction(original_features):
    original_features.loc[0, "mouse_direction_class"] = 9
    with pytest.raises(ValueError, match="direction"):
        audit_recipe(original_features, "base", scale=True)


def test_full_audit_reads_only_original_training_contents(tmp_path, monkeypatch):
    normal_dir = tmp_path / "normal"
    mixed_dir = tmp_path / "mixed"
    normal_dir.mkdir()
    mixed_dir.mkdir()
    for index in range(5):
        raw = pd.DataFrame({
            "Time (seconds)": np.arange(75, dtype=float),
            "Event Type": "mousemove", "Action": "",
            "X Coordinate": np.arange(75), "Y Coordinate": 0,
            "Is Cheating": False,
        })
        raw.to_csv(normal_dir / f"normal_{index}.csv", index=False)
        (mixed_dir / f"mixed_{index}.csv").write_text("must never be read\n")
    normal_train, normal_val = train_test_split(
        sorted(normal_dir.glob("*.csv")), test_size=0.2, random_state=42,
    )
    mixed_val, mixed_test = train_test_split(
        sorted(mixed_dir.glob("*.csv")), test_size=0.7, random_state=42,
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({
        key: [path.name for path in paths] for key, paths in {
            "normal_train": normal_train, "normal_val": normal_val,
            "mixed_val": mixed_val, "mixed_test": mixed_test,
        }.items()
    }))
    monkeypatch.setattr(audit, "NORMAL_DIR", normal_dir)
    monkeypatch.setattr(audit, "MIXED_DIR", mixed_dir)
    monkeypatch.setattr(audit, "SPLIT_INFO_PATH", split_path)
    read_paths = []
    read_csv = pd.read_csv

    def record_read(path, *args, **kwargs):
        read_paths.append(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", record_read)
    manifest = audit.run_original_training_audit(tmp_path / "report")
    assert read_paths == normal_train
    assert manifest["parent_rows"] == 8
    assert manifest["micro_rows_with_overlap"] == 72
    assert manifest["encoded_dimension"] == len(manifest["output_features"]) == 33
    assert manifest["augmentation"] is False
    assert manifest["read_held_out_contents"] is False

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from cheatdetect.data.feature_schema import CONTINUOUS, COUNT_LOG, SIGNED, SOURCE_FEATURES
from cheatdetect.data.preprocessing import FeaturePreprocessor, gamma_center
from cheatdetect.data.feature_schema import OUTPUT_FEATURES, SCALED


@pytest.fixture
def features():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(0.1, 5, (40, 25)), columns=SOURCE_FEATURES)
    frame["mouse_direction_class"] = np.arange(40) % 9
    frame[["mouse_straightness", "mouse_idle_time_ratio"]] /= 5
    counts = ["copy_events", "paste_events", "mouse_direction_changes", "mouse_click_count", "mouse_sharp_angles", *COUNT_LOG]
    frame[counts] = frame[counts].round()
    frame[list(SIGNED)] -= 2.5
    return frame


@pytest.mark.parametrize("recipe", ["base", "log", "yj", "quantile"])
def test_fixed_encoding_scaling_and_serialization(features, recipe, tmp_path):
    processor = FeaturePreprocessor(recipe).fit(features)
    output = processor.transform(features)
    assert list(output.columns) == list(OUTPUT_FEATURES)
    assert output.shape == (40, 33)
    np.testing.assert_allclose(output[list(SCALED)].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(output[list(SCALED)].std(ddof=0), 1, atol=1e-12)
    np.testing.assert_array_equal(output["copy_events"], features["copy_events"])
    np.testing.assert_array_equal(output["paste_events"], features["paste_events"])
    expected_switches = features["window_switch_events"] if recipe == "base" else np.log1p(features["window_switch_events"])
    np.testing.assert_allclose(output["window_switch_events"], expected_switches)
    np.testing.assert_array_equal(output.filter(like="mouse_direction_class_").sum(axis=1), 1)
    path = tmp_path / "processor.joblib"
    joblib.dump(processor, path)
    pd.testing.assert_frame_equal(output, joblib.load(path).transform(features))


def test_exact_selective_log_operations(features):
    processor = FeaturePreprocessor("log", scale=False).fit(features)
    output = processor.transform(features)
    for feature in CONTINUOUS:
        values = features[feature]
        expected = np.sign(values) * np.log1p(np.abs(values)) if feature in SIGNED else np.log1p(values)
        np.testing.assert_allclose(output[feature], expected)
    for feature in COUNT_LOG:
        np.testing.assert_allclose(output[feature], np.log1p(features[feature]))
    for feature in ("mouse_sum_of_angles", "mouse_straightness", "mouse_click_count"):
        np.testing.assert_array_equal(output[feature], features[feature])


@pytest.mark.parametrize("recipe", ["base", "log", "yj", "quantile"])
def test_constants_and_absent_categories_remain(features, recipe):
    features["mouse_curvature_min"] = 2
    features["copy_events"] = 0
    features["mouse_direction_class"] = 0
    processor = FeaturePreprocessor(recipe).fit(features)
    output = processor.transform(features)
    np.testing.assert_allclose(output["mouse_curvature_min"], 0, atol=1e-12)
    assert output["copy_events"].eq(0).all()
    assert processor.scaler_.scale_[list(SCALED).index("mouse_curvature_min")] == 1
    assert "mouse_curvature_min" not in processor.transformers_
    future = features.iloc[:1].copy()
    future["mouse_curvature_min"] = 3
    future["copy_events"] = 1
    future["mouse_direction_class"] = 8
    transformed = processor.transform(future)
    assert transformed["copy_events"].iloc[0] == 1
    assert transformed["mouse_direction_class_8"].iloc[0] == 1
    assert transformed["mouse_curvature_min"].iloc[0] != 0


def test_imputation_and_statistics_are_frozen_on_originals(features):
    features.loc[0, "elapsed_time"] = np.inf
    processor = FeaturePreprocessor("log").fit(features)
    before = joblib.hash(processor)
    future = features.iloc[:2].copy()
    future["elapsed_time"] = [np.nan, 1e6]
    output = processor.transform(future)
    expected = (np.log1p(features["elapsed_time"].replace(np.inf, np.nan).median()) - processor.scaler_.mean_[0]) / processor.scaler_.scale_[0]
    assert output["elapsed_time"].iloc[0] == pytest.approx(expected)
    assert np.isfinite(output.to_numpy()).all()
    assert joblib.hash(processor) == before


def test_ae_flattens_original_timesteps_once(features):
    sequences = np.stack([features.iloc[:9].to_numpy(), features.iloc[9:18].to_numpy()])
    flattened = pd.DataFrame(sequences.reshape(-1, 25), columns=SOURCE_FEATURES)
    processor = FeaturePreprocessor("log").fit(flattened)
    output = processor.transform_sequences(sequences)
    assert output.shape == (2, 9, 33)
    np.testing.assert_allclose(output.reshape(-1, 33), processor.transform(flattened))
    assert processor.n_original_rows_ == 18
    assert processor.transform_sequences(np.empty((0, 9, 25))).shape == (0, 9, 33)


def test_sequence_transform_supports_other_window_geometry(features):
    sequences = features.iloc[:12].to_numpy().reshape(3, 4, 25)
    processor = FeaturePreprocessor().fit(features)
    assert processor.transform_sequences(sequences).shape == (3, 4, 33)


def test_preprocessor_is_serializable_inside_sklearn_pipeline(features, tmp_path):
    pipeline = Pipeline([
        ("preprocessing", FeaturePreprocessor("log", scale=False)),
        ("model", IsolationForest(n_estimators=3, max_samples=8, random_state=42)),
    ]).fit(features)
    scores = pipeline.decision_function(features)
    path = tmp_path / "detector.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_array_equal(scores, joblib.load(path).decision_function(features))


@pytest.mark.parametrize("feature,value", [
    ("mouse_direction_class", 9), ("mouse_direction_class", np.nan),
    ("mouse_path_length", -1), ("mouse_straightness", 2), ("copy_events", 0.5),
])
def test_invalid_domains_fail(features, feature, value):
    features.loc[0, feature] = value
    with pytest.raises(ValueError):
        FeaturePreprocessor().fit(features)


def test_missing_schema_and_all_missing_training_column_fail(features):
    with pytest.raises(ValueError, match="25"):
        FeaturePreprocessor().fit(features.drop(columns="elapsed_time"))
    features["elapsed_time"] = np.nan
    with pytest.raises(ValueError, match="finite training"):
        FeaturePreprocessor().fit(features)


def test_gamma_is_training_entry_variance(features):
    output = FeaturePreprocessor().fit_transform(features)
    assert gamma_center(output) == pytest.approx(1 / (33 * output.to_numpy().var(ddof=0)))

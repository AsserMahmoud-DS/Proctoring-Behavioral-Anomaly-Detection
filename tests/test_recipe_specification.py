import numpy as np
import pandas as pd
import pytest

from cheatdetect.data.feature_schema import COUNT_LOG, SIGNED, SOURCE_FEATURES
from cheatdetect.data.preprocessing import FeaturePreprocessor
from cheatdetect.experiments.experiment_1.audit import audit_recipe


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
@pytest.mark.parametrize("scale", [False, True])
def test_frozen_recipe_matches_phase_one_specification(features, recipe, scale):
    expected, _ = audit_recipe(features, recipe, scale)
    actual = FeaturePreprocessor(recipe, scale).fit_transform(features)
    assert list(actual.columns) == list(expected.columns)
    np.testing.assert_allclose(actual, expected)

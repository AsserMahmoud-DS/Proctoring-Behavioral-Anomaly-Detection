import numpy as np
import pandas as pd
import pytest

from cheatdetect.data.features.mouse import extract_mouse_features


def mouse_path(angles):
    radians = np.deg2rad(angles)
    coordinates = np.vstack([
        np.zeros(2), np.cumsum(np.column_stack([np.cos(radians), np.sin(radians)]), axis=0),
    ])
    return pd.DataFrame({
        "Time (seconds)": np.arange(len(coordinates), dtype=float),
        "X Coordinate": coordinates[:, 0],
        "Y Coordinate": coordinates[:, 1],
        "Event Type": "mousemove",
    })


@pytest.mark.parametrize("angles", [(179, -179), (-179, 179)])
def test_boundary_crossing_is_a_small_turn(angles):
    features = extract_mouse_features(mouse_path(angles))
    assert features["mouse_direction_changes"] == 0
    assert features["mouse_sum_of_angles"] == pytest.approx(np.deg2rad(2))
    assert features["mouse_sharp_angles"] == 1
    assert abs(features["mouse_angular_velocity_mean"]) == pytest.approx(np.deg2rad(2))


def test_tiny_boundary_crossing_preserves_sharp_angle_threshold():
    features = extract_mouse_features(mouse_path((179.999, -179.999)))
    assert features["mouse_direction_changes"] == 0
    assert features["mouse_sum_of_angles"] == pytest.approx(np.deg2rad(0.002))
    assert features["mouse_sharp_angles"] == 0


@pytest.mark.parametrize("angles,degrees", [((0, 90), 90), ((0, 180), 180)])
def test_genuine_large_turns_are_retained(angles, degrees):
    features = extract_mouse_features(mouse_path(angles))
    assert features["mouse_direction_changes"] == 1
    assert features["mouse_sum_of_angles"] == pytest.approx(np.deg2rad(degrees))
    assert features["mouse_sharp_angles"] == 1


@pytest.mark.parametrize("angles", [(), (30,), (30, 30)])
def test_insufficient_or_unchanged_directions_have_no_turns(angles):
    features = extract_mouse_features(mouse_path(angles))
    assert features["mouse_direction_changes"] == 0
    assert features["mouse_sum_of_angles"] == pytest.approx(0)
    assert features["mouse_sharp_angles"] == 0

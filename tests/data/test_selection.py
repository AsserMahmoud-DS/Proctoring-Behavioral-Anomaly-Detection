import pandas as pd

from cheatdetect.data import select_features


def test_select_features_drops_constant_and_redundant_aggregation():
    train = pd.DataFrame(
        {
            "mouse_velocity_mean": [0, 1, 2, 3, 4],
            "mouse_acceleration_mean": [0, 3, 6, 9, 12],
            "mouse_velocity_max": [0, 2, 4, 6, 8],
            "constant": [7] * 5,
        }
    )

    kept, dropped = select_features(train)

    assert kept == ["mouse_velocity_mean", "mouse_acceleration_mean"]
    assert dropped == ["constant", "mouse_velocity_max"]

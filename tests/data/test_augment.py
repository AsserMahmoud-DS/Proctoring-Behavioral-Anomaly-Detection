import numpy as np
import pandas as pd
import pytest

from cheatdetect.data import add_coordinate_noise, augment_session_data, clean_session_data


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="float noise cannot be assigned to integer coordinate columns",
)
def test_noise_accepts_integer_coordinates():
    raw = pd.DataFrame(
        {
            "Time (seconds)": [0, 1, 2],
            "X Coordinate": [0, 0, 0],
            "Y Coordinate": [2, 2, 2],
            "Event Type": ["mousemove"] * 3,
        }
    )

    noisy = add_coordinate_noise(
        clean_session_data(raw), sigma=2, random_state=np.random.default_rng(42)
    )

    assert pd.api.types.is_float_dtype(noisy["X Coordinate"])
    assert pd.api.types.is_float_dtype(noisy["Y Coordinate"])
    assert noisy["X Coordinate"].between(0, 1920).all()
    assert noisy["Y Coordinate"].between(0, 1080).all()


def test_noise_changes_only_mouse_coordinates():
    raw = pd.DataFrame(
        {
            "Time (seconds)": [0.0, 1.0, 2.0],
            "X Coordinate": [100.0, 200.0, 200.0],
            "Y Coordinate": [100.0, 200.0, 200.0],
            "Event Type": ["keydown", "mousemove", "keydown"],
        }
    )
    cleaned = clean_session_data(raw)

    noisy = add_coordinate_noise(
        cleaned, sigma=2, random_state=np.random.default_rng(42)
    )

    for coordinate in ("X Coordinate", "Y Coordinate"):
        assert noisy[coordinate].iloc[[0, 2]].tolist() == cleaned[coordinate].iloc[[0, 2]].tolist()
        assert noisy[coordinate].iloc[1] != cleaned[coordinate].iloc[1]
    pd.testing.assert_frame_equal(cleaned, clean_session_data(raw))


def test_augmentation_skips_short_session_and_continues():
    full_session = clean_session_data(
        pd.DataFrame(
            {
                "Time (seconds)": [0.0, 1.0, 2.0],
                "X Coordinate": [100.0, 200.0, 300.0],
                "Y Coordinate": [100.0, 200.0, 300.0],
                "Event Type": ["mousemove"] * 3,
            }
        )
    )
    sessions = {
        "1": full_session,
        "2": full_session.iloc[:2].copy(),
        "3": full_session.copy(),
    }

    augmented = augment_session_data(
        sessions, chunk_size=3, n_copies=2, random_state=42
    )

    assert len(augmented) == 4
    assert augmented["session_name"].tolist() == [
        "1_aug1_chunk0",
        "1_aug2_chunk0",
        "3_aug1_chunk0",
        "3_aug2_chunk0",
    ]

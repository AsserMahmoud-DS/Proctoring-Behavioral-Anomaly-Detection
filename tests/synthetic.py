"""Valid fixed-schema synthetic data for detector tests."""

import numpy as np
import pandas as pd

from cheatdetect.data.feature_schema import COUNTS, SOURCE_FEATURES


def make_source_frame(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Return a valid 25-source-feature frame with finite, in-domain values."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.uniform(0.5, 5.0, size=(n_rows, len(SOURCE_FEATURES))),
        columns=SOURCE_FEATURES,
    )
    for name in COUNTS:
        frame[name] = np.round(frame[name])
    frame["mouse_direction_class"] = rng.integers(0, 9, size=n_rows)
    for name in ("mouse_straightness", "mouse_idle_time_ratio"):
        frame[name] = np.clip(frame[name] / 5.0, 0.0, 1.0)
    return frame


def make_source_sequences(n_samples: int, seq_len: int, seed: int = 0) -> np.ndarray:
    """Return valid raw source sequences ``(n_samples, seq_len, 25)``."""
    frame = make_source_frame(n_samples * seq_len, seed)
    return frame.to_numpy(dtype=np.float32).reshape(
        n_samples, seq_len, len(SOURCE_FEATURES)
    )

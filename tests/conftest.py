"""Shared pytest fixtures for the CheatDetect test suite."""

import numpy as np
import pandas as pd
import pytest


def _raw_session(n_events: int, cheating: bool, seed: int) -> pd.DataFrame:
    """Create a minimal raw session DataFrame with the expected schema."""
    rng = np.random.default_rng(seed)
    rows = []
    x, y, t = 100.0, 100.0, 0.0
    for i in range(n_events):
        event_type = "mousemove" if i % 2 == 0 else "keydown"
        if event_type == "mousemove":
            x = float(np.clip(x + rng.normal(0, 5), 0, 1920))
            y = float(np.clip(y + rng.normal(0, 5), 0, 1080))
        t += float(rng.uniform(0.05, 0.3))
        rows.append(
            {
                "Time (seconds)": t,
                "Event Type": event_type,
                "X Coordinate": x,
                "Y Coordinate": y,
                "Action": "",
                "Is Cheating": "TRUE" if cheating else "FALSE",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def make_clean_session():
    """Factory fixture returning a cleaned session DataFrame."""
    from cheatdetect.data import clean_session_data

    def _make(n_events: int = 40, cheating: bool = False, seed: int = 0):
        return clean_session_data(_raw_session(n_events, cheating, seed))

    return _make


@pytest.fixture(scope="session")
def tiny_study(tmp_path_factory):
    """One shared tiny Experiment 1 study bundle for the runner/reporting tests.

    Feature extraction dominates the build cost, so it runs once per session;
    tests never mutate the prepared bundle.
    """
    from cheatdetect.data import dataset as data
    from study_helpers import build_tiny_study

    base = tmp_path_factory.mktemp("tiny_study")
    original_root = data.ARTIFACT_ROOT
    try:
        yield build_tiny_study(base)
    finally:
        data.ARTIFACT_ROOT = original_root

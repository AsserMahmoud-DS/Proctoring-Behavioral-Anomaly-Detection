import numpy as np
import pandas as pd

from cheatdetect.models import IsolationForestDetector


def make_example_data():
    """Small, repeatable normal training set and three scoring queries."""
    rng = np.random.default_rng(42)
    normal_rows = pd.DataFrame(
        rng.normal(size=(40, 2)), columns=["feature_a", "feature_b"]
    )
    queries = pd.DataFrame(
        {"feature_a": [0.0, 0.5, 5.0], "feature_b": [0.0, -0.5, 5.0]}
    )
    return normal_rows, queries


def test_scoring_rows_finite_and_negative_wrapped():
    detector = IsolationForestDetector(skewed_cols=[])
    normal_rows, queries = make_example_data()

    detector.fit(normal_rows)
    scores = detector.decision_function(queries)

    assert scores.shape == (len(queries),)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, -detector.pipeline.decision_function(queries))

import numpy as np
import pandas as pd

from cheatdetect.models import OCSVMDetector


def test_scoring_rows_finite_and_negative_wrapped():
    rng = np.random.default_rng(42)
    normal_rows = pd.DataFrame(
        rng.normal(size=(40, 2)), columns=["feature_a", "feature_b"]
    )
    queries = pd.DataFrame(
        {"feature_a": [0.0, 0.5, 5.0], "feature_b": [0.0, -0.5, 5.0]}
    )
    detector = OCSVMDetector(skewed_cols=[])

    detector.fit(normal_rows)
    scores = detector.decision_function(queries)

    assert scores.shape == (len(queries),)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, -detector.pipeline.decision_function(queries))

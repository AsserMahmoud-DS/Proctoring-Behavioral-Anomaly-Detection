"""Option A: detectors reuse an injected original-fitted preprocessor."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from cheatdetect.data import FeaturePreprocessor
from cheatdetect.models import IsolationForestDetector, OCSVMDetector

from synthetic import make_source_frame


def _frames():
    original = make_source_frame(40, seed=0)
    augmented = pd.concat([original, make_source_frame(40, seed=1)], ignore_index=True)
    validation = make_source_frame(20, seed=2)
    return original, augmented, validation


def test_isolation_forest_reuses_injected_preprocessor():
    original, augmented, validation = _frames()
    processor = FeaturePreprocessor("log", scale=False).fit(original)
    before = joblib.hash(processor)

    detector = IsolationForestDetector(
        recipe="log", n_estimators=5, max_samples=8, preprocessor=processor
    ).fit(augmented)
    manual = IsolationForest(
        n_estimators=5, max_samples=8, contamination=0.1, random_state=42, n_jobs=-1
    ).fit(processor.transform(augmented))

    assert detector.preprocessor is processor
    assert joblib.hash(processor) == before
    np.testing.assert_allclose(
        detector.decision_function(validation),
        -manual.decision_function(processor.transform(validation)),
    )


def test_ocsvm_reuses_injected_preprocessor():
    original, augmented, validation = _frames()
    processor = FeaturePreprocessor("quantile", scale=True).fit(original)
    before = joblib.hash(processor)

    detector = OCSVMDetector(
        recipe="quantile", nu=0.5, gamma=0.1, preprocessor=processor
    ).fit(augmented)
    manual = OneClassSVM(nu=0.5, gamma=0.1, kernel="rbf").fit(
        processor.transform(augmented)
    )

    assert detector.preprocessor is processor
    assert joblib.hash(processor) == before
    np.testing.assert_allclose(
        detector.decision_function(validation),
        -manual.decision_function(processor.transform(validation)),
    )

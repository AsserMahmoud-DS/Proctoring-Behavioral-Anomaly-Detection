"""One-Class SVM convergence diagnostics are reported, never hidden."""

import logging

import numpy as np
from sklearn.svm import OneClassSVM

from cheatdetect.models import OCSVMDetector

from synthetic import make_source_frame


def test_grid_search_reports_non_convergence(monkeypatch, caplog):
    real_fit = OneClassSVM.fit

    def not_converged(self, X, y=None, sample_weight=None):
        real_fit(self, X, y, sample_weight)
        self.n_iter_ = 7
        self.fit_status_ = 1
        return self

    monkeypatch.setattr(OneClassSVM, "fit", not_converged)

    with caplog.at_level(logging.WARNING):
        best, results = OCSVMDetector.grid_search(
            make_source_frame(20, seed=0),
            make_source_frame(10, seed=1),
            np.array([0, 1] * 5),
            {"nu": [0.5], "gamma": [0.1], "kernel": ["rbf"]},
        )

    assert best is not None
    assert results.iloc[0]["n_iter"] == 7
    assert results.iloc[0]["converged"] == False  # noqa: E712 - pandas bool row
    assert any("did not converge" in record.message for record in caplog.records)


def test_convergence_diagnostics_are_available_after_fit():
    detector = OCSVMDetector(nu=0.5, gamma=0.1).fit(make_source_frame(30, seed=0))

    diagnostics = detector.convergence_diagnostics()
    assert diagnostics["n_iter"] >= 1
    assert isinstance(diagnostics["converged"], bool)

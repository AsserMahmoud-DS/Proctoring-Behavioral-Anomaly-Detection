"""Reporting preserves ranking metrics, prevalence baseline, and threshold behavior."""

import numpy as np
import pytest
from matplotlib.axes import Axes

from cheatdetect.eval.metrics import compare_models, evaluate_model
from cheatdetect.eval.plots import plot_pr_curves


def test_report_includes_prevalence_and_threshold_metrics():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0, 1, 2, 5, 3, 4])
    result = evaluate_model("example", scores, 3.0, labels)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx(7 / 12)
    assert result["pr_auc_baseline"] == pytest.approx(1 / 3)
    assert result["threshold"] == 3.0
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["f1"] == pytest.approx(0.8)
    np.testing.assert_array_equal(result["cm"], [[3, 1], [0, 2]])
    table = compare_models([result])
    assert {"roc_auc", "pr_auc", "pr_auc_baseline", "threshold", "precision", "recall", "f1"} <= set(table.columns)
    assert table.loc["example", "pr_auc_baseline"] == pytest.approx(1 / 3)


def test_pr_plot_draws_prevalence_baseline(tmp_path, monkeypatch):
    observed = []
    draw_line = Axes.axhline

    def capture_line(axis, value, *args, **kwargs):
        observed.append(value)
        return draw_line(axis, value, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", capture_line)
    path = tmp_path / "pr.png"
    labels = np.array([0, 0, 1])
    plot_pr_curves({"example": np.array([0.0, 1.0, 2.0])}, labels, path)
    assert observed == pytest.approx([1 / 3])
    assert path.is_file()

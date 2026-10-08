"""Shared experiment CLI dispatches the study phases."""

import pytest

from cheatdetect.experiments import __main__ as cli
from cheatdetect.experiments.experiment_1 import reporting, runner


@pytest.fixture
def loaded_study(monkeypatch):
    study = object()
    monkeypatch.setattr(runner, "load_study", lambda *args, **kwargs: study)
    return study


def test_invalid_or_missing_phase_exits():
    with pytest.raises(SystemExit):
        cli.main(["--phase", "9"])
    with pytest.raises(SystemExit):
        cli.main([])


def test_phase3_builds_the_bundle(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "build_study", lambda *a, **k: calls.append(k))
    assert cli.main(["--phase", "3"]) == 0
    assert calls == [{}]


def test_phase4_runs_search_on_the_loaded_study(loaded_study, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        runner, "run_search", lambda study, **kwargs: calls.append((study, kwargs))
    )
    assert cli.main(["--phase", "4", "--run-dir", str(tmp_path)]) == 0
    assert calls == [(loaded_study, {"output_dir": tmp_path})]


def test_phase5_runs_seed_repeats(loaded_study, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        runner, "run_seed_repeats", lambda study, **kwargs: calls.append((study, kwargs))
    )
    assert cli.main(["--phase", "5", "--run-dir", str(tmp_path)]) == 0
    assert calls == [(loaded_study, {"run_dir": tmp_path})]


def test_phase6_runs_gated_reporting(loaded_study, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        reporting, "run_phase6", lambda study, **kwargs: calls.append((study, kwargs))
    )
    assert cli.main(["--phase", "6", "--run-dir", str(tmp_path)]) == 0
    assert calls == [(loaded_study, {"run_dir": tmp_path})]

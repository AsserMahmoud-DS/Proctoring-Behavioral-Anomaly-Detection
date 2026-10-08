"""Guard the removal of the legacy dynamic data API."""

import importlib

import pytest

import cheatdetect.data as data

REMOVED_MODULES = (
    "cheatdetect.data.pipeline",
    "cheatdetect.data.selection",
    "cheatdetect.data.sequences",
    "cheatdetect.data.transform",
)

REMOVED_NAMES = {
    "process_session",
    "process_sessions",
    "select_features",
    "feature_summary",
    "detect_zero_variance",
    "find_correlated_pairs",
    "classify_correlation_pair",
    "Log1pSkewed",
    "find_skewed_features",
    "extract_sequences_from_sessions",
    "augment_sequences",
    "align_sequence_features",
    "sequence_length",
}


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_legacy_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_legacy_names_are_not_exported():
    assert REMOVED_NAMES.isdisjoint(data.__all__)
    assert not any(hasattr(data, name) for name in REMOVED_NAMES)

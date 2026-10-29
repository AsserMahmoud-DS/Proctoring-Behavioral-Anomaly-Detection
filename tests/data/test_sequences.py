"""Tests for micro-chunk sequence construction (LSTM data path)."""

import numpy as np
import pandas as pd
import pytest

from cheatdetect.data import (
    align_sequence_features,
    augment_sequences,
    extract_sequences_from_sessions,
    merge_window_switch_events,
    sequence_length,
)

CHUNK, STEP, SUB, SUB_STEP = 20, 10, 5, 5
SEQ_LEN = sequence_length(CHUNK, SUB, SUB_STEP)


def _extract(sessions):
    return extract_sequences_from_sessions(sessions, CHUNK, STEP, SUB, SUB_STEP)


def test_sequence_length_formula():
    assert SEQ_LEN == 4
    assert sequence_length(50, 10, 5) == 9


def test_sequence_shape_and_count(make_clean_session):
    session = make_clean_session(n_events=40, seed=1)
    X, y, names = _extract([session])

    n_chunks = (40 - CHUNK) // STEP + 1
    assert X.shape == (n_chunks, SEQ_LEN, len(names))
    assert y.shape == (n_chunks,)
    # Must be numeric, not an object array (bool + int columns would coerce).
    assert np.issubdtype(X.dtype, np.floating)


def test_labels_follow_parent_chunk(make_clean_session):
    normal = make_clean_session(n_events=40, cheating=False, seed=2)
    cheater = make_clean_session(n_events=40, cheating=True, seed=3)

    _, y_normal, _ = _extract([normal])
    _, y_cheat, _ = _extract([cheater])

    assert y_normal.sum() == 0
    assert y_cheat.sum() == len(y_cheat)


def test_raw_session_order_is_preserved(make_clean_session):
    first = make_clean_session(n_events=40, cheating=False, seed=4)
    second = make_clean_session(n_events=40, cheating=True, seed=5)

    _, y, _ = _extract([first, second])
    n_first = (40 - CHUNK) // STEP + 1

    # All of the first session's chunks come before the second's.
    assert y[:n_first].sum() == 0
    assert y[n_first:].sum() == len(y) - n_first


def test_augmentation_multiplies_counts(make_clean_session):
    session = make_clean_session(n_events=40, seed=6)
    X_orig, _, names = _extract([session])

    X_aug, y_aug, aug_names = augment_sequences(
        [session], CHUNK, STEP, SUB, SUB_STEP, n_copies=2, random_state=0
    )

    assert X_aug.shape[0] == X_orig.shape[0] * 2
    assert X_aug.shape[1] == SEQ_LEN
    assert y_aug.sum() == 0
    assert aug_names == names


def test_alignment_selects_curated_features(make_clean_session):
    session = make_clean_session(n_events=40, seed=7)
    X, _, names = _extract([session])

    merged = merge_window_switch_events(pd.DataFrame(X[0], columns=names))
    keep = [c for c in merged.columns if c != "is_cheating"]

    aligned = align_sequence_features(X, names, keep)
    assert aligned.shape == (X.shape[0], SEQ_LEN, len(keep))


def test_alignment_raises_on_missing_feature(make_clean_session):
    session = make_clean_session(n_events=40, seed=8)
    X, _, names = _extract([session])

    with pytest.raises(ValueError):
        align_sequence_features(X, names, ["definitely_not_a_feature"])

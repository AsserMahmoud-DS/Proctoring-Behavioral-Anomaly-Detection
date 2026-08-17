"""Data loading, cleaning, feature extraction, transformation, and selection.

Public API:
    - ``load_single_session``, ``load_sessions`` — read raw CSVs.
    - ``clean_session_data``, ``clean_features`` — raw event and feature matrix cleaning.
    - ``process_session``, ``process_sessions`` — end-to-end pipeline.
    - ``extract_features_from_session``, ``extract_features_from_chunk`` —
      feature extraction from cleaned data.
    - ``extract_mouse_features``, ``extract_keyboard_features``,
      ``extract_action_features`` — domain-specific extractors.
    - ``merge_window_switch_events`` — merge blur/focus/tab-switch columns.
    - ``Log1pSkewed`` — sklearn transformer for log1p on skewed features.
    - ``find_skewed_features`` — identify columns eligible for log1p.
    - ``add_coordinate_noise``, ``augment_session_data`` —
      Gaussian-noise data augmentation for normal sessions.
    - ``select_features`` — zero-variance + correlation-based feature selection.
    - ``feature_summary`` — keep/drop decision table for EDA.
    - ``detect_zero_variance``, ``find_correlated_pairs``,
      ``classify_correlation_pair`` — selection building blocks.
"""

from .loader import load_single_session, load_sessions
from .cleaning import clean_session_data, clean_features
from .pipeline import process_session, process_sessions
from .features import (
    extract_features_from_session,
    extract_features_from_chunk,
    extract_mouse_features,
    extract_keyboard_features,
    extract_action_features,
)
from .build import merge_window_switch_events
from .transform import Log1pSkewed, find_skewed_features
from .augment import add_coordinate_noise, augment_session_data
from .selection import (
    classify_correlation_pair,
    detect_zero_variance,
    feature_summary,
    find_correlated_pairs,
    select_features,
)

__all__ = [
    "load_single_session",
    "load_sessions",
    "clean_session_data",
    "clean_features",
    "process_session",
    "process_sessions",
    "extract_features_from_session",
    "extract_features_from_chunk",
    "extract_mouse_features",
    "extract_keyboard_features",
    "extract_action_features",
    "merge_window_switch_events",
    "Log1pSkewed",
    "find_skewed_features",
    "add_coordinate_noise",
    "augment_session_data",
    "select_features",
    "feature_summary",
    "detect_zero_variance",
    "find_correlated_pairs",
    "classify_correlation_pair",
]
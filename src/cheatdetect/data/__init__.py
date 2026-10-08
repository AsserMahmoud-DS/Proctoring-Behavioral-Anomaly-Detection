"""Data loading, cleaning, feature extraction, and fixed-schema preparation.

Public API:
    - ``load_single_session``, ``load_sessions`` — read raw CSVs.
    - ``clean_session_data``, ``clean_features`` — raw event and feature matrix
      cleaning.
    - ``extract_features_from_session``, ``extract_features_from_chunk`` —
      window feature extraction; the domain-specific extractors are re-exported
      too.
    - ``merge_window_switch_events`` — merge blur/focus/tab-switch columns.
    - ``FeaturePreprocessor``, ``gamma_center``, ``validate_features`` — frozen
      25-source → 33-encoded preprocessing recipes.
    - ``PairedRepresentation``, ``build_paired_representation``,
      ``extract_paired_features``, ``load_parent_sessions`` — aligned flat and
      micro-sequence views from shared realized parent windows.
    - ``PreparedStudy``, ``prepare_study``, ``save_prepared``, ``load_prepared``,
      ``input_manifest`` — manifest-guarded dataset builder.
    - ``add_coordinate_noise``, ``augment_session_data`` — training-only
      Gaussian coordinate augmentation for normal sessions.
"""

from .loader import load_single_session, load_sessions
from .cleaning import clean_session_data, clean_features
from .features import (
    extract_features_from_session,
    extract_features_from_chunk,
    extract_mouse_features,
    extract_keyboard_features,
    extract_action_features,
)
from .build import merge_window_switch_events
from .preprocessing import FeaturePreprocessor, gamma_center, validate_features
from .paired import (
    PairedRepresentation,
    build_paired_representation,
    extract_paired_features,
    load_parent_sessions,
)
from .dataset import (
    PreparedStudy,
    input_manifest,
    load_prepared,
    prepare_study,
    save_prepared,
)
from .augment import add_coordinate_noise, augment_session_data

__all__ = [
    "load_single_session",
    "load_sessions",
    "clean_session_data",
    "clean_features",
    "extract_features_from_session",
    "extract_features_from_chunk",
    "extract_mouse_features",
    "extract_keyboard_features",
    "extract_action_features",
    "merge_window_switch_events",
    "FeaturePreprocessor",
    "gamma_center",
    "validate_features",
    "PairedRepresentation",
    "build_paired_representation",
    "extract_paired_features",
    "load_parent_sessions",
    "PreparedStudy",
    "prepare_study",
    "save_prepared",
    "load_prepared",
    "input_manifest",
    "add_coordinate_noise",
    "augment_session_data",
]

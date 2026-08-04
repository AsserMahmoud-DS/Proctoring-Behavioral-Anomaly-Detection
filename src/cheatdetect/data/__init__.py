"""Data loading, cleaning, feature extraction, and session processing.

Public API:
    - ``load_single_session``, ``load_sessions`` — read raw CSVs.
    - ``clean_session_data`` — normalize a raw session DataFrame.
    - ``process_session``, ``process_sessions`` — end-to-end pipeline.
    - ``extract_features_from_session``, ``extract_features_from_chunk`` —
      feature extraction from cleaned data.
    - ``extract_mouse_features``, ``extract_keyboard_features``,
      ``extract_action_features`` — domain-specific extractors.
    - ``add_coordinate_noise``, ``augment_session_data`` —
      Gaussian-noise data augmentation for normal sessions.
"""

from .loader import load_single_session, load_sessions
from .cleaning import clean_session_data
from .pipeline import process_session, process_sessions
from .features import (
    extract_features_from_session,
    extract_features_from_chunk,
    extract_mouse_features,
    extract_keyboard_features,
    extract_action_features,
)
from .augment import add_coordinate_noise, augment_session_data

__all__ = [
    "load_single_session",
    "load_sessions",
    "clean_session_data",
    "process_session",
    "process_sessions",
    "extract_features_from_session",
    "extract_features_from_chunk",
    "extract_mouse_features",
    "extract_keyboard_features",
    "extract_action_features",
    "add_coordinate_noise",
    "augment_session_data",
]
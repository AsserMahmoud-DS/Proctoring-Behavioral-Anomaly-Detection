"""Feature extraction subpackage.

Provides modular feature extractors for mouse kinematics, keyboard
typing behavior, and browser action events, as well as the chunk-level
orchestration that combines them into a single feature dict.
"""

from .extract import extract_features_from_session, extract_features_from_chunk
from .mouse import extract_mouse_features
from .keyboard import extract_keyboard_features
from .actions import extract_action_features

__all__ = [
    "extract_features_from_session",
    "extract_features_from_chunk",
    "extract_mouse_features",
    "extract_keyboard_features",
    "extract_action_features",
]
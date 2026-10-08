"""Aligned flat/sequence extraction from shared original or noisy parent windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .augment import add_coordinate_noise
from .build import merge_window_switch_events
from .cleaning import clean_session_data
from .features.extract import extract_features_from_chunk
from .feature_schema import SOURCE_FEATURES
from .preprocessing import validate_features


@dataclass
class PairedRepresentation:
    parent: pd.DataFrame
    sequences: np.ndarray
    labels: np.ndarray
    identities: pd.DataFrame


def _source_row(chunk: pd.DataFrame, cheating_threshold: float) -> tuple[np.ndarray, int]:
    features = merge_window_switch_events(pd.DataFrame([extract_features_from_chunk(chunk, cheating_threshold)]))
    selected = validate_features(features[list(SOURCE_FEATURES)])
    return selected.to_numpy()[0], int(features["is_cheating"].iloc[0])


def extract_paired_features(
    chunk: pd.DataFrame, sub_chunk: int = 10, sub_step: int = 5,
    cheating_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    if sub_chunk < 1 or sub_step < 1 or len(chunk) < sub_chunk:
        raise ValueError("Invalid micro-window geometry")
    parent, label = _source_row(chunk, cheating_threshold)
    micro = np.stack([
        _source_row(chunk.iloc[start:start + sub_chunk], cheating_threshold)[0]
        for start in range(0, len(chunk) - sub_chunk + 1, sub_step)
    ])
    return parent, micro, label


def build_paired_representation(
    sessions: dict[str, pd.DataFrame], synthetic: bool = False,
    n_copies: int = 2, sigma_range: tuple[float, float] = (2.0, 5.0),
    random_state: int = 42, chunk_size: int = 50, step_size: int = 25,
    sub_chunk: int = 10, sub_step: int = 5,
    screen_bounds: tuple[int, int] = (1920, 1080), cheating_threshold: float = 0.5,
) -> PairedRepresentation:
    """Extract both views from each identical realized original/noisy parent window."""
    if min(chunk_size, step_size, sub_chunk, sub_step) < 1 or sub_chunk > chunk_size:
        raise ValueError("Invalid parent/micro-window geometry")
    if min(screen_bounds) <= 0 or not 0 <= cheating_threshold <= 1:
        raise ValueError("Invalid screen bounds or label fraction")
    if n_copies < 1 or not 0 <= sigma_range[0] <= sigma_range[1]:
        raise ValueError("Invalid augmentation settings")
    rng = np.random.default_rng(random_state)
    parent_rows, sequences, labels, identities = [], [], [], []
    for filename, cleaned in sessions.items():
        for start in range(0, len(cleaned) - chunk_size + 1, step_size):
            original = cleaned.iloc[start:start + chunk_size]
            if synthetic and original["Is Cheating"].any():
                raise ValueError("Cannot augment anomalous training events")
            for copy_index in range(1, n_copies + 1) if synthetic else (0,):
                sigma = float(rng.uniform(*sigma_range)) if synthetic else 0.0
                chunk = add_coordinate_noise(original, sigma, screen_bounds, rng) if synthetic else original
                if synthetic:
                    coordinates = chunk.loc[chunk["is_mouse_event"], ["X Coordinate", "Y Coordinate"]]
                    if not np.isfinite(coordinates.to_numpy()).all():
                        raise ValueError("Nonfinite augmented coordinates")
                    if ((coordinates < 0).any().any()
                            or (coordinates["X Coordinate"] > screen_bounds[0]).any()
                            or (coordinates["Y Coordinate"] > screen_bounds[1]).any()):
                        raise ValueError("Augmented coordinates exceed clipping bounds")
                parent, micro, label = extract_paired_features(chunk, sub_chunk, sub_step, cheating_threshold)
                parent_rows.append(parent)
                sequences.append(micro)
                labels.append(label)
                identities.append({
                    "session_file": filename, "parent_start_event": start,
                    "parent_stop_event_exclusive": start + chunk_size,
                    "copy_index": copy_index, "noise_sigma_pixels": sigma,
                })
    identity_frame = pd.DataFrame(identities, columns=[
        "session_file", "parent_start_event", "parent_stop_event_exclusive",
        "copy_index", "noise_sigma_pixels",
    ])
    if identity_frame.duplicated(["session_file", "parent_start_event", "copy_index"]).any():
        raise ValueError("Duplicate parent identity")
    return PairedRepresentation(
        parent=pd.DataFrame(parent_rows, columns=SOURCE_FEATURES, dtype=float),
        sequences=np.stack(sequences) if sequences else np.empty((0, (chunk_size - sub_chunk) // sub_step + 1, len(SOURCE_FEATURES))),
        labels=np.array(labels, dtype=int), identities=identity_frame,
    )


def load_parent_sessions(
    files: list[Path], partition: str = "sessions", normal_only: bool = False,
    chunk_size: int = 50, step_size: int = 25,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load usable sessions and report zero-window exclusions without changing assignments."""
    if chunk_size < 1 or step_size < 1:
        raise ValueError("Invalid parent-window geometry")
    sessions, rows = {}, []
    names = [Path(path).name for path in files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate session filename within partition")
    for path in files:
        path = Path(path)
        raw = pd.read_csv(path)
        cleaned = clean_session_data(raw)
        if normal_only and cleaned["Is Cheating"].any():
            raise ValueError(f"Anomalous events in normal partition: {partition}")
        windows = max(0, (len(cleaned) - chunk_size) // step_size + 1)
        rows.append({
            "partition": partition, "session_file": path.name, "raw_rows": len(raw),
            "cleaned_events": len(cleaned), "parent_windows": windows,
            "included": windows > 0,
            "exclusion_reason": "" if windows else (
                "no_cleaned_events" if cleaned.empty else f"insufficient_events_for_{chunk_size}_event_window"
            ),
        })
        if windows:
            sessions[path.name] = cleaned
    if not sessions:
        details = ", ".join(f"{row['session_file']} ({row['cleaned_events']} cleaned events)" for row in rows)
        raise ValueError(f"No usable parent windows in partition {partition}: {details}")
    return sessions, pd.DataFrame(rows)

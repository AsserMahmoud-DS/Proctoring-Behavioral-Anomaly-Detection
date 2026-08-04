"""Gaussian-noise data augmentation for normal mouse sessions.

Adds jitter to the X/Y coordinates of mouse events at the raw-chunk
level, then re-extracts features so that derived kinematic quantities
(velocity, acceleration, curvature, etc.) naturally reflect the noise.

This is the only safe way to augment a time-series with positional
noise: if we added noise at the session level, duplicated timestamps
would corrupt the temporal structure.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .features.extract import extract_features_from_chunk

logger = logging.getLogger(__name__)


def add_coordinate_noise(
    chunk: pd.DataFrame,
    sigma: float,
    screen_bounds: Tuple[int, int] = (1920, 1080),
    random_state: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to mouse X/Y coordinates in a raw chunk.

    Only rows where ``is_mouse_event`` is True receive noise; keyboard
    and action rows retain their forward-filled positions (which are
    not real cursor data).

    Args:
        chunk: A cleaned DataFrame slice (output of
            ``clean_session_data``), typically 5 events for a single
            chunk window.
        sigma: Standard deviation (in pixels) of the Gaussian noise.
        screen_bounds: ``(width, height)`` used to clip coordinates so
            they stay within the visible viewport.
        random_state: Numpy random generator for reproducibility.

    Returns:
        A **new** DataFrame with noisy coordinates (original is not
        mutated).
    """
    rng = random_state or np.random.default_rng()

    noisy = chunk.copy()
    mouse_mask = noisy["is_mouse_event"].values

    if mouse_mask.any():
        screen_w, screen_h = screen_bounds
        n = int(mouse_mask.sum())

        noisy.loc[mouse_mask, "X Coordinate"] = np.clip(
            noisy.loc[mouse_mask, "X Coordinate"]
            + rng.normal(0, sigma, size=n),
            0, screen_w,
        )
        noisy.loc[mouse_mask, "Y Coordinate"] = np.clip(
            noisy.loc[mouse_mask, "Y Coordinate"]
            + rng.normal(0, sigma, size=n),
            0, screen_h,
        )

    return noisy


def augment_session_data(
    sessions: dict[str, pd.DataFrame],
    chunk_size: int = 5,
    step_size: int = 1,
    n_copies: int = 2,
    sigma_range: Tuple[float, float] = (2.0, 5.0),
    screen_bounds: Tuple[int, int] = (1920, 1080),
    cheating_threshold: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Augment normal sessions by adding Gaussian noise to raw chunks.

    For each session the function:
      1. Slides the same event-count window used by the pipeline
         (``chunk_size`` events, ``step_size`` stride).
      2. For every raw chunk, creates ``n_copies`` augmented copies
         by adding independent Gaussian noise to X/Y coordinates.
         The noise level is drawn from
         ``Uniform(sigma_range[0], sigma_range[1])`` per copy.
      3. Extracts the full 34-feature vector from each noisy chunk
         using ``extract_features_from_chunk``.

    All augmented rows are marked ``is_cheating = False`` and given
    a ``session_name`` of the form
    ``{original_name}_aug{copy_idx}_chunk{chunk_idx}``.

    Args:
        sessions: Mapping of session name → cleaned DataFrame (output
            of ``load_sessions`` + ``clean_session_data``).
        chunk_size: Events per window (must match pipeline).
        step_size: Stride between windows (must match pipeline).
        n_copies: Number of noisy copies per raw chunk.
        sigma_range: ``(min, max)`` for uniform sigma draw per copy.
        screen_bounds: ``(width, height)`` for coordinate clipping.
        cheating_threshold: Passed to ``extract_features_from_chunk``.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with the same column schema as the pipeline output,
        containing **only** augmented rows (no original data).
    """
    rng = np.random.default_rng(random_state)

    all_rows: list[dict] = []

    for session_name, df_session in sessions.items():
        n_events = len(df_session)
        if n_events < chunk_size:
            logger.warning(
                "Session %s too short for chunking (%d < %d), skipping",
                session_name, n_events, chunk_size,
            )
            continue

        n_chunks = (n_events - chunk_size) // step_size + 1

        for chunk_idx, start in enumerate(
            range(0, n_events - chunk_size + 1, step_size)
        ):
            raw_chunk = df_session.iloc[start: start + chunk_size]

            for copy_idx in range(1, n_copies + 1):
                sigma = rng.uniform(sigma_range[0], sigma_range[1])
                noisy_chunk = add_coordinate_noise(
                    raw_chunk,
                    sigma=sigma,
                    screen_bounds=screen_bounds,
                    random_state=rng,
                )
                features = extract_features_from_chunk(
                    noisy_chunk, cheating_threshold
                )
                features["session_name"] = (
                    f"{session_name}_aug{copy_idx}_chunk{chunk_idx}"
                )
                features["is_cheating"] = False
                all_rows.append(features)

    if not all_rows:
        logger.warning("No augmented rows produced")
        return pd.DataFrame()

    df_aug = pd.DataFrame(all_rows)
    logger.info(
        "Augmentation produced %d chunks from %d sessions",
        len(df_aug),
        len(sessions),
    )
    return df_aug
"""End-to-end session processing pipeline.

Orchestrates the full load → clean → extract pipeline for single
sessions or batches, accepting file paths, directories, globs, or
pre-loaded DataFrames as input.
"""

import glob
import logging
import os
from pathlib import Path

import pandas as pd

from .cleaning import clean_session_data
from .features.extract import extract_features_from_session
from .loader import load_single_session

logger = logging.getLogger(__name__)


def process_session(
    source: str | Path | pd.DataFrame,
    session_name: str = "",
    chunk_size: int = 5,
    step_size: int = 1,
    cheating_threshold: float = 0.5,
) -> pd.DataFrame:
    """Process a single session through the full pipeline.

    Accepts either a file path (string or Path) or a pre-loaded
    DataFrame. The pipeline steps are:
        1. Load (if a path is given) → 2. Clean → 3. Extract features

    Args:
        source: CSV file path or an already-loaded DataFrame.
        session_name: Label attached to every extracted chunk. When
            source is a path, defaults to the file stem.
        chunk_size: Number of consecutive events per feature window.
        step_size: Slide between windows (1 = maximum overlap).
        cheating_threshold: Fraction of cheating events in a chunk
            required to label the chunk as cheating.

    Returns:
        DataFrame of extracted features (one row per chunk), or an
        empty DataFrame if processing fails.
    """
    # Resolve source to a raw DataFrame
    if isinstance(source, pd.DataFrame):
        df_raw = source
        if not session_name:
            session_name = "unnamed_session"
    else:
        source = Path(source)
        session_name = session_name or source.stem
        df_raw = load_single_session(source)
        if df_raw.empty:
            return pd.DataFrame()

    df_clean = clean_session_data(df_raw)
    if df_clean.empty:
        logger.warning("No data after cleaning for session: %s", session_name)
        return pd.DataFrame()

    features_df = extract_features_from_session(
        df_clean,
        chunk_size=chunk_size,
        step_size=step_size,
        cheating_threshold=cheating_threshold,
        session_name=session_name,
    )
    return features_df


def process_sessions(
    sources: str | Path | list[str | Path] | dict[str, pd.DataFrame],
    chunk_size: int = 5,
    step_size: int = 1,
    cheating_threshold: float = 0.5,
) -> pd.DataFrame:
    """Process multiple sessions and concatenate the results.

    Flexible input: pass a directory path, a glob pattern, a list of
    file paths, or a dict of {session_name: DataFrame}.

    Args:
        sources: One of:
            - A directory path (loads all *.csv inside).
            - A glob pattern (e.g. "data/raw/mixed/*.csv").
            - A single file path.
            - A list of file paths.
            - A dict mapping session names to pre-loaded DataFrames.
        chunk_size: Events per feature window.
        step_size: Slide between windows.
        cheating_threshold: Cheating-event fraction threshold per chunk.

    Returns:
        A single DataFrame with all sessions concatenated, or an empty
        DataFrame if no sessions processed successfully.
    """
    # Normalize inputs into a uniform list of (session_name, source) tuples
    sessions_to_process: list[tuple[str, pd.DataFrame | Path | str]] = []

    if isinstance(sources, dict):
        for name, df in sources.items():
            sessions_to_process.append((name, df))
    elif isinstance(sources, (str, Path)):
        source_str = str(sources)
        if "*" in source_str or "?" in source_str:
            file_paths = glob.glob(source_str)
        elif os.path.isdir(source_str):
            file_paths = glob.glob(os.path.join(source_str, "*.csv"))
        else:
            file_paths = [source_str]
        sessions_to_process = [(Path(fp).stem, fp) for fp in file_paths]
    elif isinstance(sources, list):
        for source in sources:
            source = Path(source)
            sessions_to_process.append((source.stem, str(source)))
    else:
        raise TypeError(f"Unsupported sources type: {type(sources)}")

    logger.info("Processing %d sessions", len(sessions_to_process))

    all_features: list[pd.DataFrame] = []
    successful = 0

    for session_name, source in sessions_to_process:
        result = process_session(
            source,
            session_name=session_name,
            chunk_size=chunk_size,
            step_size=step_size,
            cheating_threshold=cheating_threshold,
        )
        if not result.empty:
            all_features.append(result)
            successful += 1
        else:
            logger.warning("Failed to process session: %s", session_name)

    if not all_features:
        logger.error("No sessions processed successfully")
        return pd.DataFrame()

    combined = pd.concat(all_features, ignore_index=True)
    total = len(combined)
    cheating = int(combined["is_cheating"].sum()) if "is_cheating" in combined.columns else 0
    logger.info(
        "Processed %d/%d sessions | %d chunks | %d cheating | %d normal",
        successful,
        len(sessions_to_process),
        total,
        cheating,
        total - cheating,
    )
    return combined
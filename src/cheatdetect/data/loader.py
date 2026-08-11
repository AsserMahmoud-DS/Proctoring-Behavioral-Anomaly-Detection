"""Data loading utilities for raw session CSV files.

Provides functions to read individual sessions or entire directories
of CSV files into pandas DataFrames for downstream processing.
"""

import glob
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_single_session(file_path: str | Path) -> pd.DataFrame:
    """Load a single session CSV file into a DataFrame.

    Args:
        file_path: Path to the session CSV file.

    Returns:
        DataFrame with the raw session data, or an empty DataFrame
        if the file cannot be read.
    """
    file_path = Path(file_path)
    logger.info("Loading session: %s", file_path.name)
    try:
        df = pd.read_csv(file_path)
        logger.info("Loaded %d rows from %s", len(df), file_path.name)
        return df
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return pd.DataFrame()


def load_sessions(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load all CSV session files from a directory.

    Each file is loaded into a DataFrame keyed by its filename.

    Args:
        directory: Path to the directory containing CSV files.

    Returns:
        Dictionary mapping filenames to DataFrames. Empty dict if
        no CSV files are found.
    """
    directory = Path(directory)
    search_path = str(directory / "*.csv")
    csv_files = glob.glob(search_path)

    if not csv_files:
        logger.warning("No CSV files found in: %s", directory)
        return {}

    sessions: dict[str, pd.DataFrame] = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        sessions[filename] = df

    logger.info("Loaded %d sessions from %s", len(sessions), directory)
    return sessions
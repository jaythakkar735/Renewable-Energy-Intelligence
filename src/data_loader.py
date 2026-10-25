"""Data loader for Open Power System Data (OPSD) Time Series."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import load_config, get_project_root
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_raw_country_data(
    config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Loads and extracts the selected country time series from the raw OPSD CSV.

    Args:
        config: Loaded config dictionary. If None, loads from config/config.yaml.
        use_cache: If True, checks for an existing clean parquet file in data/processed/.
        force_reload: If True, ignores cached parquet and parses raw CSV.

    Returns:
        DataFrame indexed by UTC timestamp with clean column names and float32 data.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    country_code = config["data"]["selected_country"]
    cache_path = processed_dir / f"{country_code}_raw_extracted.parquet"

    if use_cache and not force_reload and cache_path.exists():
        logger.info(f"Loading cached extracted data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    raw_dir = Path(config["paths"]["raw_data_dir"])
    raw_csv = raw_dir / config["paths"]["raw_csv_filename"]
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw OPSD CSV file not found at: {raw_csv}. "
            "Please verify path in config/config.yaml."
        )

    col_map: Dict[str, str] = config["data"]["columns_mapping"]
    cols_to_read = list(col_map.keys())

    logger.info(f"Reading {len(cols_to_read)} columns from {raw_csv.name}...")

    # Define dtypes for memory efficiency
    dtype_spec = {c: np.float32 for c in cols_to_read if c != "utc_timestamp"}

    df = pd.read_csv(
        raw_csv,
        usecols=cols_to_read,
        dtype=dtype_spec,
    )

    # Parse UTC timestamp
    logger.info("Parsing UTC timestamps...")
    df["timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.drop(columns=["utc_timestamp"])

    # Rename columns to standard names
    rename_dict = {orig: target for orig, target in col_map.items() if orig != "utc_timestamp"}
    df = df.rename(columns=rename_dict)

    # Sort by timestamp and set as index
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    logger.info(
        f"Extracted {len(df):,} hourly rows from {df.index.min()} to {df.index.max()} "
        f"for country {country_code}."
    )

    # Save to parquet cache
    df.to_parquet(cache_path, index=True)
    logger.info(f"Saved extracted data to {cache_path}")

    return df

"""Preprocessing and data quality validation for OPSD time series."""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import load_config, get_project_root
from src.utils import setup_logger

logger = setup_logger(__name__)


def validate_and_clean_time_series(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validates temporal regularity, handles missing values, and applies physical bounds.

    Args:
        df: Raw extracted DataFrame indexed by UTC DatetimeIndex.
        config: Loaded config dictionary.

    Returns:
        Tuple of (clean_df, audit_metrics_dict).
    """
    if config is None:
        config = load_config()

    logger.info("Starting data preprocessing and quality audit...")
    audit = {}
    audit["raw_total_rows"] = int(len(df))
    audit["raw_start"] = str(df.index.min())
    audit["raw_end"] = str(df.index.max())

    # 1. Enforce regular hourly frequency
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pd.DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware (UTC).")

    # Check for duplicate timestamps
    dup_count = int(df.index.duplicated().sum())
    audit["duplicate_timestamps"] = dup_count
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate timestamps. Dropping duplicates.")
        df = df[~df.index.duplicated(keep="first")]

    # Reindex to strict complete hourly grid
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1h",
        tz="UTC",
    )
    missing_intervals = len(full_index) - len(df)
    audit["missing_timestamps_on_grid"] = missing_intervals
    if missing_intervals > 0:
        logger.warning(f"Found {missing_intervals} missing hourly intervals. Reindexing.")
        df = df.reindex(full_index)

    # 2. Record raw missing values per column
    raw_missing = df.isna().sum().to_dict()
    audit["raw_missing_per_column"] = {k: int(v) for k, v in raw_missing.items()}

    # 3. Clean and impute
    # Solar generation: physically bounded >= 0
    if "solar_generation_mw" in df.columns:
        # Clip negatives if any
        neg_count = int((df["solar_generation_mw"] < 0).sum())
        audit["solar_negative_count"] = neg_count
        df["solar_generation_mw"] = df["solar_generation_mw"].clip(lower=0.0)

    # Impute isolated short gaps (forward fill up to 3 hours, then backward fill leading boundary)
    # Note: ENTSO-E starting boundary hours (first 1-2 hours) get backfilled from first valid hour
    imputed_df = df.copy()
    imputed_df = imputed_df.ffill(limit=6).bfill(limit=6)

    # If any remaining NaNs exist, linear interpolation for remaining small internal gaps
    remaining_nans = imputed_df.isna().sum().to_dict()
    audit["remaining_nans_after_fill"] = {k: int(v) for k, v in remaining_nans.items()}
    if any(v > 0 for v in remaining_nans.values()):
        logger.info("Applying linear interpolation for remaining isolated gaps...")
        imputed_df = imputed_df.interpolate(method="time").bfill().ffill()

    # Cast to float32
    for col in imputed_df.columns:
        imputed_df[col] = imputed_df[col].astype(np.float32)

    audit["clean_total_rows"] = int(len(imputed_df))
    audit["clean_start"] = str(imputed_df.index.min())
    audit["clean_end"] = str(imputed_df.index.max())

    logger.info(
        f"Preprocessing complete. Clean dataset spans {audit['clean_start']} "
        f"to {audit['clean_end']} ({len(imputed_df):,} rows)."
    )

    return imputed_df, audit


def get_clean_data(
    config: Optional[Dict[str, Any]] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Convenience function returning the clean preprocessed DataFrame, cached as parquet."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    country_code = config["data"]["selected_country"]
    clean_path = processed_dir / f"{country_code}_clean.parquet"

    if not force_recompute and clean_path.exists():
        logger.info(f"Loading cached clean data from {clean_path}")
        return pd.read_parquet(clean_path)

    from src.data_loader import load_raw_country_data

    raw_df = load_raw_country_data(config=config)
    clean_df, audit = validate_and_clean_time_series(raw_df, config=config)

    clean_df.to_parquet(clean_path, index=True)
    logger.info(f"Saved clean data to {clean_path}")

    # Also save audit report to reports/
    reports_dir = root / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_file = reports_dir / f"{country_code}_data_audit.json"
    import json
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Saved audit summary to {audit_file}")

    return clean_df

"""Preprocessing, telemetry freeze detection, and quality audit for OPSD time series."""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import pandas as pd
import numpy as np

from src.config import load_config, get_project_root
from src.utils import setup_logger

logger = setup_logger(__name__)


def detect_telemetry_gaps(series: pd.Series, min_run_hours: int = 3) -> List[Dict[str, Any]]:
    """Detects telemetry freezes (consecutive identical values >= min_run_hours during daylight or non-zero night).

    Returns:
        List of dicts describing each gap: start, end, duration_hours, stuck_value_mw.
    """
    gaps = []
    cur_start = None
    cur_val = None
    count = 0
    prev_ts = None

    for ts, val in series.items():
        if pd.isna(val):
            continue
        # Non-zero stuck value check (> 1.0 MW)
        if cur_val is not None and val == cur_val and val > 1.0:
            count += 1
        else:
            if count >= min_run_hours and cur_start is not None and prev_ts is not None:
                gaps.append({
                    "start": str(cur_start),
                    "end": str(prev_ts),
                    "hours": int(count),
                    "stuck_value_mw": round(float(cur_val), 3),
                })
            cur_start = ts
            cur_val = val
            count = 1
        prev_ts = ts

    if count >= min_run_hours and cur_start is not None and prev_ts is not None:
        gaps.append({
            "start": str(cur_start),
            "end": str(prev_ts),
            "hours": int(count),
            "stuck_value_mw": round(float(cur_val), 3),
        })

    return gaps


def validate_and_clean_time_series(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validates temporal regularity, flags telemetry gaps, and applies physical bounds.

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

    # Record raw missing values per column
    raw_missing = df.isna().sum().to_dict()
    audit["raw_missing_per_column"] = {k: int(v) for k, v in raw_missing.items()}

    # Solar generation: clip negatives
    if "solar_generation_mw" in df.columns:
        neg_count = int((df["solar_generation_mw"] < 0).sum())
        audit["solar_negative_count"] = neg_count
        df["solar_generation_mw"] = df["solar_generation_mw"].clip(lower=0.0)

    # Telemetry gap detection on raw solar generation
    raw_solar = df["solar_generation_mw"].copy()
    telemetry_gaps = detect_telemetry_gaps(raw_solar, min_run_hours=3)
    audit["telemetry_gaps_count"] = len(telemetry_gaps)
    audit["telemetry_gaps"] = telemetry_gaps

    # Create boolean telemetry gap mask
    gap_mask = pd.Series(False, index=df.index)
    for g in telemetry_gaps:
        start_ts = pd.to_datetime(g["start"])
        end_ts = pd.to_datetime(g["end"])
        gap_mask.loc[start_ts:end_ts] = True

    df["is_telemetry_gap"] = gap_mask

    # Impute isolated short missing values (not the telemetry freezes)
    imputed_df = df.copy()
    imputed_df = imputed_df.ffill(limit=6).bfill(limit=6)

    remaining_nans = imputed_df.isna().sum().to_dict()
    audit["remaining_nans_after_fill"] = {k: int(v) for k, v in remaining_nans.items()}
    if any(v > 0 for v in remaining_nans.values()):
        logger.info("Applying time-interpolation for remaining isolated boundary gaps...")
        imputed_df = imputed_df.interpolate(method="time").bfill().ffill()

    # Cast numerical columns to float32
    for col in imputed_df.columns:
        if col != "is_telemetry_gap":
            imputed_df[col] = imputed_df[col].astype(np.float32)

    # Compute trailing 365-day historical capacity proxy available at issue time
    # (strictly lagged by 48h to prevent lookahead)
    imputed_df["capacity_proxy_mw"] = (
        imputed_df["solar_generation_mw"]
        .shift(48)
        .rolling(24 * 365, min_periods=720)
        .max()
        .bfill()
        .astype(np.float32)
    )

    audit["clean_total_rows"] = int(len(imputed_df))
    audit["clean_start"] = str(imputed_df.index.min())
    audit["clean_end"] = str(imputed_df.index.max())

    logger.info(
        f"Preprocessing complete. Clean dataset spans {audit['clean_start']} "
        f"to {audit['clean_end']} ({len(imputed_df):,} rows). "
        f"Detected {len(telemetry_gaps)} flat telemetry segments."
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
        return pd.read_parquet(clean_path)

    from src.data_loader import load_raw_country_data

    raw_df = load_raw_country_data(config=config)
    clean_df, audit = validate_and_clean_time_series(raw_df, config=config)

    clean_df.to_parquet(clean_path, index=True)
    logger.info(f"Saved clean data to {clean_path}")

    # Save audit report and gap report to reports/
    reports_dir = root / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_file = reports_dir / f"{country_code}_data_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    gaps_file = reports_dir / "telemetry_gaps.json"
    with open(gaps_file, "w", encoding="utf-8") as f:
        json.dump(audit["telemetry_gaps"], f, indent=2)

    return clean_df

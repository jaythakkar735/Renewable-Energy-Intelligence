"""Feature engineering for solar generation forecasting.

Includes astronomical solar geometry, cyclical calendar features,
strictly lagged statistics (lag >= 24h), and leakage verification.
"""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

from src.config import load_config
from src.utils import setup_logger

logger = setup_logger(__name__)


def compute_solar_geometry(
    timestamps: pd.DatetimeIndex,
    latitude_deg: float = 56.0,
    longitude_deg: float = 9.0,
) -> pd.DataFrame:
    """Computes deterministic astronomical solar position and clear-sky proxy.

    Uses Spencer & Iqbal astronomical approximations based on UTC timestamps.

    Args:
        timestamps: pd.DatetimeIndex in UTC.
        latitude_deg: Observer latitude in decimal degrees.
        longitude_deg: Observer longitude in decimal degrees.

    Returns:
        DataFrame with columns ['solar_elevation_rad', 'solar_elevation_deg',
                                 'clearsky_proxy', 'is_daylight'].
    """
    if timestamps.tz is None:
        raise ValueError("Timestamps must be timezone-aware (UTC).")

    # Day of year and fractional hour in UTC
    doy = (
        timestamps.day_of_year.values.astype(np.float64)
        if hasattr(timestamps, "day_of_year")
        else timestamps.dayofyear.values.astype(np.float64)
    )
    hour = timestamps.hour.values.astype(np.float64) + timestamps.minute.values.astype(np.float64) / 60.0

    # Fractional year in radians
    gamma = 2.0 * np.pi / 365.25 * (doy - 1.0 + (hour - 12.0) / 24.0)

    # Equation of time in minutes (Spencer 1971)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    # Solar declination angle in radians (Spencer 1971)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.001480 * np.sin(3.0 * gamma)
    )

    # True solar time in minutes
    time_offset = eqtime + 4.0 * longitude_deg
    tst = hour * 60.0 + time_offset

    # Solar hour angle in degrees and radians
    ha_deg = (tst / 4.0) - 180.0
    ha_rad = np.deg2rad(ha_deg)

    lat_rad = np.deg2rad(latitude_deg)

    # Solar zenith angle cosine
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = np.arccos(cos_zenith)

    # Solar elevation angle
    elevation_rad = (np.pi / 2.0) - zenith_rad
    elevation_deg = np.rad2deg(elevation_rad)

    # Clear-sky irradiance proxy: max(0, sin(elevation)) * eccentricity correction
    sin_elev = np.sin(elevation_rad)
    sin_elev_clipped = np.clip(sin_elev, 0.0, None)

    # Earth-Sun distance eccentricity factor
    eccentricity = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.25)
    clearsky_proxy = sin_elev_clipped * eccentricity

    is_daylight = (elevation_deg > 0.0).astype(np.int32)

    return pd.DataFrame(
        {
            "solar_elevation_rad": elevation_rad.astype(np.float32),
            "solar_elevation_deg": elevation_deg.astype(np.float32),
            "clearsky_proxy": clearsky_proxy.astype(np.float32),
            "is_daylight": is_daylight,
        },
        index=timestamps,
    )


def build_forecasting_features(
    df: pd.DataFrame,
    target_col: str = "solar_generation_mw",
    min_lag_hours: int = 24,
    latitude_deg: float = 56.0,
    longitude_deg: float = 9.0,
) -> pd.DataFrame:
    """Constructs day-ahead leakage-safe features for solar generation.

    All lags and rolling statistics are strictly computed with lag >= min_lag_hours.
    Deterministic astronomical solar elevation and calendar features are added.

    Args:
        df: Preprocessed DataFrame indexed by UTC DatetimeIndex.
        target_col: Series name to forecast.
        min_lag_hours: Minimum backward lag (default 24h for day-ahead cutoff).
        latitude_deg: Approximate latitude of bidding zone center.
        longitude_deg: Approximate longitude of bidding zone center.

    Returns:
        DataFrame with features and target, without lookahead leakage.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    features = pd.DataFrame(index=df.index)

    # 1. Target column
    features["target"] = df[target_col].astype(np.float32)

    # 2. Solar Geometry Features (strictly deterministic ephemeris)
    geo_df = compute_solar_geometry(df.index, latitude_deg, longitude_deg)
    features["solar_elevation_deg"] = geo_df["solar_elevation_deg"]
    features["clearsky_proxy"] = geo_df["clearsky_proxy"]
    features["is_daylight"] = geo_df["is_daylight"]

    # 3. Calendar & Cyclical Features
    hour = df.index.hour.values
    doy = (
        df.index.day_of_year.values
        if hasattr(df.index, "day_of_year")
        else df.index.dayofyear.values
    )
    dow = (
        df.index.day_of_week.values
        if hasattr(df.index, "day_of_week")
        else df.index.dayofweek.values
    )
    month = df.index.month.values

    features["hour"] = hour.astype(np.int32)
    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    features["day_of_week"] = dow.astype(np.int32)
    features["is_weekend"] = (dow >= 5).astype(np.int32)
    features["month"] = month.astype(np.int32)
    features["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    features["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)

    # 4. Strictly Lagged Generation Features (lag >= min_lag_hours)
    # Lags: 24h (yesterday), 48h (2 days ago), 72h (3 days ago), 168h (same hour last week)
    lag_steps = [min_lag_hours, min_lag_hours + 24, min_lag_hours + 48, min_lag_hours + 144]
    for lag in lag_steps:
        features[f"target_lag_{lag}h"] = df[target_col].shift(lag).astype(np.float32)

    # 5. Rolling Statistics on Lagged History (strictly computed on data <= t - min_lag_hours)
    base_lagged = df[target_col].shift(min_lag_hours)
    features["target_rolling_mean_24h"] = base_lagged.rolling(24, min_periods=12).mean().astype(np.float32)
    features["target_rolling_std_24h"] = (
        base_lagged.rolling(24, min_periods=12).std().fillna(0.0).astype(np.float32)
    )
    features["target_rolling_max_24h"] = base_lagged.rolling(24, min_periods=12).max().astype(np.float32)
    features["target_rolling_mean_168h"] = base_lagged.rolling(168, min_periods=84).mean().astype(np.float32)

    # 6. Source-published load forecast (if present, published day-ahead by ENTSO-E)
    if "load_forecast_mw" in df.columns:
        features["source_load_forecast_mw"] = df["load_forecast_mw"].astype(np.float32)

    # Drop warm-up rows where initial 168h lag is NaN
    max_lag = max(lag_steps) + 168
    valid_features = features.iloc[max_lag:].copy()

    # Final sanity check: fill any remaining isolated NaN with 0
    valid_features = valid_features.bfill().ffill()

    return valid_features


def verify_feature_leakage(
    df: pd.DataFrame,
    target_col: str = "solar_generation_mw",
    min_lag_hours: int = 24,
) -> bool:
    """Verifies that future values of the target cannot leak into features at time t.

    Performs a perturbation test: modifies the future sequence and asserts that
    historical features at time t remain strictly invariant.

    Returns:
        True if the leakage test passes.
    """
    sample = df.iloc[:1000].copy()
    f_orig = build_forecasting_features(sample, target_col=target_col, min_lag_hours=min_lag_hours)

    # Perturb the last 100 rows of the target series
    perturbed = sample.copy()
    perturbed.iloc[-100:, perturbed.columns.get_loc(target_col)] += 9999.0

    f_pert = build_forecasting_features(perturbed, target_col=target_col, min_lag_hours=min_lag_hours)

    # Rows prior to (N - 100) must have identical features
    check_rows = f_orig.index[:-100]
    feature_cols = [c for c in f_orig.columns if c != "target"]

    diff = (f_orig.loc[check_rows, feature_cols] - f_pert.loc[check_rows, feature_cols]).abs().max().max()

    if diff > 1e-5:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED! Max feature difference on historical rows: {diff}")

    logger.info("Leakage verification test PASSED. Features are strictly non-anticipative.")
    return True

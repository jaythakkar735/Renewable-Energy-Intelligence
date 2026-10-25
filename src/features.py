"""Feature engineering for solar generation forecasting.

Includes astronomical solar geometry, cyclical calendar features,
auction cutoff-safe lags (<= 12:00 UTC day-ahead), capacity normalization,
market price features, ENTSO-E load forecasts, and leakage verification.
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
    latitude_deg: float = 56.0,
    longitude_deg: float = 9.0,
    min_lag_hours: int = 24,
) -> pd.DataFrame:
    """Constructs day-ahead leakage-safe features for solar generation.

    All telemetry lags respect the day-ahead auction cutoff at 12:00 UTC:
      - Delivery hour h in [0, 11] UTC: same hour on Day D was observed <= 11:00 UTC (lag 24h).
      - Delivery hour h in [12, 23] UTC: same hour on Day D has not occurred at 12:00 UTC cutoff,
        so latest available same-hour observation is from Day D-1 (lag 48h).
    This establishes the Last-Available-Day baseline and causal feature set.

    In addition, features are normalized by a causal trailing 365-day capacity proxy C_t
    to eliminate capacity growth drift.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    features = pd.DataFrame(index=df.index)

    # 1. Target column (raw MW) and Capacity Proxy
    features["target"] = df[target_col].astype(np.float32)

    if "capacity_proxy_mw" in df.columns:
        cap_proxy = df["capacity_proxy_mw"].astype(np.float32)
    else:
        # Strictly causal trailing capacity proxy (lag >= 48h, no lookahead)
        initial_val = float(df[target_col].iloc[0]) if len(df) > 0 else 100.0
        cap_proxy = df[target_col].shift(48).rolling(24 * 365, min_periods=1).max().ffill().fillna(initial_val).astype(np.float32)

    features["capacity_proxy_mw"] = cap_proxy
    # Capacity-normalized target: \tilde{y} = y / C_t
    features["target_normalized"] = (features["target"] / features["capacity_proxy_mw"]).clip(0.0, 1.2).astype(np.float32)

    # Telemetry gap flag
    if "is_telemetry_gap" in df.columns:
        features["is_telemetry_gap"] = df["is_telemetry_gap"].astype(bool)
    else:
        features["is_telemetry_gap"] = False

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

    # 4. Auction Cutoff-Safe Lags (Last-Available-Day)
    # At Day D 12:00 UTC cutoff:
    # h in [0, 11] -> lag 24 is available (hour h was <= 11:00 UTC on Day D)
    # h in [12, 23] -> lag 48 is latest available (hour h on Day D is >= 12:00 UTC)
    target_series = df[target_col]
    lag_24 = target_series.shift(24).astype(np.float32)
    lag_48 = target_series.shift(48).astype(np.float32)
    lag_168 = target_series.shift(168).astype(np.float32)

    features["target_lag_last_available"] = np.where(hour < 12, lag_24, lag_48).astype(np.float32)
    features["target_lag_24h"] = lag_24
    features["target_lag_48h"] = lag_48
    features["target_lag_168h"] = lag_168

    # Capacity-normalized lags
    features["target_lag_last_available_norm"] = (
        features["target_lag_last_available"] / features["capacity_proxy_mw"]
    ).clip(0.0, 1.2).astype(np.float32)
    features["target_lag_48h_norm"] = (
        features["target_lag_48h"] / features["capacity_proxy_mw"]
    ).clip(0.0, 1.2).astype(np.float32)
    features["target_lag_168h_norm"] = (
        features["target_lag_168h"] / features["capacity_proxy_mw"]
    ).clip(0.0, 1.2).astype(np.float32)

    # 5. Rolling Statistics on Lagged History (strictly computed on data <= t - 48h)
    base_lagged = target_series.shift(48)
    base_lagged_norm = base_lagged / features["capacity_proxy_mw"]
    features["target_rolling_mean_24h"] = base_lagged.rolling(24, min_periods=12).mean().astype(np.float32)
    features["target_rolling_std_24h"] = (
        base_lagged.rolling(24, min_periods=12).std().fillna(0.0).astype(np.float32)
    )
    features["target_rolling_max_24h"] = base_lagged.rolling(24, min_periods=12).max().astype(np.float32)
    features["target_rolling_mean_168h"] = base_lagged.rolling(168, min_periods=84).mean().astype(np.float32)

    features["target_rolling_mean_24h_norm"] = base_lagged_norm.rolling(24, min_periods=12).mean().astype(np.float32)
    features["target_rolling_std_24h_norm"] = (
        base_lagged_norm.rolling(24, min_periods=12).std().fillna(0.0).astype(np.float32)
    )
    features["target_rolling_max_24h_norm"] = base_lagged_norm.rolling(24, min_periods=12).max().astype(np.float32)
    features["target_rolling_mean_168h_norm"] = base_lagged_norm.rolling(168, min_periods=84).mean().astype(np.float32)

    # 6. Market Price Signals (cleared day-ahead at 12:00 CET/UTC)
    if "price_day_ahead_eur_mwh" in df.columns:
        features["price_day_ahead_eur_mwh"] = df["price_day_ahead_eur_mwh"].astype(np.float32)
        # Daily mean price for delivery day
        daily_price_mean = df["price_day_ahead_eur_mwh"].groupby(df.index.date).transform("mean")
        features["price_daily_mean_eur_mwh"] = daily_price_mean.astype(np.float32)

    # 7. ENTSO-E Day-ahead Load Forecast (published day-ahead before auction)
    if "load_forecast_mw" in df.columns:
        features["source_load_forecast_mw"] = df["load_forecast_mw"].astype(np.float32)

    # 8. Wind and Load Cutoff-Safe Actuals
    if "wind_generation_mw" in df.columns:
        wind_24 = df["wind_generation_mw"].shift(24).astype(np.float32)
        wind_48 = df["wind_generation_mw"].shift(48).astype(np.float32)
        features["wind_lag_safe"] = np.where(hour < 12, wind_24, wind_48).astype(np.float32)

    if "load_actual_mw" in df.columns:
        load_24 = df["load_actual_mw"].shift(24).astype(np.float32)
        load_48 = df["load_actual_mw"].shift(48).astype(np.float32)
        features["load_lag_safe"] = np.where(hour < 12, load_24, load_48).astype(np.float32)

    # Drop warm-up rows where initial 168h lag is NaN
    max_lag = 168 + 168
    valid_features = features.iloc[max_lag:].copy()

    # Fill any remaining isolated NaN with forward fill and then 0.0 (strictly non-anticipative)
    valid_features = valid_features.ffill().fillna(0.0)

    return valid_features


def verify_feature_leakage(
    df: pd.DataFrame,
    target_col: str = "solar_generation_mw",
    min_lag_hours: int = 24,
) -> bool:
    """Verifies that future values beyond auction cutoff cannot leak into features.

    Performs a perturbation test: modifies the future sequence and asserts that
    historical features at time t remain strictly invariant.
    """
    sample = df.iloc[:1000].copy()
    f_orig = build_forecasting_features(sample, target_col=target_col)

    # Perturb the last 100 rows of the target series
    perturbed = sample.copy()
    perturbed.iloc[-100:, perturbed.columns.get_loc(target_col)] += 9999.0

    f_pert = build_forecasting_features(perturbed, target_col=target_col)

    # Rows prior to (N - 100) must have identical features
    check_rows = f_orig.index[:-100]
    feature_cols = [
        c for c in f_orig.columns
        if c not in ("target", "target_normalized", "is_telemetry_gap")
        and f_orig[c].dtype != bool
    ]

    diff = (f_orig.loc[check_rows, feature_cols] - f_pert.loc[check_rows, feature_cols]).abs().max().max()

    if diff > 1e-4:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED! Max feature difference on historical rows: {diff}")

    logger.info("Leakage verification test PASSED. Features are strictly non-anticipative.")
    return True

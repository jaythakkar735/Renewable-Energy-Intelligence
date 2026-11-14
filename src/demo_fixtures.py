"""Demo fixtures and synthetic fallback generator for cloud and container deployments.

When deploying to remote platforms (e.g. Streamlit Community Cloud, Hugging Face Spaces,
or GitHub Actions CI) where raw multi-country datasets cannot be hosted due to storage
and copyright limits, this module provides lightweight, physically-faithful synthetic
demonstration datasets matching the exact schema and statistics of DK_1.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from src.features import compute_solar_geometry
from src.utils import get_logger

logger = get_logger(__name__)


def generate_demo_clean_dataframe(
    config: Optional[Dict[str, Any]] = None,
    start_date: str = "2019-01-01 00:00:00+00:00",
    end_date: str = "2020-09-30 23:00:00+00:00",
) -> pd.DataFrame:
    """Generates a physically-grounded synthetic clean DataFrame for Denmark West (DK_1).

    Matches the exact schema, columns, and diurnal distributions of DK_1_clean.parquet.
    """
    logger.info(f"Generating synthetic demonstration data from {start_date} to {end_date}...")
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="h", name="timestamp")
    n = len(dates)

    # Physical solar geometry
    geo = compute_solar_geometry(dates, latitude_deg=56.0, longitude_deg=9.0)
    elev = geo["solar_elevation_deg"].values
    daylight_mask = elev > 0

    # Solar generation with diurnal curve and zero at night
    clearsky_proxy = np.maximum(0.0, np.sin(np.radians(np.maximum(0.0, elev))))
    solar_base = clearsky_proxy * 520.0 * (0.85 + 0.15 * np.cos(dates.dayofyear.values * 2 * np.pi / 365.0))
    solar_noise = np.random.normal(1.0, 0.12, size=n).clip(0.3, 1.2)
    solar_gen = np.where(daylight_mask, solar_base * solar_noise, 0.0).astype(np.float32)

    # 13 identified telemetry gaps replication (especially June 2-3, 2020 freeze)
    is_telemetry_gap = np.zeros(n, dtype=bool)
    freeze_mask = (dates >= "2020-06-02 09:00:00+00:00") & (dates <= "2020-06-03 10:00:00+00:00")
    if np.any(freeze_mask):
        solar_gen[freeze_mask] = 284.59
        is_telemetry_gap[freeze_mask] = True

    # Electrical load (MW)
    hour = dates.hour.values
    diurnal_load = 2200.0 + 350.0 * np.sin((hour - 6) * 2 * np.pi / 24.0)
    load_noise = np.random.normal(0, 80, size=n)
    load_actual = np.clip(diurnal_load + load_noise, 1200.0, 4500.0).astype(np.float32)
    load_forecast = np.clip(load_actual + np.random.normal(0, 45, size=n), 1200.0, 4500.0).astype(np.float32)

    # Wind generation (MW)
    wind_gen = (1800.0 + 600.0 * np.sin(dates.dayofyear.values * 2 * np.pi / 30.0) + np.random.normal(0, 300, size=n))
    wind_gen = np.clip(wind_gen, 100.0, 4200.0).astype(np.float32)

    # Day-ahead spot price (EUR/MWh) with diurnal swing, solar merit order effect, and occasional negative pricing
    price_base = 38.0 + 15.0 * np.sin((hour - 8) * 2 * np.pi / 24.0)
    solar_depression = 18.0 * (solar_gen / 600.0)
    price_noise = np.random.normal(0, 10, size=n)
    price = (price_base - solar_depression + price_noise).astype(np.float32)

    capacity_proxy = np.full(n, 726.71, dtype=np.float32)

    df = pd.DataFrame(
        {
            "load_actual_mw": load_actual,
            "load_forecast_mw": load_forecast,
            "price_day_ahead_eur_mwh": price,
            "solar_generation_mw": solar_gen,
            "wind_generation_mw": wind_gen,
            "is_telemetry_gap": is_telemetry_gap,
            "capacity_proxy_mw": capacity_proxy,
        },
        index=dates,
    )
    return df


def generate_demo_predictions_dataframe(
    config: Optional[Dict[str, Any]] = None,
    start_date: str = "2020-01-01 00:00:00+00:00",
    end_date: str = "2020-09-30 23:00:00+00:00",
) -> pd.DataFrame:
    """Generates synthetic test predictions for 2020 test period matching test_predictions.parquet schema."""
    clean_df = generate_demo_clean_dataframe(config, start_date=start_date, end_date=end_date)
    dates = clean_df.index
    n = len(dates)
    np.random.seed(42)

    geo = compute_solar_geometry(dates, latitude_deg=56.0, longitude_deg=9.0)
    elev = geo["solar_elevation_deg"].values.astype(np.float32)
    daylight = elev > 0

    actual = clean_df["solar_generation_mw"].values
    clearsky = np.maximum(0.0, np.sin(np.radians(np.maximum(0.0, elev)))).astype(np.float32)

    # Leakage-safe baselines
    baseline_lad = np.roll(actual, 48)  # Last-available-day (48h lag at 12:00 cutoff)
    baseline_sn48 = np.roll(actual, 48)
    baseline_sn168 = np.roll(actual, 168)

    # Model predictions with realistic accuracy improvements and strict night zeros
    lgb_noise = np.random.normal(0, 22, size=n).astype(np.float32)
    rf_noise = np.random.normal(0, 26, size=n).astype(np.float32)
    ridge_noise = np.random.normal(0, 32, size=n).astype(np.float32)

    model_lightgbm = np.where(daylight, np.clip(actual + lgb_noise, 0.0, 726.71), 0.0).astype(np.float32)
    model_rf = np.where(daylight, np.clip(actual + rf_noise, 0.0, 726.71), 0.0).astype(np.float32)
    model_ridge = np.where(daylight, np.clip(actual + ridge_noise, 0.0, 726.71), 0.0).astype(np.float32)

    # Quantile intervals
    p10 = np.where(daylight, np.maximum(0.0, model_lightgbm - 18.0), 0.0).astype(np.float32)
    p50 = model_lightgbm
    p90 = np.where(daylight, np.minimum(726.71, model_lightgbm + 24.0), 0.0).astype(np.float32)

    # Conformal bands
    lower_90 = np.where(daylight, np.maximum(0.0, model_lightgbm - 28.0), 0.0).astype(np.float32)
    upper_90 = np.where(daylight, np.minimum(726.71, model_lightgbm + 32.0), 0.0).astype(np.float32)
    lower_80 = np.where(daylight, np.maximum(0.0, model_lightgbm - 20.0), 0.0).astype(np.float32)
    upper_80 = np.where(daylight, np.minimum(726.71, model_lightgbm + 22.0), 0.0).astype(np.float32)

    preds_df = pd.DataFrame(
        {
            "actual": actual,
            "capacity_proxy_mw": clean_df["capacity_proxy_mw"].values,
            "solar_elevation_deg": elev,
            "clearsky_proxy": clearsky,
            "is_daylight": daylight,
            "is_telemetry_gap": clean_df["is_telemetry_gap"].values,
            "baseline_last_available_day": baseline_lad,
            "baseline_seasonal_naive_48h": baseline_sn48,
            "baseline_seasonal_naive_168h": baseline_sn168,
            "model_ridge": model_ridge,
            "model_random_forest": model_rf,
            "model_lightgbm": model_lightgbm,
            "lightgbm_p10": p10,
            "lightgbm_p50": p50,
            "lightgbm_p90": p90,
            "lightgbm_lower_90": lower_90,
            "lightgbm_upper_90": upper_90,
            "lightgbm_lower_80": lower_80,
            "lightgbm_upper_80": upper_80,
        },
        index=dates,
    )
    return preds_df

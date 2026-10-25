"""Tests for feature engineering, solar geometry, chronological split, and leakage prevention."""
import numpy as np
import pandas as pd
import pytest

from src.features import compute_solar_geometry, build_forecasting_features, verify_feature_leakage
from src.forecasting import split_train_val_test
from src.config import load_config


def test_solar_geometry_physical_properties():
    """Validates that astronomical solar geometry obeys physical bounds."""
    # Midsummer noon vs midnight in Jutland, Denmark (56N, 9E)
    dates = pd.date_range("2020-06-21 00:00:00", "2020-06-21 23:00:00", freq="1h", tz="UTC")
    geo = compute_solar_geometry(dates, latitude_deg=56.0, longitude_deg=9.0)

    assert "solar_elevation_deg" in geo.columns
    assert "clearsky_proxy" in geo.columns
    assert "is_daylight" in geo.columns

    # Solar elevation should be highest around solar noon (11:00-12:00 UTC)
    solar_noon_idx = dates[11]
    assert geo.loc[solar_noon_idx, "solar_elevation_deg"] > 50.0  # Approx 57.4 degrees
    assert geo.loc[dates[0], "solar_elevation_deg"] < 0.0         # Nighttime (sun below horizon)
    assert geo.loc[dates[0], "clearsky_proxy"] == 0.0             # Night clearsky must be 0
    assert geo.loc[dates[0], "is_daylight"] == 0


def test_feature_engineering_strictly_non_negative_lags():
    """Ensures feature engineering does not introduce future lookahead."""
    dates = pd.date_range("2020-01-01 00:00:00", "2020-01-20 23:00:00", freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    synthetic_solar = rng.uniform(0, 500, size=len(dates))
    synthetic_load = rng.uniform(1000, 3000, size=len(dates))

    df = pd.DataFrame(
        {
            "solar_generation_mw": synthetic_solar,
            "load_forecast_mw": synthetic_load,
        },
        index=dates,
    )

    # Must pass strict leakage test
    assert verify_feature_leakage(df, target_col="solar_generation_mw", min_lag_hours=24)

    feat_df = build_forecasting_features(df, target_col="solar_generation_mw", min_lag_hours=24)
    assert not feat_df.isna().any().any()
    assert "target_lag_24h" in feat_df.columns
    assert "target_lag_168h" in feat_df.columns
    assert "target_rolling_mean_24h" in feat_df.columns


def test_chronological_split_integrity():
    """Ensures train, validation, and test splits are strictly disjoint and ordered."""
    dates = pd.date_range("2017-01-01 00:00:00", "2020-09-30 23:00:00", freq="1h", tz="UTC")
    df = pd.DataFrame({"target": np.ones(len(dates))}, index=dates)

    config = {
        "splits": {
            "train_end": "2018-12-31 23:00:00+00:00",
            "val_end": "2019-12-31 23:00:00+00:00",
            "test_end": "2020-09-30 23:00:00+00:00",
        }
    }

    train, val, test = split_train_val_test(df, config)

    assert train.index.max() <= pd.to_datetime(config["splits"]["train_end"])
    assert val.index.min() > train.index.max()
    assert val.index.max() <= pd.to_datetime(config["splits"]["val_end"])
    assert test.index.min() > val.index.max()
    assert test.index.max() <= pd.to_datetime(config["splits"]["test_end"])

    # Disjointness check
    assert len(train.index.intersection(val.index)) == 0
    assert len(val.index.intersection(test.index)) == 0
    assert len(train.index.intersection(test.index)) == 0

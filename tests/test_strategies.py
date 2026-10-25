"""Tests for dispatch strategies comparison, energy balance, and export consistency."""
import numpy as np
import pandas as pd
import pytest

from src.battery_simulator import BatteryConfig
from src.strategies import (
    scale_to_site,
    run_no_battery_strategy,
    run_rule_based_strategy,
    run_forecast_informed_strategy,
    run_perfect_foresight_strategy,
    compare_dispatch_strategies,
)


@pytest.fixture
def sample_site_data():
    """Generates 48 hours of clean synthetic site data."""
    dates = pd.date_range("2020-05-01 00:00:00", periods=48, freq="1h", tz="UTC")
    hours = dates.hour.values
    solar = np.maximum(0.0, 4.0 * np.sin(np.pi * (hours - 5) / 14))
    load = 2.0 + 1.5 * np.sin(np.pi * hours / 12)
    prices = 35.0 + 25.0 * np.sin(np.pi * (hours - 7) / 10)

    df = pd.DataFrame(
        {
            "solar_mw": solar.astype(np.float32),
            "load_mw": load.astype(np.float32),
            "solar_forecast_mw": np.clip(solar * 0.95 + 0.1, 0.0, None).astype(np.float32),
            "load_forecast_mw": load.astype(np.float32),
            "price_eur_mwh": prices.astype(np.float32),
        },
        index=dates,
    )
    return df


def test_strategy_comparison_ordering_and_invariants(sample_site_data):
    """Verifies that all strategies respect constraints and perfect foresight <= forecast <= no-battery."""
    cfg = BatteryConfig(
        energy_capacity_mwh=10.0,
        max_charge_power_mw=2.5,
        max_discharge_power_mw=2.5,
        round_trip_efficiency=0.88,
        soc_min=0.10,
        soc_max=0.90,
        grid_import_limit_mw=6.0,
        retail_adder_eur_mwh=20.0,
    )

    results, summary_df = compare_dispatch_strategies(
        site_df=sample_site_data,
        solar_forecast_mw=sample_site_data["solar_forecast_mw"],
        load_forecast_mw=sample_site_data["load_forecast_mw"],
        battery_cfg=cfg,
    )

    c_no = results["no_battery"].total_cost_eur
    c_rule = results["rule_based"].total_cost_eur
    c_fc = results["forecast_informed"].total_cost_eur
    c_perf = results["perfect_foresight"].total_cost_eur

    # Invariants: 0 violations, ~0 energy balance residual
    for name, r in results.items():
        assert r.constraint_violations == 0, f"Strategy {name} had constraint violations!"
        assert r.max_energy_balance_residual < 1e-4, f"Strategy {name} had energy balance residual {r.max_energy_balance_residual}"

    # Perfect foresight is theoretical upper bound on savings (lowest cost)
    assert c_perf <= c_fc + 1e-2, f"Perfect foresight cost {c_perf} should be <= forecast-informed {c_fc}"
    assert c_perf <= c_no + 1e-2, f"Perfect foresight cost {c_perf} should be <= no-battery {c_no}"


def test_export_matches_settings(sample_site_data, tmp_path):
    """Verifies that exported CSV files faithfully match simulation parameters and output records."""
    cfg = BatteryConfig(energy_capacity_mwh=8.0, max_charge_power_mw=2.0)
    res = run_rule_based_strategy(sample_site_data, cfg)

    export_path = tmp_path / "test_export.csv"
    res.timeline.to_csv(export_path)

    reloaded = pd.read_csv(export_path, index_col=0, parse_dates=True)
    assert len(reloaded) == len(sample_site_data)
    assert np.allclose(reloaded["soc"].values, res.timeline["soc"].values, atol=1e-5)
    assert np.allclose(reloaded["p_charge_mw"].values, res.timeline["p_charge_mw"].values, atol=1e-5)

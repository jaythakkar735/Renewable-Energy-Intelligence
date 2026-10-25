"""Tests for battery physical simulator, SoC dynamics, and settlement invariants."""
import pytest
import numpy as np
import pandas as pd

from src.battery_simulator import BatteryConfig, BatterySimulator


def test_battery_soc_update_and_efficiency():
    """Validates SoC updates with round-trip efficiency losses.

    Config: E_cap = 10 MWh, RTE = 81% -> eta_ch = eta_dis = 0.90.
    Initial SoC: 0.50 (5.0 MWh stored).
    Step 1: Charge at 2.0 MW for 1 hour.
      Energy entering cells = 2.0 * 0.90 = 1.80 MWh.
      New SoC = 0.50 + 1.80 / 10.0 = 0.68.
    Step 2: Discharge at 1.8 MW for 1 hour.
      Energy drawn from cells = 1.8 / 0.90 = 2.0 MWh.
      New SoC = 0.68 - 2.0 / 10.0 = 0.48.
    """
    cfg = BatteryConfig(
        energy_capacity_mwh=10.0,
        max_charge_power_mw=5.0,
        max_discharge_power_mw=5.0,
        round_trip_efficiency=0.81,  # sqrt(0.81) = 0.90
        soc_min=0.0,
        soc_max=1.0,
        soc_initial=0.50,
        retail_adder_eur_mwh=20.0,
        export_ratio=1.0,
    )
    sim = BatterySimulator(cfg)

    # Step 1: Charge
    res1 = sim.step(
        solar_mw=2.0,
        load_mw=0.0,
        price_eur_mwh=30.0,
        requested_p_charge_mw=2.0,
        requested_p_discharge_mw=0.0,
    )
    assert res1["p_charge_mw"] == pytest.approx(2.0, abs=1e-5)
    assert res1["soc"] == pytest.approx(0.68, abs=1e-5)
    assert res1["residual_mw"] == pytest.approx(0.0, abs=1e-5)

    # Step 2: Discharge
    res2 = sim.step(
        solar_mw=0.0,
        load_mw=1.8,
        price_eur_mwh=50.0,
        requested_p_charge_mw=0.0,
        requested_p_discharge_mw=1.8,
    )
    assert res2["p_discharge_mw"] == pytest.approx(1.8, abs=1e-5)
    assert res2["soc"] == pytest.approx(0.48, abs=1e-5)
    assert res2["residual_mw"] == pytest.approx(0.0, abs=1e-5)


def test_battery_power_and_soc_clamping():
    """Ensures simulator clamps power ratings and respects SoC bounds."""
    cfg = BatteryConfig(
        energy_capacity_mwh=10.0,
        max_charge_power_mw=2.0,
        max_discharge_power_mw=2.0,
        round_trip_efficiency=1.0,
        soc_min=0.20,
        soc_max=0.80,
        soc_initial=0.75,
    )
    sim = BatterySimulator(cfg)

    # Attempt to charge at 5.0 MW (exceeds max power rating 2.0 MW, and only 0.5 MWh headroom to 0.80)
    # Headroom = (0.80 - 0.75) * 10 = 0.5 MWh.
    # Therefore actual charge power should be clamped to 0.5 MW!
    res = sim.step(
        solar_mw=5.0,
        load_mw=0.0,
        price_eur_mwh=30.0,
        requested_p_charge_mw=5.0,
        requested_p_discharge_mw=0.0,
    )
    assert res["p_charge_mw"] == pytest.approx(0.5, abs=1e-5)
    assert res["soc"] == pytest.approx(0.80, abs=1e-5)
    assert res["residual_mw"] == pytest.approx(0.0, abs=1e-5)


def test_simultaneous_charge_and_discharge_raises_error():
    """Asserts that simultaneous charging and discharging is strictly prohibited."""
    cfg = BatteryConfig()
    sim = BatterySimulator(cfg)

    with pytest.raises(ValueError, match="Simultaneous charge"):
        sim.step(
            solar_mw=2.0,
            load_mw=2.0,
            price_eur_mwh=30.0,
            requested_p_charge_mw=1.0,
            requested_p_discharge_mw=1.0,
        )


def test_hand_worked_cost_calculation():
    """Validates cost formula against hand calculation.

    Solar = 0 MW, Load = 3.0 MW.
    Battery Discharges = 1.0 MW.
    Grid Import required = 2.0 MW (2.0 MWh in 1h).
    Price = 40.0 EUR/MWh, Retail Adder = 15.0 EUR/MWh.
    Import Price = 55.0 EUR/MWh.
    Degradation cost = 2.0 EUR/MWh throughput -> 1.0 MWh * 2.0 = 2.0 EUR.
    Expected Net Cost = 2.0 MWh * 55.0 EUR/MWh + 2.0 EUR = 112.0 EUR.
    """
    cfg = BatteryConfig(
        retail_adder_eur_mwh=15.0,
        degradation_cost_eur_mwh=2.0,
        soc_initial=0.50,
    )
    sim = BatterySimulator(cfg)
    res = sim.step(
        solar_mw=0.0,
        load_mw=3.0,
        price_eur_mwh=40.0,
        requested_p_charge_mw=0.0,
        requested_p_discharge_mw=1.0,
    )
    assert res["p_import_mw"] == pytest.approx(2.0, abs=1e-5)
    assert res["import_cost_eur"] == pytest.approx(110.0, abs=1e-5)
    assert res["degradation_cost_eur"] == pytest.approx(2.0, abs=1e-5)
    assert res["net_cost_eur"] == pytest.approx(112.0, abs=1e-5)
    assert res["residual_mw"] == pytest.approx(0.0, abs=1e-5)

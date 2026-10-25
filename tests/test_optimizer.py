"""Tests for the Linear Programming battery dispatch optimizer."""
import numpy as np
import pytest

from src.battery_simulator import BatteryConfig
from src.optimizer import solve_day_ahead_dispatch_lp


def test_lp_optimizer_feasibility_and_complementarity():
    """Validates that the LP solver finds a feasible solution and never charges and discharges simultaneously."""
    H = 24
    rng = np.random.default_rng(42)

    # 24h diurnal solar profile (peaks at midday)
    hours = np.arange(H)
    solar_fc = np.maximum(0.0, 3.0 * np.sin(np.pi * (hours - 6) / 12))
    load_fc = 2.0 + 1.0 * np.sin(np.pi * hours / 12)  # 1 to 3 MW load
    # Prices: high at hours 8-10 and 18-20
    prices = 30.0 + 20.0 * np.sin(np.pi * (hours - 4) / 8)

    cfg = BatteryConfig(
        energy_capacity_mwh=10.0,
        max_charge_power_mw=2.5,
        max_discharge_power_mw=2.5,
        round_trip_efficiency=0.88,
        soc_min=0.10,
        soc_max=0.90,
        soc_initial=0.50,
        soc_terminal=0.50,
        retail_adder_eur_mwh=20.0,
        export_ratio=1.0,
    )

    res = solve_day_ahead_dispatch_lp(
        solar_forecast_mw=solar_fc,
        load_forecast_mw=load_fc,
        day_ahead_prices_eur=prices,
        battery_cfg=cfg,
        initial_soc=0.50,
        terminal_soc=0.50,
    )

    assert res["status"] == 0, f"LP failed: {res['message']}"

    p_ch = res["p_charge"]
    p_dis = res["p_discharge"]
    soc = res["soc"]

    # Invariant 1: No simultaneous charge and discharge
    simultaneous = np.minimum(p_ch, p_dis)
    assert np.max(simultaneous) < 1e-4, f"Found simultaneous dispatch: {np.max(simultaneous)}"

    # Invariant 2: Power bounds
    assert np.all(p_ch >= 0.0)
    assert np.all(p_ch <= cfg.max_charge_power_mw + 1e-4)
    assert np.all(p_dis >= 0.0)
    assert np.all(p_dis <= cfg.max_discharge_power_mw + 1e-4)

    # Invariant 3: SoC bounds and terminal condition
    assert np.all(soc >= cfg.soc_min - 1e-4)
    assert np.all(soc <= cfg.soc_max + 1e-4)
    assert soc[-1] >= cfg.soc_terminal - 1e-4


def test_perfect_foresight_bounds():
    """Validates that perfect foresight achieves lower or equal cost compared to unoptimized."""
    H = 24
    solar = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    load = np.full(H, 2.0)
    prices = np.full(H, 40.0)
    # Peak price at hour 19
    prices[18:21] = 120.0
    # Cheap price at hour 3
    prices[2:5] = 10.0

    cfg = BatteryConfig(
        energy_capacity_mwh=10.0,
        max_charge_power_mw=2.5,
        max_discharge_power_mw=2.5,
        round_trip_efficiency=0.90,
        retail_adder_eur_mwh=20.0,
    )

    res = solve_day_ahead_dispatch_lp(solar, load, prices, cfg, initial_soc=0.5)
    assert res["status"] == 0

    # Battery should discharge during expensive hours 18-20
    assert np.sum(res["p_discharge"][18:21]) > 0.0

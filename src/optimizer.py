"""Linear Programming battery dispatch optimizer using scipy.optimize.linprog (HiGHS)."""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.battery_simulator import BatteryConfig
from src.utils import setup_logger

logger = setup_logger(__name__)


def solve_day_ahead_dispatch_lp(
    solar_forecast_mw: np.ndarray,
    load_forecast_mw: np.ndarray,
    day_ahead_prices_eur: np.ndarray,
    battery_cfg: BatteryConfig,
    initial_soc: float,
    terminal_soc: Optional[float] = None,
    dt_hours: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Solves the optimal day-ahead battery dispatch schedule via Linear Programming.

    Minimizes total net energy procurement cost while respecting battery power limits,
    SoC bounds, round-trip efficiency losses, and grid interconnect constraints.

    Args:
        solar_forecast_mw: Hourly projected solar generation array of length H.
        load_forecast_mw: Hourly projected load demand array of length H.
        day_ahead_prices_eur: Day-ahead spot market price array of length H (EUR/MWh).
        battery_cfg: Battery storage specifications.
        initial_soc: State of charge at the start of the horizon.
        terminal_soc: Minimum state of charge required at horizon end.
        dt_hours: Time step duration in hours (default 1.0).

    Returns:
        Dictionary of optimal scheduled arrays:
        {'p_charge': ..., 'p_discharge': ..., 'p_import': ...,
         'p_export': ..., 'p_curtail': ..., 'soc': ..., 'status': int, 'message': str}
    """
    H = len(solar_forecast_mw)
    if len(load_forecast_mw) != H or len(day_ahead_prices_eur) != H:
        raise ValueError("Length of solar, load, and price arrays must match.")

    # 6 decision variables per time step:
    # 0: p_ch, 1: p_dis, 2: p_imp, 3: p_exp, 4: p_curt, 5: soc
    num_vars = 6 * H

    # Objective vector c:
    # Minimize sum_t [ C_deg * p_ch + C_deg * p_dis + (price + adder) * p_imp - (price * export_ratio) * p_exp ] * dt
    c = np.zeros(num_vars, dtype=np.float64)
    adder = battery_cfg.retail_adder_eur_mwh
    ratio = battery_cfg.export_ratio
    deg_cost = battery_cfg.degradation_cost_eur_mwh

    for t in range(H):
        idx = 6 * t
        price_t = float(day_ahead_prices_eur[t])
        c[idx + 0] = deg_cost * dt_hours                          # p_ch
        c[idx + 1] = deg_cost * dt_hours                          # p_dis
        c[idx + 2] = (price_t + adder) * dt_hours                 # p_imp
        c[idx + 3] = -(price_t * ratio) * dt_hours                # p_exp
        c[idx + 4] = 0.0                                          # p_curt
        c[idx + 5] = 0.0                                          # soc

    # Equality constraints A_eq * x = b_eq
    # 1. Power balance (H equations):
    # -p_ch(t) + p_dis(t) + p_imp(t) - p_exp(t) - p_curt(t) = load(t) - solar(t)
    # 2. SoC transition (H equations):
    # - (eta_ch * dt / E_cap) * p_ch(t) + (dt / (eta_dis * E_cap)) * p_dis(t) + soc(t) - soc(t-1) = 0
    num_eq = 2 * H
    A_eq = np.zeros((num_eq, num_vars), dtype=np.float64)
    b_eq = np.zeros(num_eq, dtype=np.float64)

    eta_ch = battery_cfg.charge_efficiency
    eta_dis = battery_cfg.discharge_efficiency
    E_cap = battery_cfg.energy_capacity_mwh

    coeff_ch = -(eta_ch * dt_hours) / E_cap
    coeff_dis = dt_hours / (eta_dis * E_cap)

    for t in range(H):
        idx = 6 * t
        # Power balance
        row_pb = t
        A_eq[row_pb, idx + 0] = -1.0  # -p_ch
        A_eq[row_pb, idx + 1] = 1.0   # +p_dis
        A_eq[row_pb, idx + 2] = 1.0   # +p_imp
        A_eq[row_pb, idx + 3] = -1.0  # -p_exp
        A_eq[row_pb, idx + 4] = -1.0  # -p_curt
        b_eq[row_pb] = float(load_forecast_mw[t] - solar_forecast_mw[t])

        # SoC dynamic transition
        row_soc = H + t
        A_eq[row_soc, idx + 0] = coeff_ch
        A_eq[row_soc, idx + 1] = coeff_dis
        A_eq[row_soc, idx + 5] = 1.0  # +soc(t)
        if t == 0:
            b_eq[row_soc] = initial_soc
        else:
            prev_idx = 6 * (t - 1)
            A_eq[row_soc, prev_idx + 5] = -1.0  # -soc(t-1)
            b_eq[row_soc] = 0.0

    # Variable bounds: (min, max)
    bounds = []
    for t in range(H):
        s_avail = float(max(0.0, solar_forecast_mw[t]))
        bounds.append((0.0, battery_cfg.max_charge_power_mw))       # p_ch
        bounds.append((0.0, battery_cfg.max_discharge_power_mw))    # p_dis
        bounds.append((0.0, battery_cfg.grid_import_limit_mw))      # p_imp
        bounds.append((0.0, battery_cfg.grid_export_limit_mw))      # p_exp
        # Curtailment is bounded by available solar to prevent negative-price energy burning
        bounds.append((0.0, s_avail))                               # p_curt
        # Terminal constraint on the final step
        if t == H - 1 and terminal_soc is not None:
            bounds.append((max(battery_cfg.soc_min, terminal_soc), battery_cfg.soc_max))
        else:
            bounds.append((battery_cfg.soc_min, battery_cfg.soc_max))

    # Solve using HiGHS solver
    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )

    if not res.success:
        logger.warning(f"LP optimization failed with message: {res.message}. Falling back to zero dispatch.")
        zeros = np.zeros(H, dtype=np.float64)
        return {
            "p_charge": zeros,
            "p_discharge": zeros,
            "p_import": np.clip(load_forecast_mw - solar_forecast_mw, 0.0, None),
            "p_export": np.clip(solar_forecast_mw - load_forecast_mw, 0.0, None),
            "p_curtail": zeros,
            "soc": np.full(H, initial_soc, dtype=np.float64),
            "status": res.status,
            "message": res.message,
            "cost_objective": 0.0,
        }

    sol = res.x.reshape((H, 6))
    p_charge = np.clip(sol[:, 0], 0.0, battery_cfg.max_charge_power_mw)
    p_discharge = np.clip(sol[:, 1], 0.0, battery_cfg.max_discharge_power_mw)
    p_import = np.clip(sol[:, 2], 0.0, battery_cfg.grid_import_limit_mw)
    p_export = np.clip(sol[:, 3], 0.0, battery_cfg.grid_export_limit_mw)
    p_curtail = np.clip(sol[:, 4], 0.0, None)
    soc_vals = np.clip(sol[:, 5], battery_cfg.soc_min, battery_cfg.soc_max)

    # Post-solve physical complementarity: guarantee no simultaneous charge and discharge
    overlap = (p_charge > 1e-4) & (p_discharge > 1e-4)
    if np.any(overlap):
        net = p_charge - p_discharge
        p_charge = np.where(net > 0, net, 0.0)
        p_discharge = np.where(net < 0, -net, 0.0)

    return {
        "p_charge": p_charge,
        "p_discharge": p_discharge,
        "p_import": p_import,
        "p_export": p_export,
        "p_curtail": p_curtail,
        "soc": soc_vals,
        "status": res.status,
        "message": res.message,
        "cost_objective": res.fun,
    }

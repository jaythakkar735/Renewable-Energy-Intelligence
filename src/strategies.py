"""Dispatch strategies: No-Battery, Rule-Based, Forecast-Informed LP, and Perfect Foresight."""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.battery_simulator import BatteryConfig, BatterySimulator, SimulationResult
from src.optimizer import solve_day_ahead_dispatch_lp
from src.utils import setup_logger

logger = setup_logger(__name__)


def scale_to_site(
    clean_df: pd.DataFrame,
    peak_load_mw: float = 5.0,
    solar_capacity_mw: float = 5.0,
) -> pd.DataFrame:
    """Scales regional system load and solar generation to facility site scale.

    Args:
        clean_df: Clean DataFrame containing 'load_actual_mw', 'solar_generation_mw',
                  'load_forecast_mw', and 'price_day_ahead_eur_mwh'.
        peak_load_mw: Target peak facility load in MW.
        solar_capacity_mw: Target installed on-site solar PV capacity in MW.

    Returns:
        DataFrame with site-scaled columns: 'load_mw', 'solar_mw', 'load_forecast_mw',
        'price_eur_mwh', and unscaled regional references.
    """
    site_df = pd.DataFrame(index=clean_df.index)

    max_hist_load = float(clean_df["load_actual_mw"].max())
    max_hist_solar = float(clean_df["solar_generation_mw"].max())

    load_scale = peak_load_mw / max_hist_load if max_hist_load > 0 else 1.0
    solar_scale = solar_capacity_mw / max_hist_solar if max_hist_solar > 0 else 1.0

    site_df["load_mw"] = (clean_df["load_actual_mw"] * load_scale).astype(np.float32)
    site_df["solar_mw"] = (clean_df["solar_generation_mw"] * solar_scale).astype(np.float32)
    site_df["load_forecast_mw"] = (clean_df["load_forecast_mw"] * load_scale).astype(np.float32)
    site_df["price_eur_mwh"] = clean_df["price_day_ahead_eur_mwh"].astype(np.float32)

    site_df["solar_scale_factor"] = solar_scale
    site_df["load_scale_factor"] = load_scale

    return site_df


def run_no_battery_strategy(
    site_df: pd.DataFrame,
    battery_cfg: BatteryConfig,
) -> SimulationResult:
    """Simulates passive baseline without energy storage."""
    sim = BatterySimulator(battery_cfg)
    plan_df = site_df.copy()
    plan_df["p_charge_req"] = 0.0
    plan_df["p_discharge_req"] = 0.0
    return sim.simulate(plan_df, strategy_name="No Battery Baseline")


def run_rule_based_strategy(
    site_df: pd.DataFrame,
    battery_cfg: BatteryConfig,
) -> SimulationResult:
    """Simulates a rule-based dispatch heuristic.

    Logic:
    - Whenever solar > load: prioritize charging solar surplus up to max charge power.
    - During daily peak price hours (top 35% of prices for the day): discharge to cover deficit.
    - During daily cheap price hours (bottom 25% of prices for the day) and when SoC < 50%:
      charge from grid up to 50% of charge rating.
    """
    plan_df = site_df.copy()
    plan_df["p_charge_req"] = 0.0
    plan_df["p_discharge_req"] = 0.0

    # Process day-by-day to extract local daily price quantiles
    days = pd.Series(plan_df.index.date, index=plan_df.index)
    unique_days = days.unique()

    p_ch_list = []
    p_dis_list = []

    for d in unique_days:
        day_mask = (days == d)
        sub = plan_df.loc[day_mask]
        prices = sub["price_eur_mwh"].values
        q_low = float(np.quantile(prices, 0.25))
        q_high = float(np.quantile(prices, 0.65))

        for _, row in sub.iterrows():
            solar = float(row["solar_mw"])
            load = float(row["load_mw"])
            price = float(row["price_eur_mwh"])
            surplus = solar - load

            ch = 0.0
            dis = 0.0

            if surplus > 0:
                # Direct solar surplus charge
                ch = min(surplus, battery_cfg.max_charge_power_mw)
            else:
                deficit = -surplus
                if price >= q_high:
                    # High price: discharge to offset peak grid cost
                    dis = min(deficit, battery_cfg.max_discharge_power_mw)
                elif price <= q_low:
                    # Low price: opportunistic grid charge
                    ch = min(battery_cfg.max_charge_power_mw * 0.5, battery_cfg.grid_import_limit_mw)

            p_ch_list.append(ch)
            p_dis_list.append(dis)

    plan_df["p_charge_req"] = p_ch_list
    plan_df["p_discharge_req"] = p_dis_list

    sim = BatterySimulator(battery_cfg)
    return sim.simulate(plan_df, strategy_name="Rule-Based Baseline")


def run_forecast_informed_strategy(
    site_df: pd.DataFrame,
    solar_forecast_mw: np.ndarray | pd.Series,
    load_forecast_mw: np.ndarray | pd.Series,
    battery_cfg: BatteryConfig,
    strategy_label: str = "Forecast-Informed Dispatch",
) -> SimulationResult:
    """Simulates rolling 24-hour day-ahead LP dispatch using generation and load forecasts.

    Solves the LP schedule day-ahead, then simulates realized dispatch against actuals.
    """
    plan_df = site_df.copy()
    plan_df["solar_forecast_mw"] = np.asarray(solar_forecast_mw, dtype=np.float32)
    plan_df["load_forecast_mw"] = np.asarray(load_forecast_mw, dtype=np.float32)

    days = pd.Series(plan_df.index.date, index=plan_df.index)
    unique_days = days.unique()

    p_ch_scheduled = []
    p_dis_scheduled = []
    current_soc = battery_cfg.soc_initial

    for d in unique_days:
        day_mask = (days == d)
        sub = plan_df.loc[day_mask]
        H = len(sub)

        s_fc = sub["solar_forecast_mw"].values
        l_fc = sub["load_forecast_mw"].values
        prices = sub["price_eur_mwh"].values

        lp_res = solve_day_ahead_dispatch_lp(
            solar_forecast_mw=s_fc,
            load_forecast_mw=l_fc,
            day_ahead_prices_eur=prices,
            battery_cfg=battery_cfg,
            initial_soc=current_soc,
            terminal_soc=battery_cfg.soc_terminal if H == 24 else None,
            dt_hours=1.0,
        )

        p_ch_scheduled.extend(lp_res["p_charge"])
        p_dis_scheduled.extend(lp_res["p_discharge"])
        current_soc = float(lp_res["soc"][-1])

    plan_df["p_charge_req"] = p_ch_scheduled
    plan_df["p_discharge_req"] = p_dis_scheduled

    sim = BatterySimulator(battery_cfg)
    return sim.simulate(plan_df, strategy_name=strategy_label)


def run_perfect_foresight_strategy(
    site_df: pd.DataFrame,
    battery_cfg: BatteryConfig,
) -> SimulationResult:
    """Simulates upper bound theoretical LP dispatch using actual ground truth generation and load."""
    return run_forecast_informed_strategy(
        site_df=site_df,
        solar_forecast_mw=site_df["solar_mw"].values,
        load_forecast_mw=site_df["load_mw"].values,
        battery_cfg=battery_cfg,
        strategy_label="Perfect-Foresight (Upper Bound)",
    )


def compare_dispatch_strategies(
    site_df: pd.DataFrame,
    solar_forecast_mw: np.ndarray | pd.Series,
    load_forecast_mw: np.ndarray | pd.Series,
    battery_cfg: BatteryConfig,
) -> Tuple[Dict[str, SimulationResult], pd.DataFrame]:
    """Executes all 4 strategies and computes comparative dispatch & cost summary."""
    res_no_batt = run_no_battery_strategy(site_df, battery_cfg)
    res_rule = run_rule_based_strategy(site_df, battery_cfg)
    res_forecast = run_forecast_informed_strategy(
        site_df, solar_forecast_mw, load_forecast_mw, battery_cfg
    )
    res_perfect = run_perfect_foresight_strategy(site_df, battery_cfg)

    results = {
        "no_battery": res_no_batt,
        "rule_based": res_rule,
        "forecast_informed": res_forecast,
        "perfect_foresight": res_perfect,
    }

    c_no = res_no_batt.total_cost_eur
    c_rule = res_rule.total_cost_eur
    c_fc = res_forecast.total_cost_eur
    c_perf = res_perfect.total_cost_eur

    # Perfect foresight gap: Cost(Rule) - Cost(Perfect)
    pf_gap = max(0.01, c_rule - c_perf)
    fc_captured = max(0.0, c_rule - c_fc)
    pct_gap_captured = (fc_captured / pf_gap) * 100.0

    table_data = []
    for key, r in [
        ("No Battery Baseline", res_no_batt),
        ("Rule-Based Strategy", res_rule),
        ("Forecast-Informed (LP)", res_forecast),
        ("Perfect Foresight (Reference)", res_perfect),
    ]:
        cost = r.total_cost_eur
        sav_no = c_no - cost
        sav_rule = c_rule - cost
        pct_sav_no = (sav_no / c_no) * 100.0 if c_no > 0 else 0.0

        table_data.append({
            "Strategy": key,
            "Total Cost (EUR)": round(cost, 2),
            "Savings vs No Battery (EUR)": round(sav_no, 2),
            "Savings vs No Battery (%)": round(pct_sav_no, 2),
            "Savings vs Rule-Based (EUR)": round(sav_rule, 2),
            "Energy Charged (MWh)": r.total_charged_mwh,
            "Energy Discharged (MWh)": r.total_discharged_mwh,
            "Grid Import (MWh)": r.total_grid_import_mwh,
            "Grid Export (MWh)": r.total_grid_export_mwh,
            "Curtailment (MWh)": r.total_curtailed_mwh,
            "Eq. Full Cycles": r.equivalent_full_cycles,
            "Max Balance Residual (MW)": r.max_energy_balance_residual,
            "Violations": r.constraint_violations,
        })

    summary_df = pd.DataFrame(table_data)
    return results, summary_df

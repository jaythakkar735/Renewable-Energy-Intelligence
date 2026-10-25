"""Dispatch strategies: No-Battery, Rule-Based, LP + Naive, LP + ML, and Perfect Foresight.

Includes 5-strategy comparative analysis, savings waterfall decomposition,
and weekly block bootstrap confidence intervals.
"""
from typing import Dict, Any, List, Optional, Tuple
import json
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

    if "load_forecast_mw" in clean_df.columns:
        site_df["load_forecast_mw"] = (clean_df["load_forecast_mw"] * load_scale).astype(np.float32)
    else:
        site_df["load_forecast_mw"] = site_df["load_mw"]

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
    """Simulates a rule-based dispatch heuristic using known day-ahead prices only.

    Logic:
    - Whenever solar > load: prioritize charging solar surplus up to max charge power.
    - During daily peak price hours (top 35% of prices for the day): discharge to cover deficit.
    - During daily cheap price hours (bottom 25% of prices for the day) and when SoC < 50%:
      charge from grid up to 50% of charge rating.
    """
    plan_df = site_df.copy()
    plan_df["p_charge_req"] = 0.0
    plan_df["p_discharge_req"] = 0.0

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
                ch = min(surplus, battery_cfg.max_charge_power_mw)
            else:
                deficit = -surplus
                if price >= q_high:
                    dis = min(deficit, battery_cfg.max_discharge_power_mw)
                elif price <= q_low:
                    ch = min(battery_cfg.max_charge_power_mw * 0.5, battery_cfg.grid_import_limit_mw)

            p_ch_list.append(ch)
            p_dis_list.append(dis)

    plan_df["p_charge_req"] = p_ch_list
    plan_df["p_discharge_req"] = p_dis_list

    sim = BatterySimulator(battery_cfg)
    return sim.simulate(plan_df, strategy_name="Rule-Based Strategy")


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
        strategy_label="Perfect Foresight (Upper Bound)",
    )


def compute_savings_waterfall(
    cost_no_batt: float,
    cost_rule: float,
    cost_naive: float,
    cost_ml: float,
    cost_perf: float,
) -> Dict[str, Any]:
    """Decomposes electricity procurement savings into 4 interpretable stages:
    1. Addition of battery with heuristic rule: Cost(No Batt) - Cost(Rule)
    2. Smarter LP scheduling (Naive forecast): Cost(Rule) - Cost(LP+Naive)
    3. Incremental value of ML solar forecast: Cost(LP+Naive) - Cost(LP+ML)
    4. Remaining gap to perfect foresight: Cost(LP+ML) - Cost(Perfect)
    """
    rule_savings = cost_no_batt - cost_rule
    smarter_scheduling_value = cost_rule - cost_naive
    ml_forecast_value = cost_naive - cost_ml
    perfect_foresight_gap = cost_ml - cost_perf

    total_possible_savings = cost_no_batt - cost_perf
    total_realized_savings = cost_no_batt - cost_ml

    return {
        "cost_no_battery_eur": round(cost_no_batt, 2),
        "cost_rule_based_eur": round(cost_rule, 2),
        "cost_lp_naive_eur": round(cost_naive, 2),
        "cost_lp_ml_eur": round(cost_ml, 2),
        "cost_perfect_foresight_eur": round(cost_perf, 2),
        "rule_based_savings_eur": round(rule_savings, 2),
        "smarter_scheduling_value_eur": round(smarter_scheduling_value, 2),
        "ml_forecast_value_eur": round(ml_forecast_value, 2),
        "perfect_foresight_gap_eur": round(perfect_foresight_gap, 2),
        "total_realized_savings_eur": round(total_realized_savings, 2),
        "total_possible_savings_eur": round(total_possible_savings, 2),
        "realized_share_of_possible_pct": round(
            (total_realized_savings / total_possible_savings * 100.0) if total_possible_savings > 0 else 0.0, 2
        ),
    }


def compute_block_bootstrap_ci(
    results_dict: Dict[str, SimulationResult],
    block_length_hours: int = 168,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes weekly block bootstrap confidence intervals for dispatch savings.

    Resamples weekly contiguous blocks (168h) with replacement across the test period.
    """
    rng = np.random.RandomState(seed)

    res_no = results_dict["no_battery"].timeline["net_cost_eur"].values
    res_rule = results_dict["rule_based"].timeline["net_cost_eur"].values
    res_naive = results_dict["lp_naive"].timeline["net_cost_eur"].values
    res_ml = results_dict["lp_ml"].timeline["net_cost_eur"].values
    res_perf = results_dict["perfect_foresight"].timeline["net_cost_eur"].values

    n_hours = len(res_no)
    n_blocks = n_hours // block_length_hours
    block_indices = [
        np.arange(i * block_length_hours, (i + 1) * block_length_hours)
        for i in range(n_blocks)
    ]
    # Include remainder if present
    remainder = n_hours % block_length_hours
    if remainder > 0:
        block_indices.append(np.arange(n_blocks * block_length_hours, n_hours))
        n_blocks += 1

    boot_fc_value = []
    boot_rule_savings = []
    boot_total_savings = []

    for _ in range(n_bootstraps):
        sampled_block_ids = rng.choice(n_blocks, size=n_blocks, replace=True)
        idx_sample = np.concatenate([block_indices[b] for b in sampled_block_ids])[:n_hours]

        c_no_b = np.sum(res_no[idx_sample])
        c_rule_b = np.sum(res_rule[idx_sample])
        c_naive_b = np.sum(res_naive[idx_sample])
        c_ml_b = np.sum(res_ml[idx_sample])

        boot_fc_value.append(c_naive_b - c_ml_b)
        boot_rule_savings.append(c_rule_b - c_ml_b)
        boot_total_savings.append(c_no_b - c_ml_b)

    boot_fc_value = np.array(boot_fc_value)
    boot_rule_savings = np.array(boot_rule_savings)
    boot_total_savings = np.array(boot_total_savings)

    def ci_stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 2),
            "ci_lower_95": round(float(np.percentile(arr, 2.5)), 2),
            "ci_upper_95": round(float(np.percentile(arr, 97.5)), 2),
            "std": round(float(np.std(arr)), 2),
        }

    # Daily win rate
    timeline = results_dict["no_battery"].timeline.copy()
    daily_df = pd.DataFrame({
        "rule": results_dict["rule_based"].timeline["net_cost_eur"],
        "lp_naive": results_dict["lp_naive"].timeline["net_cost_eur"],
        "lp_ml": results_dict["lp_ml"].timeline["net_cost_eur"],
        "perfect": results_dict["perfect_foresight"].timeline["net_cost_eur"],
    }, index=timeline.index)

    daily_sum = daily_df.groupby(daily_df.index.date).sum()
    # Which strategy had lowest cost each day among practical strategies (rule, naive, ml)
    practical_cols = ["rule", "lp_naive", "lp_ml"]
    best_practical = daily_sum[practical_cols].idxmin(axis=1)
    win_counts = best_practical.value_counts(normalize=True).to_dict()

    # Daily savings of ML vs Naive: best and worst days
    daily_ml_val = daily_sum["lp_naive"] - daily_sum["lp_ml"]
    best_day = str(daily_ml_val.idxmax())
    best_day_val = round(float(daily_ml_val.max()), 2)
    worst_day = str(daily_ml_val.idxmin())
    worst_day_val = round(float(daily_ml_val.min()), 2)

    return {
        "block_bootstrap_n": n_bootstraps,
        "block_size_hours": block_length_hours,
        "ml_forecast_value_ci": ci_stats(boot_fc_value),
        "savings_vs_rule_based_ci": ci_stats(boot_rule_savings),
        "total_savings_vs_no_battery_ci": ci_stats(boot_total_savings),
        "daily_win_rate_pct": {
            "Rule-Based": round(win_counts.get("rule", 0.0) * 100.0, 1),
            "LP + Naive": round(win_counts.get("lp_naive", 0.0) * 100.0, 1),
            "LP + ML": round(win_counts.get("lp_ml", 0.0) * 100.0, 1),
        },
        "best_day": {"date": best_day, "ml_forecast_value_eur": best_day_val},
        "worst_day": {"date": worst_day, "ml_forecast_value_eur": worst_day_val},
    }


def compare_dispatch_strategies(
    site_df: pd.DataFrame,
    solar_forecast_ml_mw: Optional[np.ndarray | pd.Series] = None,
    load_forecast_mw: Optional[np.ndarray | pd.Series] = None,
    solar_forecast_naive_mw: Optional[np.ndarray | pd.Series] = None,
    battery_cfg: Optional[BatteryConfig] = None,
    solar_forecast_mw: Optional[np.ndarray | pd.Series] = None,
) -> Tuple[Dict[str, SimulationResult], pd.DataFrame, Dict[str, Any]]:
    """Executes all 5 strategies, computes waterfall decomposition, and block bootstrap CIs.

    Strategies:
      1. No Battery Baseline
      2. Rule-Based Strategy
      3. LP + Naive Forecast (Last-Available-Day baseline)
      4. LP + ML Forecast (LightGBM forecast)
      5. Perfect Foresight (Upper Bound)
    """
    if battery_cfg is None:
        battery_cfg = BatteryConfig()

    if solar_forecast_ml_mw is None and solar_forecast_mw is not None:
        solar_forecast_ml_mw = solar_forecast_mw

    if solar_forecast_ml_mw is None:
        solar_forecast_ml_mw = site_df["solar_mw"]

    if solar_forecast_naive_mw is None:
        solar_forecast_naive_mw = solar_forecast_ml_mw

    if load_forecast_mw is None:
        load_forecast_mw = site_df.get("load_forecast_mw", site_df["load_mw"])

    res_no = run_no_battery_strategy(site_df, battery_cfg)
    res_rule = run_rule_based_strategy(site_df, battery_cfg)
    res_naive = run_forecast_informed_strategy(
        site_df, solar_forecast_naive_mw, load_forecast_mw, battery_cfg,
        strategy_label="LP + Naive Forecast"
    )
    res_ml = run_forecast_informed_strategy(
        site_df, solar_forecast_ml_mw, load_forecast_mw, battery_cfg,
        strategy_label="LP + ML Forecast"
    )
    res_perf = run_perfect_foresight_strategy(site_df, battery_cfg)

    results = {
        "no_battery": res_no,
        "rule_based": res_rule,
        "lp_naive": res_naive,
        "lp_ml": res_ml,
        "forecast_informed": res_ml,  # Backward compatible alias
        "perfect_foresight": res_perf,
    }

    c_no = res_no.total_cost_eur
    c_rule = res_rule.total_cost_eur
    c_naive = res_naive.total_cost_eur
    c_ml = res_ml.total_cost_eur
    c_perf = res_perf.total_cost_eur

    # Waterfall decomposition
    waterfall = compute_savings_waterfall(c_no, c_rule, c_naive, c_ml, c_perf)

    # Summary table
    table_data = []
    for key, r in [
        ("No Battery Baseline", res_no),
        ("Rule-Based Strategy", res_rule),
        ("LP + Naive Forecast", res_naive),
        ("LP + ML Forecast", res_ml),
        ("Perfect Foresight (Reference)", res_perf),
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
    return results, summary_df, waterfall

"""Sensitivity analysis sweeps for battery capacity, power, efficiency, tariffs, and forecast quality."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import load_config, get_project_root
from src.battery_simulator import BatteryConfig
from src.strategies import scale_to_site, compare_dispatch_strategies, run_forecast_informed_strategy
from src.utils import setup_logger

logger = setup_logger(__name__)


def run_sensitivity_sweeps(
    clean_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Precomputes offline sensitivity sweeps across battery and economic dimensions.

    Sweeps:
    1. Capacity sweep (5, 10, 15, 20 MWh)
    2. Power sweep (1.0, 1.5, 2.5, 3.5, 5.0 MW)
    3. Round-trip efficiency sweep (0.76, 0.82, 0.88, 0.94)
    4. Retail tariff adder sweep (0, 10, 20, 30, 40 EUR/MWh)
    5. Forecast quality comparison (Perfect, LightGBM, Ridge, Persistence, Degraded noise)

    Returns:
        Dictionary containing sweep tables as serializable records.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    reports_dir = root / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_slice = clean_df.loc[preds_df.index]
    site_df = scale_to_site(
        test_slice,
        peak_load_mw=config["site"]["peak_load_mw"],
        solar_capacity_mw=config["site"]["solar_capacity_mw"],
    )

    solar_fc_lgb = preds_df["model_lightgbm"].values * site_df["solar_scale_factor"].values
    solar_fc_ridge = preds_df["model_ridge"].values * site_df["solar_scale_factor"].values
    solar_fc_persist = preds_df["baseline_persistence_24h"].values * site_df["solar_scale_factor"].values

    base_cfg = BatteryConfig(
        energy_capacity_mwh=config["battery"]["energy_capacity_mwh"],
        max_charge_power_mw=config["battery"]["max_charge_power_mw"],
        max_discharge_power_mw=config["battery"]["max_discharge_power_mw"],
        round_trip_efficiency=config["battery"]["round_trip_efficiency"],
        soc_min=config["battery"]["soc_min"],
        soc_max=config["battery"]["soc_max"],
        grid_import_limit_mw=config["site"]["grid_import_limit_mw"],
        grid_export_limit_mw=config["site"]["grid_export_limit_mw"],
        retail_adder_eur_mwh=config["tariffs"]["retail_adder_eur_mwh"],
        export_ratio=config["tariffs"]["export_ratio"],
        degradation_cost_eur_mwh=config["battery"]["degradation_cost_eur_mwh"],
    )

    logger.info("Executing Capacity Sensitivity Sweep...")
    # 1. Capacity Sweep
    capacity_records = []
    for cap in [5.0, 10.0, 15.0, 20.0]:
        cfg = BatteryConfig(
            energy_capacity_mwh=cap,
            max_charge_power_mw=base_cfg.max_charge_power_mw,
            max_discharge_power_mw=base_cfg.max_discharge_power_mw,
            round_trip_efficiency=base_cfg.round_trip_efficiency,
            grid_import_limit_mw=base_cfg.grid_import_limit_mw,
            retail_adder_eur_mwh=base_cfg.retail_adder_eur_mwh,
        )
        _, summ = compare_dispatch_strategies(site_df, solar_fc_lgb, site_df["load_forecast_mw"], cfg)
        fc_row = summ[summ["Strategy"] == "Forecast-Informed (LP)"].iloc[0]
        perf_row = summ[summ["Strategy"] == "Perfect Foresight (Reference)"].iloc[0]
        capacity_records.append({
            "capacity_mwh": cap,
            "forecast_cost_eur": fc_row["Total Cost (EUR)"],
            "forecast_savings_no_battery_eur": fc_row["Savings vs No Battery (EUR)"],
            "forecast_savings_rule_based_eur": fc_row["Savings vs Rule-Based (EUR)"],
            "perfect_cost_eur": perf_row["Total Cost (EUR)"],
            "cycles": fc_row["Eq. Full Cycles"],
        })
    df_cap = pd.DataFrame(capacity_records)
    df_cap.to_csv(reports_dir / "sensitivity_capacity.csv", index=False)

    logger.info("Executing Power Rating Sweep...")
    # 2. Power Rating Sweep
    power_records = []
    for pwr in [1.0, 1.5, 2.5, 3.5, 5.0]:
        cfg = BatteryConfig(
            energy_capacity_mwh=base_cfg.energy_capacity_mwh,
            max_charge_power_mw=pwr,
            max_discharge_power_mw=pwr,
            round_trip_efficiency=base_cfg.round_trip_efficiency,
            grid_import_limit_mw=base_cfg.grid_import_limit_mw,
            retail_adder_eur_mwh=base_cfg.retail_adder_eur_mwh,
        )
        _, summ = compare_dispatch_strategies(site_df, solar_fc_lgb, site_df["load_forecast_mw"], cfg)
        fc_row = summ[summ["Strategy"] == "Forecast-Informed (LP)"].iloc[0]
        power_records.append({
            "power_mw": pwr,
            "forecast_cost_eur": fc_row["Total Cost (EUR)"],
            "savings_rule_based_eur": fc_row["Savings vs Rule-Based (EUR)"],
            "grid_import_mwh": fc_row["Grid Import (MWh)"],
            "cycles": fc_row["Eq. Full Cycles"],
        })
    df_pwr = pd.DataFrame(power_records)
    df_pwr.to_csv(reports_dir / "sensitivity_power.csv", index=False)

    logger.info("Executing Round-Trip Efficiency Sweep...")
    # 3. Efficiency Sweep
    eff_records = []
    for rte in [0.76, 0.82, 0.88, 0.94]:
        cfg = BatteryConfig(
            energy_capacity_mwh=base_cfg.energy_capacity_mwh,
            max_charge_power_mw=base_cfg.max_charge_power_mw,
            max_discharge_power_mw=base_cfg.max_discharge_power_mw,
            round_trip_efficiency=rte,
            grid_import_limit_mw=base_cfg.grid_import_limit_mw,
            retail_adder_eur_mwh=base_cfg.retail_adder_eur_mwh,
        )
        _, summ = compare_dispatch_strategies(site_df, solar_fc_lgb, site_df["load_forecast_mw"], cfg)
        fc_row = summ[summ["Strategy"] == "Forecast-Informed (LP)"].iloc[0]
        eff_records.append({
            "round_trip_efficiency": rte,
            "forecast_cost_eur": fc_row["Total Cost (EUR)"],
            "savings_rule_based_eur": fc_row["Savings vs Rule-Based (EUR)"],
            "losses_mwh": round(fc_row["Energy Charged (MWh)"] - fc_row["Energy Discharged (MWh)"], 2),
            "cycles": fc_row["Eq. Full Cycles"],
        })
    df_eff = pd.DataFrame(eff_records)
    df_eff.to_csv(reports_dir / "sensitivity_efficiency.csv", index=False)

    logger.info("Executing Retail Adder Sweep...")
    # 4. Retail Adder Sweep
    adder_records = []
    for add in [0.0, 10.0, 20.0, 30.0, 40.0]:
        cfg = BatteryConfig(
            energy_capacity_mwh=base_cfg.energy_capacity_mwh,
            max_charge_power_mw=base_cfg.max_charge_power_mw,
            max_discharge_power_mw=base_cfg.max_discharge_power_mw,
            round_trip_efficiency=base_cfg.round_trip_efficiency,
            grid_import_limit_mw=base_cfg.grid_import_limit_mw,
            retail_adder_eur_mwh=add,
        )
        _, summ = compare_dispatch_strategies(site_df, solar_fc_lgb, site_df["load_forecast_mw"], cfg)
        no_row = summ[summ["Strategy"] == "No Battery Baseline"].iloc[0]
        fc_row = summ[summ["Strategy"] == "Forecast-Informed (LP)"].iloc[0]
        rule_row = summ[summ["Strategy"] == "Rule-Based Strategy"].iloc[0]
        adder_records.append({
            "retail_adder_eur_mwh": add,
            "no_battery_cost_eur": no_row["Total Cost (EUR)"],
            "rule_based_cost_eur": rule_row["Total Cost (EUR)"],
            "forecast_cost_eur": fc_row["Total Cost (EUR)"],
            "forecast_savings_no_battery_eur": fc_row["Savings vs No Battery (EUR)"],
            "forecast_savings_rule_based_eur": fc_row["Savings vs Rule-Based (EUR)"],
        })
    df_add = pd.DataFrame(adder_records)
    df_add.to_csv(reports_dir / "sensitivity_adder.csv", index=False)

    logger.info("Executing Forecast Quality Sweep...")
    # 5. Forecast Quality vs Cost Impact Sweep
    # Synthetic degraded forecast: add 30% and 60% gaussian noise to solar forecast
    np.random.seed(42)
    noise_30 = np.clip(solar_fc_lgb + np.random.normal(0, 0.3 * np.std(solar_fc_lgb), len(solar_fc_lgb)), 0.0, None)
    noise_60 = np.clip(solar_fc_lgb + np.random.normal(0, 0.6 * np.std(solar_fc_lgb), len(solar_fc_lgb)), 0.0, None)

    quality_candidates = [
        ("Perfect Foresight", site_df["solar_mw"].values),
        ("LightGBM Model", solar_fc_lgb),
        ("Ridge Model", solar_fc_ridge),
        ("Persistence Baseline", solar_fc_persist),
        ("Degraded Forecast (+30% Noise)", noise_30),
        ("Degraded Forecast (+60% Noise)", noise_60),
    ]

    quality_records = []
    c_rule = summary_df = None
    _, summ_ref = compare_dispatch_strategies(site_df, solar_fc_lgb, site_df["load_forecast_mw"], base_cfg)
    c_rule = float(summ_ref[summ_ref["Strategy"] == "Rule-Based Strategy"]["Total Cost (EUR)"].iloc[0])
    c_no = float(summ_ref[summ_ref["Strategy"] == "No Battery Baseline"]["Total Cost (EUR)"].iloc[0])

    for label, fc_series in quality_candidates:
        res = run_forecast_informed_strategy(
            site_df=site_df,
            solar_forecast_mw=fc_series,
            load_forecast_mw=site_df["load_forecast_mw"],
            battery_cfg=base_cfg,
            strategy_label=label,
        )
        # Compute RMSE of this solar forecast vs actual
        rmse_fc = float(np.sqrt(np.mean((fc_series - site_df["solar_mw"].values) ** 2)))
        quality_records.append({
            "forecast_source": label,
            "solar_forecast_rmse_mw": round(rmse_fc, 3),
            "simulated_total_cost_eur": res.total_cost_eur,
            "savings_vs_rule_based_eur": round(c_rule - res.total_cost_eur, 2),
            "savings_vs_no_battery_eur": round(c_no - res.total_cost_eur, 2),
            "cycles": res.equivalent_full_cycles,
        })
    df_qual = pd.DataFrame(quality_records)
    df_qual.to_csv(reports_dir / "sensitivity_forecast_quality.csv", index=False)

    all_sens = {
        "capacity": df_cap.to_dict(orient="records"),
        "power": df_pwr.to_dict(orient="records"),
        "efficiency": df_eff.to_dict(orient="records"),
        "adder": df_add.to_dict(orient="records"),
        "forecast_quality": df_qual.to_dict(orient="records"),
    }

    import json
    with open(reports_dir / "sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_sens, f, indent=2)

    logger.info("All sensitivity sweeps successfully precomputed and saved to reports/!")
    return all_sens

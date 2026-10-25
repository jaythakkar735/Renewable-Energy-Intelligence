"""Sensitivity analysis sweeps for battery capacity, power, efficiency, tariffs, export ratio, and forecast quality."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd

from src.config import load_config, get_project_root
from src.battery_simulator import BatteryConfig
from src.strategies import (
    scale_to_site,
    compare_dispatch_strategies,
    run_forecast_informed_strategy,
    compute_block_bootstrap_ci,
)
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
    5. Export ratio sweep (0.0, 0.25, 0.50, 0.75, 1.0)
    6. Forecast quality sensitivity curve (controlled noise vs nMAE vs cost)

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

    solar_scale = site_df["solar_scale_factor"].values
    solar_fc_lgb = preds_df["model_lightgbm"].values * solar_scale
    solar_fc_naive = preds_df["baseline_last_available_day"].values * solar_scale
    solar_actual = site_df["solar_mw"].values
    load_fc = site_df["load_forecast_mw"].values

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

    # ---------------------------------------------------------
    # Baseline 5-Strategy Run
    # ---------------------------------------------------------
    results_base, summ_base, waterfall_base = compare_dispatch_strategies(
        site_df, solar_fc_lgb, load_fc, solar_fc_naive, base_cfg
    )
    summ_base.to_csv(reports_dir / "dispatch_summary.csv", index=False)
    with open(reports_dir / "savings_decomposition.json", "w", encoding="utf-8") as f:
        json.dump(waterfall_base, f, indent=2)

    # Compute block bootstrap CIs
    logger.info("Computing weekly block bootstrap confidence intervals (B=1000)...")
    bootstrap_res = compute_block_bootstrap_ci(results_base, block_length_hours=168, n_bootstraps=1000, seed=42)
    with open(reports_dir / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_res, f, indent=2)

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
        _, summ, _ = compare_dispatch_strategies(site_df, solar_fc_lgb, load_fc, solar_fc_naive, cfg)
        ml_row = summ[summ["Strategy"] == "LP + ML Forecast"].iloc[0]
        naive_row = summ[summ["Strategy"] == "LP + Naive Forecast"].iloc[0]
        perf_row = summ[summ["Strategy"] == "Perfect Foresight (Reference)"].iloc[0]
        capacity_records.append({
            "capacity_mwh": cap,
            "cost_lp_ml_eur": ml_row["Total Cost (EUR)"],
            "cost_lp_naive_eur": naive_row["Total Cost (EUR)"],
            "cost_perfect_eur": perf_row["Total Cost (EUR)"],
            "savings_vs_no_battery_eur": ml_row["Savings vs No Battery (EUR)"],
            "savings_vs_rule_based_eur": ml_row["Savings vs Rule-Based (EUR)"],
            "ml_forecast_value_eur": round(naive_row["Total Cost (EUR)"] - ml_row["Total Cost (EUR)"], 2),
            "cycles": ml_row["Eq. Full Cycles"],
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
        _, summ, _ = compare_dispatch_strategies(site_df, solar_fc_lgb, load_fc, solar_fc_naive, cfg)
        ml_row = summ[summ["Strategy"] == "LP + ML Forecast"].iloc[0]
        naive_row = summ[summ["Strategy"] == "LP + Naive Forecast"].iloc[0]
        power_records.append({
            "power_mw": pwr,
            "cost_lp_ml_eur": ml_row["Total Cost (EUR)"],
            "savings_vs_rule_based_eur": ml_row["Savings vs Rule-Based (EUR)"],
            "ml_forecast_value_eur": round(naive_row["Total Cost (EUR)"] - ml_row["Total Cost (EUR)"], 2),
            "grid_import_mwh": ml_row["Grid Import (MWh)"],
            "cycles": ml_row["Eq. Full Cycles"],
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
        _, summ, _ = compare_dispatch_strategies(site_df, solar_fc_lgb, load_fc, solar_fc_naive, cfg)
        ml_row = summ[summ["Strategy"] == "LP + ML Forecast"].iloc[0]
        eff_records.append({
            "round_trip_efficiency": rte,
            "cost_lp_ml_eur": ml_row["Total Cost (EUR)"],
            "savings_vs_rule_based_eur": ml_row["Savings vs Rule-Based (EUR)"],
            "losses_mwh": round(ml_row["Energy Charged (MWh)"] - ml_row["Energy Discharged (MWh)"], 2),
            "cycles": ml_row["Eq. Full Cycles"],
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
        _, summ, _ = compare_dispatch_strategies(site_df, solar_fc_lgb, load_fc, solar_fc_naive, cfg)
        no_row = summ[summ["Strategy"] == "No Battery Baseline"].iloc[0]
        rule_row = summ[summ["Strategy"] == "Rule-Based Strategy"].iloc[0]
        ml_row = summ[summ["Strategy"] == "LP + ML Forecast"].iloc[0]
        adder_records.append({
            "retail_adder_eur_mwh": add,
            "no_battery_cost_eur": no_row["Total Cost (EUR)"],
            "rule_based_cost_eur": rule_row["Total Cost (EUR)"],
            "lp_ml_cost_eur": ml_row["Total Cost (EUR)"],
            "savings_vs_no_battery_eur": ml_row["Savings vs No Battery (EUR)"],
            "savings_vs_rule_based_eur": ml_row["Savings vs Rule-Based (EUR)"],
        })
    df_add = pd.DataFrame(adder_records)
    df_add.to_csv(reports_dir / "sensitivity_adder.csv", index=False)

    logger.info("Executing Export Ratio Sweep...")
    # 5. Export Ratio Sweep
    export_records = []
    for exp_r in [0.0, 0.25, 0.50, 0.75, 1.0]:
        cfg = BatteryConfig(
            energy_capacity_mwh=base_cfg.energy_capacity_mwh,
            max_charge_power_mw=base_cfg.max_charge_power_mw,
            max_discharge_power_mw=base_cfg.max_discharge_power_mw,
            round_trip_efficiency=base_cfg.round_trip_efficiency,
            grid_import_limit_mw=base_cfg.grid_import_limit_mw,
            retail_adder_eur_mwh=base_cfg.retail_adder_eur_mwh,
            export_ratio=exp_r,
        )
        _, summ, _ = compare_dispatch_strategies(site_df, solar_fc_lgb, load_fc, solar_fc_naive, cfg)
        ml_row = summ[summ["Strategy"] == "LP + ML Forecast"].iloc[0]
        export_records.append({
            "export_ratio": exp_r,
            "cost_lp_ml_eur": ml_row["Total Cost (EUR)"],
            "savings_vs_rule_based_eur": ml_row["Savings vs Rule-Based (EUR)"],
            "grid_export_mwh": ml_row["Grid Export (MWh)"],
            "curtailment_mwh": ml_row["Curtailment (MWh)"],
        })
    df_exp = pd.DataFrame(export_records)
    df_exp.to_csv(reports_dir / "sensitivity_export_ratio.csv", index=False)

    logger.info("Executing Controlled Forecast Quality Sensitivity Curve...")
    # 6. Controlled Forecast Quality vs Cost Impact Sweep
    np.random.seed(42)
    std_solar = float(np.std(solar_actual))
    peak_site_solar = float(np.max(solar_actual))

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    quality_records = []
    cost_naive = float(waterfall_base["cost_lp_naive_eur"])

    c_rule_base = float(waterfall_base["cost_rule_based_eur"])
    c_no_base = float(waterfall_base["cost_no_battery_eur"])

    for noise in noise_levels:
        if noise == 0.0:
            fc_series = solar_actual.copy()
            label = "Perfect Foresight"
        else:
            noise_vector = np.random.normal(0, noise * std_solar, len(solar_actual))
            fc_series = np.clip(solar_actual + noise_vector, 0.0, peak_site_solar * 1.15)
            # Apply night zero
            fc_series = np.where(site_df.index.hour < 5, 0.0, fc_series)
            label = f"Noise {int(noise * 100)}%"

        res = run_forecast_informed_strategy(
            site_df=site_df,
            solar_forecast_mw=fc_series,
            load_forecast_mw=load_fc,
            battery_cfg=base_cfg,
            strategy_label=label,
        )
        mae = float(np.mean(np.abs(fc_series - solar_actual)))
        rmse = float(np.sqrt(np.mean((fc_series - solar_actual) ** 2)))
        nmae_pct = (mae / peak_site_solar) * 100.0 if peak_site_solar > 0 else 0.0

        quality_records.append({
            "noise_fraction": noise,
            "label": label,
            "forecast_source": label,
            "solar_forecast_rmse_mw": round(rmse, 3),
            "forecast_nmae_pct": round(nmae_pct, 2),
            "simulated_total_cost_eur": res.total_cost_eur,
            "cost_vs_naive_eur": round(res.total_cost_eur - cost_naive, 2),
            "savings_vs_rule_based_eur": round(c_rule_base - res.total_cost_eur, 2),
            "savings_vs_no_battery_eur": round(c_no_base - res.total_cost_eur, 2),
            "cycles": res.equivalent_full_cycles,
            "beats_naive": bool(res.total_cost_eur < cost_naive),
        })

    # Also add the actual ML model and Naive baseline for comparison
    mae_lgb = float(np.mean(np.abs(solar_fc_lgb - solar_actual)))
    rmse_lgb = float(np.sqrt(np.mean((solar_fc_lgb - solar_actual) ** 2)))
    nmae_lgb = (mae_lgb / peak_site_solar) * 100.0
    cost_ml_val = float(waterfall_base["cost_lp_ml_eur"])
    quality_records.append({
        "noise_fraction": -1.0,
        "label": "LightGBM Model",
        "forecast_source": "LightGBM Model",
        "solar_forecast_rmse_mw": round(rmse_lgb, 3),
        "forecast_nmae_pct": round(nmae_lgb, 2),
        "simulated_total_cost_eur": cost_ml_val,
        "cost_vs_naive_eur": round(cost_ml_val - cost_naive, 2),
        "savings_vs_rule_based_eur": round(c_rule_base - cost_ml_val, 2),
        "savings_vs_no_battery_eur": round(c_no_base - cost_ml_val, 2),
        "cycles": results_base["lp_ml"].equivalent_full_cycles,
        "beats_naive": bool(cost_ml_val < cost_naive),
    })

    mae_naive = float(np.mean(np.abs(solar_fc_naive - solar_actual)))
    rmse_naive = float(np.sqrt(np.mean((solar_fc_naive - solar_actual) ** 2)))
    nmae_naive = (mae_naive / peak_site_solar) * 100.0
    quality_records.append({
        "noise_fraction": -2.0,
        "label": "Last-Available-Day Naive",
        "forecast_source": "Last-Available-Day Naive",
        "solar_forecast_rmse_mw": round(rmse_naive, 3),
        "forecast_nmae_pct": round(nmae_naive, 2),
        "simulated_total_cost_eur": cost_naive,
        "cost_vs_naive_eur": 0.0,
        "savings_vs_rule_based_eur": round(c_rule_base - cost_naive, 2),
        "savings_vs_no_battery_eur": round(c_no_base - cost_naive, 2),
        "cycles": results_base["lp_naive"].equivalent_full_cycles,
        "beats_naive": False,
    })

    df_qual = pd.DataFrame(quality_records)
    df_qual.to_csv(reports_dir / "sensitivity_forecast_quality.csv", index=False)

    all_sens = {
        "capacity": df_cap.to_dict(orient="records"),
        "power": df_pwr.to_dict(orient="records"),
        "efficiency": df_eff.to_dict(orient="records"),
        "adder": df_add.to_dict(orient="records"),
        "export_ratio": df_exp.to_dict(orient="records"),
        "forecast_quality": df_qual.to_dict(orient="records"),
        "bootstrap": bootstrap_res,
        "waterfall": waterfall_base,
    }

    with open(reports_dir / "sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_sens, f, indent=2)

    logger.info("All sensitivity sweeps successfully precomputed and saved to reports/!")
    return all_sens

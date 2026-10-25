import nbformat as nbf
from pathlib import Path

nb_dir = Path("notebooks")
nb_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# Notebook 01: Data Audit
# -------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# 01 - Data Audit & Ingestion
## Renewable Energy Forecasting & Battery Storage Optimization

This notebook conducts a data audit of the Open Power System Data (OPSD) Time Series package (dated 2020-10-06).
It audits column structures, temporal continuity, missing value patterns across European bidding zones, and justifies the selection of **Denmark West (`DK_1`)** as our primary bidding zone.
"""),
    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

from src.config import load_config
from src.data_loader import load_raw_country_data
from src.preprocessing import validate_and_clean_time_series, get_clean_data
import pandas as pd
import json

config = load_config()
print("Selected country:", config["data"]["selected_country"])
print("Configured columns:", config["data"]["columns_mapping"])
"""),
    nbf.v4.new_markdown_cell("""### 1. Load Raw Country Observations & Verify Frequency"""),
    nbf.v4.new_code_cell("""raw_df = load_raw_country_data(config=config, use_cache=True)
print(f"Total raw observations: {len(raw_df):,}")
print(f"Time span: {raw_df.index.min()} to {raw_df.index.max()}")
print("\\nMissing values per column in raw OPSD data:")
print(raw_df.isna().sum())
"""),
    nbf.v4.new_markdown_cell("""### 2. Validate Regular Hourly Grid & Clean Series"""),
    nbf.v4.new_code_cell("""clean_df, audit_summary = validate_and_clean_time_series(raw_df, config=config)
print("Data Quality & Cleaning Audit Summary:")
print(json.dumps(audit_summary, indent=2))
print("\\nClean dataset summary statistics:")
display(clean_df.describe())
"""),
    nbf.v4.new_markdown_cell("""### 3. Verification of Physical Invariants
- Solar generation >= 0 MW
- Zero missing values on the complete UTC hourly index
"""),
    nbf.v4.new_code_cell("""assert (clean_df['solar_generation_mw'] >= 0).all(), "Negative solar generation detected!"
assert clean_df.isna().sum().sum() == 0, "Remaining NaNs detected!"
print("Physical sanity check passed successfully!")
""")
]

with open(nb_dir / "01_data_audit.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)

# -------------------------------------------------------------
# Notebook 02: Exploratory Data Analysis
# -------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# 02 - Exploratory Data Analysis
## Diurnal & Seasonal Patterns in Generation, Load, and Market Prices

This notebook analyzes the historical relationship between solar PV generation, system electrical demand, and day-ahead electricity prices in Denmark West (`DK_1`).
"""),
    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

from src.preprocessing import get_clean_data
import pandas as pd
import numpy as np

clean_df = get_clean_data()
clean_df['hour'] = clean_df.index.hour
clean_df['month'] = clean_df.index.month
clean_df['season'] = clean_df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})
"""),
    nbf.v4.new_markdown_cell("""### 1. Diurnal Generation and Price Profiles by Season"""),
    nbf.v4.new_code_cell("""diurnal = clean_df.groupby(['season', 'hour']).agg({
    'solar_generation_mw': 'mean',
    'load_actual_mw': 'mean',
    'price_day_ahead_eur_mwh': 'mean'
}).reset_index()

print("Average midday solar generation (Hour 12 UTC) by Season:")
print(diurnal[diurnal['hour'] == 12][['season', 'solar_generation_mw', 'price_day_ahead_eur_mwh']])
"""),
    nbf.v4.new_markdown_cell("""### 2. Spot Price Distribution & Negative Pricing Frequency"""),
    nbf.v4.new_code_cell("""prices = clean_df['price_day_ahead_eur_mwh']
neg_hours = (prices < 0).sum()
pct_neg = (prices < 0).mean() * 100
print(f"Total negative price hours: {neg_hours:,} ({pct_neg:.2f}%)")
print(f"Minimum spot price: {prices.min():.2f} EUR/MWh")
print(f"Median spot price:  {prices.median():.2f} EUR/MWh")
print(f"99th percentile:    {prices.quantile(0.99):.2f} EUR/MWh")
""")
]

with open(nb_dir / "02_exploratory_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)

# -------------------------------------------------------------
# Notebook 03: Forecasting Models
# -------------------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("""# 03 - Renewable Generation Forecasting
## Leakage-Safe Feature Engineering, Baseline Evaluation, and ML Models

This notebook trains day-ahead hourly solar forecasting models using strictly historical information (lag >= 24h), solar geometry clear-sky proxies, and evaluates them against Persistence and Seasonal Naive baselines.
"""),
    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

from src.preprocessing import get_clean_data
from src.features import build_forecasting_features, verify_feature_leakage
from src.forecasting import train_and_evaluate_models
import pandas as pd

clean_df = get_clean_data()
print("Running anti-leakage verification test...")
verify_feature_leakage(clean_df)
"""),
    nbf.v4.new_markdown_cell("""### 1. Train Models & Evaluate on Chronological Holdout Set (2020)"""),
    nbf.v4.new_code_cell("""preds, metrics, leaderboard = train_and_evaluate_models(clean_df, force_retrain=False)
print("=== Model Leaderboard (Sorted by MAE) ===")
display(leaderboard)
"""),
    nbf.v4.new_markdown_cell("""### 2. Conformal Prediction Intervals & Empirical Coverage"""),
    nbf.v4.new_code_cell("""print("90% Conformal Prediction Interval Metrics:")
print(metrics.get("Conformal_90_Coverage"))
print("80% Conformal Prediction Interval Metrics:")
print(metrics.get("Conformal_80_Coverage"))
""")
]

with open(nb_dir / "03_forecasting.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb3, f)

# -------------------------------------------------------------
# Notebook 04: Battery Simulation & Optimization
# -------------------------------------------------------------
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell("""# 04 - Battery Dispatch & Cost Optimization
## Linear Program Formulation, Rolling Simulation, and Strategy Comparison

This notebook scales historical time series to a commercial facility site (5 MW peak load, 5 MW solar PV) and simulates four battery dispatch strategies:
1. No Battery Baseline
2. Rule-Based Strategy (heuristic peak discharge / off-peak charge)
3. Forecast-Informed Strategy (rolling 24h LP via HiGHS using LightGBM generation forecast)
4. Perfect Foresight Reference (upper bound LP with actual generation)
"""),
    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

import pandas as pd
from src.preprocessing import get_clean_data
from src.battery_simulator import BatteryConfig
from src.strategies import scale_to_site, compare_dispatch_strategies

clean_df = get_clean_data()
preds_df = pd.read_parquet('../reports/test_predictions.parquet')

# Align chronological test slice
test_slice = clean_df.loc[preds_df.index]
site_df = scale_to_site(test_slice, peak_load_mw=5.0, solar_capacity_mw=5.0)

# Scaled solar forecast
solar_fc_scaled = preds_df['model_lightgbm'] * site_df['solar_scale_factor']

battery_cfg = BatteryConfig(
    energy_capacity_mwh=10.0,
    max_charge_power_mw=2.5,
    max_discharge_power_mw=2.5,
    round_trip_efficiency=0.88,
    soc_min=0.10,
    soc_max=0.90,
    grid_import_limit_mw=6.0,
    grid_export_limit_mw=5.0,
    retail_adder_eur_mwh=20.0,
    export_ratio=1.0,
    degradation_cost_eur_mwh=0.0
)

results, summary_df = compare_dispatch_strategies(
    site_df=site_df,
    solar_forecast_mw=solar_fc_scaled,
    load_forecast_mw=site_df['load_forecast_mw'],
    battery_cfg=battery_cfg
)
display(summary_df)
"""),
    nbf.v4.new_markdown_cell("""### 1. Invariant & Constraint Verification
- Energy balance residuals must be ~0.
- Constraint violations must be 0.
"""),
    nbf.v4.new_code_cell("""for name, r in results.items():
    print(f"Strategy: {r.strategy_name:<32} | Violations: {r.constraint_violations} | Max Residual: {r.max_energy_balance_residual:.6f} MW")
    assert r.constraint_violations == 0
    assert r.max_energy_balance_residual < 1e-4
print("All physical and operational constraints passed with zero violations!")
""")
]

with open(nb_dir / "04_battery_simulation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb4, f)

# -------------------------------------------------------------
# Notebook 05: Model Evaluation & Sensitivity
# -------------------------------------------------------------
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("""# 05 - Model Evaluation & Sensitivity Analysis
## Economic Impact of Forecast Quality & Sensitivity Sweeps

This notebook examines the value of forecast accuracy, evaluating how much of the theoretical savings gap is captured by the forecast-informed dispatch and conducting multi-dimensional sensitivity sweeps.
"""),
    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

import json
import pandas as pd

with open('../reports/sensitivity_results.json', 'r') as f:
    sens = json.load(f)

print("Forecast Quality Impact on Electricity Procurement Cost:")
df_qual = pd.DataFrame(sens['forecast_quality'])
display(df_qual)
"""),
    nbf.v4.new_markdown_cell("""### 1. Value of Forecast & Savings Gap Analysis"""),
    nbf.v4.new_code_cell("""perf_cost = df_qual.loc[df_qual['forecast_source'] == 'Perfect Foresight', 'simulated_total_cost_eur'].iloc[0]
lgb_cost = df_qual.loc[df_qual['forecast_source'] == 'LightGBM Model', 'simulated_total_cost_eur'].iloc[0]
diff_eur = lgb_cost - perf_cost
print(f"Cost penalty from forecast error (LightGBM vs Perfect Foresight): {diff_eur:,.2f} EUR")
"""),
    nbf.v4.new_markdown_cell("""### 2. Battery Storage Capacity Sensitivity"""),
    nbf.v4.new_code_cell("""df_cap = pd.DataFrame(sens['capacity'])
print("Capacity Sensitivity Table:")
display(df_cap)
""")
]

with open(nb_dir / "05_model_evaluation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb5, f)

print("Successfully wrote all 5 notebooks to notebooks/!")

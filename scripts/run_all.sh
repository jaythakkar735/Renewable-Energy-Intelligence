#!/usr/bin/env bash
# ==============================================================================
# Renewable Energy Intelligence - Full Pipeline Execution Script
# Runs end-to-end audit, preprocessing, model training, battery dispatch,
# sensitivity sweeps, test suite, and notebook re-execution.
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="/home/asus/anaconda3/envs/renewable-ei/bin/python"
PYTEST_BIN="/home/asus/anaconda3/envs/renewable-ei/bin/pytest"
JUPYTER_BIN="/home/asus/anaconda3/envs/renewable-ei/bin/jupyter"

echo "======================================================================"
echo " Renewable Energy Intelligence: End-to-End Upgrade Execution"
echo " Project Root: $PROJECT_ROOT"
echo " Python:       $($PYTHON_BIN --version)"
echo "======================================================================"

echo ""
echo "[Step 1/6] Running Data Preprocessing and Telemetry Audit..."
$PYTHON_BIN -c "
from src.preprocessing import get_clean_data
clean_df = get_clean_data(force_recompute=True)
print(f'Clean time series ready: {len(clean_df):,} hourly rows.')
"

echo ""
echo "[Step 2/6] Training Baselines, Ridge, Random Forest, LightGBM & Conformal Models..."
$PYTHON_BIN -c "
from src.preprocessing import get_clean_data
from src.forecasting import train_and_evaluate_models
clean_df = get_clean_data()
preds, metrics, leaderboard = train_and_evaluate_models(clean_df, force_retrain=True)
print('Forecasting models successfully trained and evaluated.')
"

echo ""
echo "[Step 3/6] Running 5-Strategy Dispatch Simulation & Sensitivity Sweeps..."
$PYTHON_BIN -c "
import pandas as pd
from src.preprocessing import get_clean_data
from src.sensitivity import run_sensitivity_sweeps
clean_df = get_clean_data()
preds_df = pd.read_parquet('reports/test_predictions.parquet')
sens = run_sensitivity_sweeps(clean_df, preds_df)
print('Dispatch simulation and sensitivity sweeps complete.')
"

echo ""
echo "[Step 4/6] Running PyTest Verification Suite..."
$PYTEST_BIN -v

echo ""
echo "[Step 5/6] Re-executing Jupyter Notebooks In-Place..."
for nb in notebooks/*.ipynb; do
    echo "  Executing $nb..."
    $JUPYTER_BIN nbconvert --to notebook --execute --inplace "$nb"
done

echo ""
echo "[Step 6/6] Pipeline Execution Finished Successfully!"
echo "To launch the interactive dashboard:"
echo "  /home/asus/anaconda3/envs/renewable-ei/bin/streamlit run app.py --server.port 8502"
echo "======================================================================"

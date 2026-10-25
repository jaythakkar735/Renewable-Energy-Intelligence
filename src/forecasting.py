"""Forecasting models, baselines, training pipelines, and conformal prediction."""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.config import load_config, get_project_root
from src.features import build_forecasting_features
from src.evaluation import compute_forecast_metrics, compute_coverage
from src.utils import setup_logger, timer

logger = setup_logger(__name__)


class ConformalIntervalPredictor:
    """Split-conformal prediction interval generator calibrated on chronological validation data."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_val: float = 0.0

    def calibrate(self, y_val: np.ndarray, y_pred_val: np.ndarray) -> "ConformalIntervalPredictor":
        residuals = np.abs(np.asarray(y_val) - np.asarray(y_pred_val))
        n = len(residuals)
        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_val = float(np.quantile(residuals, q_level))
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        clearsky_proxy: Optional[np.ndarray] = None,
        capacity_mw: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        low = np.clip(y_pred - self.q_val, 0.0, None)
        high = y_pred + self.q_val
        if clearsky_proxy is not None and capacity_mw is not None:
            # Physical upper ceiling: clear-sky irradiance proxy scaled to capacity + headroom
            ceiling = np.maximum(0.0, clearsky_proxy * (capacity_mw * 1.15))
            high = np.minimum(high, ceiling)
            low = np.minimum(low, high)
        return low, high


def split_train_val_test(
    features_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits feature DataFrame chronologically into Train, Validation, and Test subsets."""
    if config is None:
        config = load_config()

    train_end = pd.to_datetime(config["splits"]["train_end"])
    val_end = pd.to_datetime(config["splits"]["val_end"])
    test_end = pd.to_datetime(config["splits"]["test_end"])

    train_df = features_df.loc[:train_end].copy()
    val_df = features_df.loc[train_end:val_end].iloc[1:].copy()
    test_df = features_df.loc[val_end:test_end].iloc[1:].copy()

    logger.info(
        f"Chronological split sizes: Train={len(train_df):,} ({train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')}), "
        f"Val={len(val_df):,} ({val_df.index.min().strftime('%Y-%m-%d')} to {val_df.index.max().strftime('%Y-%m-%d')}), "
        f"Test={len(test_df):,} ({test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})"
    )

    return train_df, val_df, test_df


def train_and_evaluate_models(
    clean_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    force_retrain: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Trains baselines, Ridge, Random Forest, and LightGBM models.

    Computes holdout test set performance, conformal intervals,
    and returns (predictions_df, metrics_dict, leaderboard_df).
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_col = config["forecasting"]["target"]
    capacity_mw = float(clean_df[target_col].max())

    # Build features
    feat_df = build_forecasting_features(
        clean_df,
        target_col=target_col,
        min_lag_hours=24,
        latitude_deg=config["data"]["latitude"],
        longitude_deg=config["data"]["longitude"],
    )

    train_df, val_df, test_df = split_train_val_test(feat_df, config)

    feature_cols = [c for c in feat_df.columns if c != "target"]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    test_daylight = test_df["is_daylight"].values.astype(bool)

    predictions = pd.DataFrame(index=test_df.index)
    predictions["actual"] = y_test
    predictions["clearsky_proxy"] = test_df["clearsky_proxy"].values
    predictions["is_daylight"] = test_daylight

    # 1. Baselines
    # Persistence 24h
    y_pred_persist = np.clip(test_df["target_lag_24h"].values, 0.0, None)
    predictions["baseline_persistence_24h"] = y_pred_persist

    # Seasonal Naive 168h (same hour last week)
    y_pred_sn168 = np.clip(test_df["target_lag_168h"].values, 0.0, None)
    predictions["baseline_seasonal_naive_168h"] = y_pred_sn168

    persist_metrics = compute_forecast_metrics(y_test, y_pred_persist, capacity_mw, test_daylight)
    sn168_metrics = compute_forecast_metrics(y_test, y_pred_sn168, capacity_mw, test_daylight)

    best_baseline_mae = min(persist_metrics["mae_all"], sn168_metrics["mae_all"])

    leaderboard = [
        {
            "model": "Persistence (24h)",
            "type": "Baseline",
            "mae_all": persist_metrics["mae_all"],
            "rmse_all": persist_metrics["rmse_all"],
            "nmae_all_pct": persist_metrics["nmae_all_pct"],
            "mae_daylight": persist_metrics["mae_daylight"],
            "skill_score": round(1.0 - (persist_metrics["mae_all"] / best_baseline_mae), 4),
            "train_time_sec": 0.0,
            "inference_time_ms": 0.0,
            "size_kb": 0.0,
        },
        {
            "model": "Seasonal Naive (168h)",
            "type": "Baseline",
            "mae_all": sn168_metrics["mae_all"],
            "rmse_all": sn168_metrics["rmse_all"],
            "nmae_all_pct": sn168_metrics["nmae_all_pct"],
            "mae_daylight": sn168_metrics["mae_daylight"],
            "skill_score": round(1.0 - (sn168_metrics["mae_all"] / best_baseline_mae), 4),
            "train_time_sec": 0.0,
            "inference_time_ms": 0.0,
            "size_kb": 0.0,
        },
    ]

    all_metrics = {
        "Persistence_24h": persist_metrics,
        "Seasonal_Naive_168h": sn168_metrics,
    }

    # 2. Ridge Regression
    scaler_path = models_dir / "scaler.joblib"
    ridge_path = models_dir / "ridge_model.joblib"

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)

    ridge = Ridge(alpha=100.0, random_state=42)
    with timer() as t_train:
        ridge.fit(X_train_scaled, y_train)
    train_time_ridge = t_train["elapsed_seconds"]

    joblib.dump(ridge, ridge_path)
    size_ridge_kb = os.path.getsize(ridge_path) / 1024.0

    start_inf = time.perf_counter()
    y_pred_ridge = np.clip(ridge.predict(X_test_scaled), 0.0, None)
    inf_time_ridge_ms = (time.perf_counter() - start_inf) * 1000.0

    predictions["model_ridge"] = y_pred_ridge
    ridge_metrics = compute_forecast_metrics(y_test, y_pred_ridge, capacity_mw, test_daylight, best_baseline_mae)
    all_metrics["Ridge"] = ridge_metrics
    leaderboard.append({
        "model": "Ridge Regression",
        "type": "Linear ML",
        "mae_all": ridge_metrics["mae_all"],
        "rmse_all": ridge_metrics["rmse_all"],
        "nmae_all_pct": ridge_metrics["nmae_all_pct"],
        "mae_daylight": ridge_metrics["mae_daylight"],
        "skill_score": ridge_metrics["skill_score_vs_baseline"],
        "train_time_sec": round(train_time_ridge, 3),
        "inference_time_ms": round(inf_time_ridge_ms, 2),
        "size_kb": round(size_ridge_kb, 1),
    })

    # 3. Random Forest (constrained size)
    rf_path = models_dir / "random_forest_model.joblib"
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        max_features="sqrt",
        random_state=42,
        n_jobs=2,
    )
    with timer() as t_train:
        rf.fit(X_train, y_train)
    train_time_rf = t_train["elapsed_seconds"]

    joblib.dump(rf, rf_path)
    size_rf_kb = os.path.getsize(rf_path) / 1024.0

    start_inf = time.perf_counter()
    y_pred_rf = np.clip(rf.predict(X_test), 0.0, None)
    inf_time_rf_ms = (time.perf_counter() - start_inf) * 1000.0

    predictions["model_random_forest"] = y_pred_rf
    rf_metrics = compute_forecast_metrics(y_test, y_pred_rf, capacity_mw, test_daylight, best_baseline_mae)
    all_metrics["Random_Forest"] = rf_metrics
    leaderboard.append({
        "model": "Random Forest",
        "type": "Ensemble ML",
        "mae_all": rf_metrics["mae_all"],
        "rmse_all": rf_metrics["rmse_all"],
        "nmae_all_pct": rf_metrics["nmae_all_pct"],
        "mae_daylight": rf_metrics["mae_daylight"],
        "skill_score": rf_metrics["skill_score_vs_baseline"],
        "train_time_sec": round(train_time_rf, 3),
        "inference_time_ms": round(inf_time_rf_ms, 2),
        "size_kb": round(size_rf_kb, 1),
    })

    # 4. LightGBM (CPU-first, fast gradient boosting)
    lgb_path = models_dir / "lightgbm_model.joblib"
    lgb_model = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    with timer() as t_train:
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )
    train_time_lgb = t_train["elapsed_seconds"]

    joblib.dump(lgb_model, lgb_path)
    size_lgb_kb = os.path.getsize(lgb_path) / 1024.0

    start_inf = time.perf_counter()
    y_pred_lgb = np.clip(lgb_model.predict(X_test), 0.0, None)
    inf_time_lgb_ms = (time.perf_counter() - start_inf) * 1000.0

    predictions["model_lightgbm"] = y_pred_lgb
    lgb_metrics = compute_forecast_metrics(y_test, y_pred_lgb, capacity_mw, test_daylight, best_baseline_mae)
    all_metrics["LightGBM"] = lgb_metrics
    leaderboard.append({
        "model": "LightGBM",
        "type": "Gradient Boosting",
        "mae_all": lgb_metrics["mae_all"],
        "rmse_all": lgb_metrics["rmse_all"],
        "nmae_all_pct": lgb_metrics["nmae_all_pct"],
        "mae_daylight": lgb_metrics["mae_daylight"],
        "skill_score": lgb_metrics["skill_score_vs_baseline"],
        "train_time_sec": round(train_time_lgb, 3),
        "inference_time_ms": round(inf_time_lgb_ms, 2),
        "size_kb": round(size_lgb_kb, 1),
    })

    # 5. Conformal Prediction Intervals on Best Model (LightGBM)
    # Calibrate on validation set
    y_val_pred_lgb = np.clip(lgb_model.predict(X_val), 0.0, None)
    conformal_90 = ConformalIntervalPredictor(alpha=0.10)
    conformal_90.calibrate(y_val, y_val_pred_lgb)

    low_90, high_90 = conformal_90.predict_intervals(
        y_pred_lgb,
        clearsky_proxy=test_df["clearsky_proxy"].values,
        capacity_mw=capacity_mw,
    )
    predictions["lightgbm_lower_90"] = low_90
    predictions["lightgbm_upper_90"] = high_90

    coverage_90 = compute_coverage(y_test, low_90, high_90)
    all_metrics["Conformal_90_Coverage"] = coverage_90

    conformal_80 = ConformalIntervalPredictor(alpha=0.20)
    conformal_80.calibrate(y_val, y_val_pred_lgb)
    low_80, high_80 = conformal_80.predict_intervals(
        y_pred_lgb,
        clearsky_proxy=test_df["clearsky_proxy"].values,
        capacity_mw=capacity_mw,
    )
    predictions["lightgbm_lower_80"] = low_80
    predictions["lightgbm_upper_80"] = high_80
    coverage_80 = compute_coverage(y_test, low_80, high_80)
    all_metrics["Conformal_80_Coverage"] = coverage_80

    # Save artifacts
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("mae_all").reset_index(drop=True)
    leaderboard_df.to_csv(reports_dir / "model_leaderboard.csv", index=False)
    predictions.to_parquet(reports_dir / "test_predictions.parquet", index=True)

    import json
    with open(reports_dir / "forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Model training and evaluation successfully completed!")
    logger.info("\n" + leaderboard_df.to_string())

    return predictions, all_metrics, leaderboard_df

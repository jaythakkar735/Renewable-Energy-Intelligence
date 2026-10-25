"""Forecasting models, baselines, training pipelines, conformal prediction, and quantile models."""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.config import load_config, get_project_root
from src.features import build_forecasting_features
from src.evaluation import compute_forecast_metrics, compute_coverage, compute_conformal_breakdown
from src.utils import setup_logger, timer

logger = setup_logger(__name__)


class ConformalIntervalPredictor:
    """Split-conformal prediction interval generator calibrated on chronological validation data."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_val: float = 0.0

    def calibrate(
        self,
        y_val: np.ndarray,
        y_pred_val: np.ndarray,
        daylight_mask: Optional[np.ndarray] = None,
    ) -> "ConformalIntervalPredictor":
        """Calibrates non-conformity score quantile on validation residuals.

        If daylight_mask is provided, calibrates on daylight residuals to avoid
        under-estimating the uncertainty during active generation hours.
        """
        y_v = np.asarray(y_val)
        y_p = np.asarray(y_pred_val)
        if daylight_mask is not None:
            mask = np.asarray(daylight_mask, dtype=bool)
            y_v = y_v[mask]
            y_p = y_p[mask]

        residuals = np.abs(y_v - y_p)
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_val = float(np.quantile(residuals, q_level))
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        elevation_deg: np.ndarray,
        clearsky_proxy: Optional[np.ndarray] = None,
        capacity_proxy: Optional[np.ndarray | float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates conformal lower and upper bounds with hard night zeros and physical ceilings."""
        elev = np.asarray(elevation_deg)
        y_p = np.asarray(y_pred)

        low = np.clip(y_p - self.q_val, 0.0, None)
        high = y_p + self.q_val

        # Hard physical ceiling based on capacity and clear sky
        if capacity_proxy is not None:
            cap = np.asarray(capacity_proxy)
            high = np.minimum(high, cap * 1.15)

        if clearsky_proxy is not None and capacity_proxy is not None:
            cap = np.asarray(capacity_proxy)
            ceiling = np.maximum(0.0, np.asarray(clearsky_proxy) * (cap * 1.15))
            # Only apply ceiling when clear sky proxy is meaningful
            high = np.where(clearsky_proxy > 0.05, np.minimum(high, ceiling), high)

        low = np.minimum(low, high)

        # Enforce night zeros: when solar elevation <= 0, interval is exactly [0, 0]
        night_mask = elev <= 0.0
        low = np.where(night_mask, 0.0, low)
        high = np.where(night_mask, 0.0, high)

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
    """Trains baselines and ML models with cutoff-safety and capacity normalization.

    Baselines:
      1. Last-Available-Day (lag 24 for h<12, lag 48 for h>=12)
      2. Seasonal Naive 48h (same hour 2 days ago)
      3. Seasonal Naive 168h (same hour 7 days ago)

    Models:
      1. Ridge Regression (capacity-normalized target)
      2. Random Forest (capacity-normalized target)
      3. LightGBM (capacity-normalized target + early stopping on validation)
      4. LightGBM Quantile Regressors (P10, P50, P90)
      5. Conformal Prediction Intervals (80% and 90% with night zeros)

    Returns:
      (predictions_df, metrics_dict, leaderboard_df)
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_col = config["forecasting"]["target"]

    # Build features
    feat_df = build_forecasting_features(
        clean_df,
        target_col=target_col,
        latitude_deg=config["data"]["latitude"],
        longitude_deg=config["data"]["longitude"],
    )

    train_df, val_df, test_df = split_train_val_test(feat_df, config)

    # Candidate feature columns for ML training
    non_feature_cols = {
        "target", "target_normalized", "is_telemetry_gap",
        "target_lag_last_available", "target_lag_24h", "target_lag_48h", "target_lag_168h",
        "target_rolling_mean_24h", "target_rolling_std_24h", "target_rolling_max_24h", "target_rolling_mean_168h",
    }
    feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]

    logger.info(f"Model feature columns ({len(feature_cols)}): {feature_cols}")

    # Exclude telemetry gap rows from training loss
    train_valid_mask = ~train_df["is_telemetry_gap"].values
    val_valid_mask = ~val_df["is_telemetry_gap"].values
    test_valid_mask = ~test_df["is_telemetry_gap"].values

    X_train = train_df.loc[train_valid_mask, feature_cols].values
    y_train_norm = train_df.loc[train_valid_mask, "target_normalized"].values
    y_train_raw = train_df.loc[train_valid_mask, "target"].values

    X_val = val_df[feature_cols].values
    y_val_norm = val_df["target_normalized"].values
    y_val_raw = val_df["target"].values
    val_cap_proxy = val_df["capacity_proxy_mw"].values
    val_elev = val_df["solar_elevation_deg"].values

    X_test = test_df[feature_cols].values
    y_test_raw = test_df["target"].values
    test_cap_proxy = test_df["capacity_proxy_mw"].values
    test_elev = test_df["solar_elevation_deg"].values
    test_daylight = test_df["is_daylight"].values.astype(bool)
    test_clearsky = test_df["clearsky_proxy"].values

    capacity_mw = float(np.max(test_cap_proxy))

    predictions = pd.DataFrame(index=test_df.index)
    predictions["actual"] = y_test_raw
    predictions["capacity_proxy_mw"] = test_cap_proxy
    predictions["solar_elevation_deg"] = test_elev
    predictions["clearsky_proxy"] = test_clearsky
    predictions["is_daylight"] = test_daylight
    predictions["is_telemetry_gap"] = test_df["is_telemetry_gap"].values

    def apply_physical_rules(pred_norm: np.ndarray, cap_proxy: np.ndarray, elev: np.ndarray) -> np.ndarray:
        """Converts normalized predictions back to MW and enforces physical constraints:
        1. Non-negativity and upper ceiling [0, C_t * 1.15]
        2. Hard night zeros when solar_elevation_deg <= 0.
        """
        mw = np.clip(pred_norm, 0.0, 1.15) * cap_proxy
        mw = np.where(elev <= 0.0, 0.0, mw)
        return np.clip(mw, 0.0, cap_proxy * 1.15).astype(np.float32)

    # -------------------------------------------------------------
    # 1. BASELINES (Leakage-Safe, Night Zeros Enforced)
    # -------------------------------------------------------------
    # Baseline A: Last-Available-Day (lag 24 for h<12, lag 48 for h>=12)
    lad_raw = test_df["target_lag_last_available"].values
    lad_pred = np.where(test_elev <= 0.0, 0.0, np.clip(lad_raw, 0.0, test_cap_proxy * 1.15))
    predictions["baseline_last_available_day"] = lad_pred.astype(np.float32)

    # Baseline B: Seasonal Naive 48h (strictly lag 48 for all hours)
    sn48_raw = test_df["target_lag_48h"].values
    sn48_pred = np.where(test_elev <= 0.0, 0.0, np.clip(sn48_raw, 0.0, test_cap_proxy * 1.15))
    predictions["baseline_seasonal_naive_48h"] = sn48_pred.astype(np.float32)

    # Baseline C: Seasonal Naive 168h (strictly lag 168 for all hours)
    sn168_raw = test_df["target_lag_168h"].values
    sn168_pred = np.where(test_elev <= 0.0, 0.0, np.clip(sn168_raw, 0.0, test_cap_proxy * 1.15))
    predictions["baseline_seasonal_naive_168h"] = sn168_pred.astype(np.float32)

    lad_metrics = compute_forecast_metrics(
        y_test_raw, lad_pred, capacity_mw, test_daylight, valid_mask=test_valid_mask
    )
    sn48_metrics = compute_forecast_metrics(
        y_test_raw, sn48_pred, capacity_mw, test_daylight, valid_mask=test_valid_mask
    )
    sn168_metrics = compute_forecast_metrics(
        y_test_raw, sn168_pred, capacity_mw, test_daylight, valid_mask=test_valid_mask
    )

    best_baseline_mae = min(lad_metrics["mae_all"], sn48_metrics["mae_all"], sn168_metrics["mae_all"])

    leaderboard = [
        {
            "model": "Last-Available-Day",
            "type": "Baseline (Auction Cutoff)",
            "mae_all": lad_metrics["mae_all"],
            "rmse_all": lad_metrics["rmse_all"],
            "nmae_all_pct": lad_metrics["nmae_all_pct"],
            "mae_daylight": lad_metrics["mae_daylight"],
            "skill_score": round(1.0 - (lad_metrics["mae_all"] / best_baseline_mae), 4),
            "train_time_sec": 0.0,
            "inference_time_ms": 0.0,
            "size_kb": 0.0,
        },
        {
            "model": "Seasonal Naive (48h)",
            "type": "Baseline (Lag 48h)",
            "mae_all": sn48_metrics["mae_all"],
            "rmse_all": sn48_metrics["rmse_all"],
            "nmae_all_pct": sn48_metrics["nmae_all_pct"],
            "mae_daylight": sn48_metrics["mae_daylight"],
            "skill_score": round(1.0 - (sn48_metrics["mae_all"] / best_baseline_mae), 4),
            "train_time_sec": 0.0,
            "inference_time_ms": 0.0,
            "size_kb": 0.0,
        },
        {
            "model": "Seasonal Naive (168h)",
            "type": "Baseline (Lag 168h)",
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
        "Last_Available_Day": lad_metrics,
        "Seasonal_Naive_48h": sn48_metrics,
        "Seasonal_Naive_168h": sn168_metrics,
    }

    # -------------------------------------------------------------
    # 2. RIDGE REGRESSION (Capacity-Normalized)
    # -------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "scaler.joblib")

    ridge = Ridge(alpha=100.0, random_state=42)
    with timer() as t_train:
        ridge.fit(X_train_scaled, y_train_norm)
    train_time_ridge = t_train["elapsed_seconds"]

    joblib.dump(ridge, models_dir / "ridge_model.joblib")
    size_ridge_kb = os.path.getsize(models_dir / "ridge_model.joblib") / 1024.0

    start_inf = time.perf_counter()
    y_pred_ridge_norm = ridge.predict(X_test_scaled)
    y_pred_ridge = apply_physical_rules(y_pred_ridge_norm, test_cap_proxy, test_elev)
    inf_time_ridge_ms = (time.perf_counter() - start_inf) * 1000.0

    predictions["model_ridge"] = y_pred_ridge
    ridge_metrics = compute_forecast_metrics(
        y_test_raw, y_pred_ridge, capacity_mw, test_daylight, best_baseline_mae, valid_mask=test_valid_mask
    )
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

    # -------------------------------------------------------------
    # 3. RANDOM FOREST (Capacity-Normalized)
    # -------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        max_features="sqrt",
        random_state=42,
        n_jobs=2,
    )
    with timer() as t_train:
        rf.fit(X_train, y_train_norm)
    train_time_rf = t_train["elapsed_seconds"]

    joblib.dump(rf, models_dir / "random_forest_model.joblib")
    size_rf_kb = os.path.getsize(models_dir / "random_forest_model.joblib") / 1024.0

    start_inf = time.perf_counter()
    y_pred_rf_norm = rf.predict(X_test)
    y_pred_rf = apply_physical_rules(y_pred_rf_norm, test_cap_proxy, test_elev)
    inf_time_rf_ms = (time.perf_counter() - start_inf) * 1000.0

    predictions["model_random_forest"] = y_pred_rf
    rf_metrics = compute_forecast_metrics(
        y_test_raw, y_pred_rf, capacity_mw, test_daylight, best_baseline_mae, valid_mask=test_valid_mask
    )
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

    # -------------------------------------------------------------
    # 4. LIGHTGBM (Capacity-Normalized + Early Stopping on Val)
    # -------------------------------------------------------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    X_val_valid = X_val[val_valid_mask]
    y_val_norm_valid = y_val_norm[val_valid_mask]

    with timer() as t_train:
        lgb_model.fit(
            X_train,
            y_train_norm,
            eval_set=[(X_val_valid, y_val_norm_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )
    train_time_lgb = t_train["elapsed_seconds"]

    joblib.dump(lgb_model, models_dir / "lightgbm_model.joblib")
    size_lgb_kb = os.path.getsize(models_dir / "lightgbm_model.joblib") / 1024.0

    start_inf = time.perf_counter()
    y_pred_lgb_norm = lgb_model.predict(X_test)
    y_pred_lgb = apply_physical_rules(y_pred_lgb_norm, test_cap_proxy, test_elev)
    inf_time_lgb_ms = (time.perf_counter() - start_inf) * 1000.0

    predictions["model_lightgbm"] = y_pred_lgb
    lgb_metrics = compute_forecast_metrics(
        y_test_raw, y_pred_lgb, capacity_mw, test_daylight, best_baseline_mae, valid_mask=test_valid_mask
    )
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

    # -------------------------------------------------------------
    # 5. LIGHTGBM QUANTILE REGRESSORS (P10, P50, P90)
    # -------------------------------------------------------------
    logger.info("Training LightGBM quantile regressors (P10, P50, P90)...")
    for alpha in [0.10, 0.50, 0.90]:
        q_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        q_model.fit(X_train, y_train_norm)
        q_name = f"p{int(alpha * 100)}"
        joblib.dump(q_model, models_dir / f"lightgbm_{q_name}.joblib")
        pred_q_norm = q_model.predict(X_test)
        pred_q_mw = apply_physical_rules(pred_q_norm, test_cap_proxy, test_elev)
        predictions[f"lightgbm_{q_name}"] = pred_q_mw

    # -------------------------------------------------------------
    # 6. CONFORMAL PREDICTION INTERVALS (Night Zeros Enforced)
    # -------------------------------------------------------------
    # Predict on validation set to calibrate
    y_val_pred_lgb_norm = lgb_model.predict(X_val)
    y_val_pred_lgb = apply_physical_rules(y_val_pred_lgb_norm, val_cap_proxy, val_elev)
    val_daylight = val_df["is_daylight"].values.astype(bool)

    # 90% Conformal (alpha = 0.10)
    conformal_90 = ConformalIntervalPredictor(alpha=0.10)
    conformal_90.calibrate(y_val_raw, y_val_pred_lgb, daylight_mask=val_daylight)
    joblib.dump(conformal_90, models_dir / "conformal_90.joblib")

    low_90, high_90 = conformal_90.predict_intervals(
        y_pred_lgb,
        elevation_deg=test_elev,
        clearsky_proxy=test_clearsky,
        capacity_proxy=test_cap_proxy,
    )
    predictions["lightgbm_lower_90"] = low_90.astype(np.float32)
    predictions["lightgbm_upper_90"] = high_90.astype(np.float32)

    cov_90_breakdown = compute_conformal_breakdown(
        predictions["actual"],
        predictions["lightgbm_lower_90"],
        predictions["lightgbm_upper_90"],
        predictions["is_daylight"],
        valid_mask=pd.Series(test_valid_mask, index=test_df.index),
    )
    all_metrics["Conformal_90_Coverage"] = cov_90_breakdown

    # 80% Conformal (alpha = 0.20)
    conformal_80 = ConformalIntervalPredictor(alpha=0.20)
    conformal_80.calibrate(y_val_raw, y_val_pred_lgb, daylight_mask=val_daylight)
    joblib.dump(conformal_80, models_dir / "conformal_80.joblib")

    low_80, high_80 = conformal_80.predict_intervals(
        y_pred_lgb,
        elevation_deg=test_elev,
        clearsky_proxy=test_clearsky,
        capacity_proxy=test_cap_proxy,
    )
    predictions["lightgbm_lower_80"] = low_80.astype(np.float32)
    predictions["lightgbm_upper_80"] = high_80.astype(np.float32)

    cov_80_breakdown = compute_conformal_breakdown(
        predictions["actual"],
        predictions["lightgbm_lower_80"],
        predictions["lightgbm_upper_80"],
        predictions["is_daylight"],
        valid_mask=pd.Series(test_valid_mask, index=test_df.index),
    )
    all_metrics["Conformal_80_Coverage"] = cov_80_breakdown

    # -------------------------------------------------------------
    # 7. SAVE ARTIFACTS
    # -------------------------------------------------------------
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("mae_all").reset_index(drop=True)
    leaderboard_df.to_csv(reports_dir / "model_leaderboard.csv", index=False)
    predictions.to_parquet(reports_dir / "test_predictions.parquet", index=True)

    with open(reports_dir / "forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Model training and evaluation successfully completed!")
    logger.info("\n" + leaderboard_df.to_string())

    return predictions, all_metrics, leaderboard_df

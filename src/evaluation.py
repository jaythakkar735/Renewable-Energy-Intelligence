"""Forecasting metrics, skill score computation, residual diagnostics, and conformal coverage."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


def compute_forecast_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    capacity_mw: Optional[float] = None,
    daylight_mask: Optional[np.ndarray | pd.Series] = None,
    baseline_mae: Optional[float] = None,
    valid_mask: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for energy forecasting.

    Args:
        y_true: Ground truth observed values.
        y_pred: Forecast predictions.
        capacity_mw: Peak capacity for normalization. If None, uses max(y_true).
        daylight_mask: Boolean array where True indicates daylight hours.
        baseline_mae: MAE of the reference baseline to compute skill score.
        valid_mask: Optional boolean array to exclude telemetry gaps or invalid periods.

    Returns:
        Dictionary containing MAE, RMSE, nMAE (%), Bias, Skill Score,
        and daylight-only vs all-hours metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("Length of y_true and y_pred must match.")

    # Physical non-negativity constraint on solar power
    y_pred_arr = np.clip(y_pred_arr, 0.0, None)

    if valid_mask is not None:
        v_mask = np.asarray(valid_mask, dtype=bool)
        y_true_arr = y_true_arr[v_mask]
        y_pred_arr = y_pred_arr[v_mask]
        if daylight_mask is not None:
            daylight_mask = np.asarray(daylight_mask, dtype=bool)[v_mask]

    errors = y_pred_arr - y_true_arr
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    # All-hours metrics
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    bias = float(np.mean(errors))

    cap = capacity_mw if capacity_mw is not None else float(np.max(y_true_arr))
    nmae_pct = (mae / cap) * 100.0 if cap > 0 else 0.0
    nrmse_pct = (rmse / cap) * 100.0 if cap > 0 else 0.0

    skill_score = None
    if baseline_mae is not None and baseline_mae > 0:
        skill_score = float(1.0 - (mae / baseline_mae))

    metrics: Dict[str, Any] = {
        "count_all_hours": int(len(y_true_arr)),
        "mae_all": round(mae, 3),
        "rmse_all": round(rmse, 3),
        "nmae_all_pct": round(nmae_pct, 2),
        "nrmse_all_pct": round(nrmse_pct, 2),
        "bias_all": round(bias, 3),
        "skill_score_vs_baseline": round(skill_score, 4) if skill_score is not None else None,
        "capacity_mw": round(cap, 2),
    }

    # Daylight-only metrics
    if daylight_mask is not None:
        mask = np.asarray(daylight_mask, dtype=bool)
        if np.any(mask):
            y_t_day = y_true_arr[mask]
            y_p_day = y_pred_arr[mask]
            err_day = y_p_day - y_t_day
            mae_day = float(np.mean(np.abs(err_day)))
            rmse_day = float(np.sqrt(np.mean(err_day ** 2)))
            bias_day = float(np.mean(err_day))
            nmae_day = (mae_day / cap) * 100.0 if cap > 0 else 0.0

            metrics.update({
                "count_daylight_hours": int(np.sum(mask)),
                "mae_daylight": round(mae_day, 3),
                "rmse_daylight": round(rmse_day, 3),
                "nmae_daylight_pct": round(nmae_day, 2),
                "bias_daylight": round(bias_day, 3),
            })

    return metrics


def compute_error_by_hour(
    y_true: pd.Series,
    y_pred: pd.Series,
    valid_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Computes mean error, MAE, RMSE, and solar generation grouped by hour (0-23)."""
    df = pd.DataFrame({
        "actual": y_true,
        "pred": np.clip(y_pred, 0.0, None),
    })
    if valid_mask is not None:
        df = df.loc[valid_mask]

    df["error"] = df["pred"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2
    df["hour"] = df.index.hour

    grouped = df.groupby("hour").agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        mean_actual=("actual", "mean"),
        mean_pred=("pred", "mean"),
    ).reset_index()

    return grouped


def compute_conformal_breakdown(
    y_true: pd.Series,
    lower_bound: pd.Series,
    upper_bound: pd.Series,
    is_daylight: pd.Series,
    valid_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Computes conformal coverage disaggregated by season and by Day vs Night.

    Returns overall coverage, day/night breakdown, and seasonal breakdown.
    """
    df = pd.DataFrame({
        "actual": y_true,
        "lower": lower_bound,
        "upper": upper_bound,
        "daylight": is_daylight.astype(bool),
    })
    if valid_mask is not None:
        df = df.loc[valid_mask]

    df["covered"] = (df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    # Month to season
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    df["season"] = df.index.month.map(season_map)

    overall_cov = float(df["covered"].mean()) * 100.0
    overall_width = float(df["width"].mean())

    # Day vs Night
    day_df = df[df["daylight"]]
    night_df = df[~df["daylight"]]

    day_cov = float(day_df["covered"].mean()) * 100.0 if len(day_df) > 0 else 0.0
    day_width = float(day_df["width"].mean()) if len(day_df) > 0 else 0.0

    night_cov = float(night_df["covered"].mean()) * 100.0 if len(night_df) > 0 else 0.0
    night_width = float(night_df["width"].mean()) if len(night_df) > 0 else 0.0

    # Seasons
    seasons = {}
    for s in ["Winter", "Spring", "Summer", "Autumn"]:
        s_df = df[df["season"] == s]
        if len(s_df) > 0:
            seasons[s] = {
                "coverage_pct": round(float(s_df["covered"].mean()) * 100.0, 2),
                "mean_width_mw": round(float(s_df["width"].mean()), 2),
                "count_hours": int(len(s_df)),
            }

    return {
        "overall_coverage_pct": round(overall_cov, 2),
        "overall_mean_width_mw": round(overall_width, 2),
        "daylight_coverage_pct": round(day_cov, 2),
        "daylight_mean_width_mw": round(day_width, 2),
        "night_coverage_pct": round(night_cov, 2),
        "night_mean_width_mw": round(night_width, 2),
        "seasonal": seasons,
    }


def compute_coverage(
    y_true: np.ndarray | pd.Series,
    lower_bound: np.ndarray | pd.Series,
    upper_bound: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """Computes empirical coverage and interval sharpness for prediction intervals."""
    y = np.asarray(y_true)
    low = np.asarray(lower_bound)
    up = np.asarray(upper_bound)

    covered = (y >= low) & (y <= up)
    empirical_coverage = float(np.mean(covered)) * 100.0
    mean_width = float(np.mean(up - low))

    return {
        "empirical_coverage_pct": round(empirical_coverage, 2),
        "mean_interval_width_mw": round(mean_width, 3),
    }

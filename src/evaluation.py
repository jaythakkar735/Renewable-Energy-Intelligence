"""Forecasting metrics, skill score computation, and residual diagnostics."""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_forecast_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    capacity_mw: Optional[float] = None,
    daylight_mask: Optional[np.ndarray | pd.Series] = None,
    baseline_mae: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for energy forecasting.

    Args:
        y_true: Ground truth observed values.
        y_pred: Forecast predictions.
        capacity_mw: Peak capacity for normalization. If None, uses max(y_true).
        daylight_mask: Boolean array where True indicates daylight hours.
        baseline_mae: MAE of the reference baseline to compute skill score.

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
) -> pd.DataFrame:
    """Computes mean error, MAE, and RMSE grouped by hour of the day (0-23)."""
    df = pd.DataFrame({
        "actual": y_true,
        "pred": np.clip(y_pred, 0.0, None),
    })
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


def compute_error_by_season(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """Computes MAE and RMSE grouped by meteorological season."""
    df = pd.DataFrame({
        "actual": y_true,
        "pred": np.clip(y_pred, 0.0, None),
    })
    df["error"] = df["pred"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2

    month = df.index.month
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    df["season"] = month.map(season_map)

    grouped = df.groupby("season").agg(
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        count=("actual", "count"),
    ).loc[["Winter", "Spring", "Summer", "Autumn"]].reset_index()

    return grouped


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

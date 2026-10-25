"""Tests for forecasting metrics and evaluation formulas vs hand-worked examples."""
import numpy as np
import pytest

from src.evaluation import compute_forecast_metrics, compute_coverage


def test_hand_worked_forecast_metrics():
    """Validates metrics against explicit hand-calculated numbers.

    Ground truth: y = [10.0, 20.0, 0.0, 50.0]
    Prediction:   y_hat = [12.0, 15.0, 0.0, 40.0]
    Errors:       e = [+2.0, -5.0, 0.0, -10.0]
    Abs errors:   |e| = [2.0, 5.0, 0.0, 10.0] -> MAE = 17.0 / 4 = 4.25
    Sq errors:    e^2 = [4.0, 25.0, 0.0, 100.0] -> sum = 129.0 -> MSE = 32.25 -> RMSE = sqrt(32.25) approx 5.6789
    Bias:         sum(e) / 4 = -13.0 / 4 = -3.25
    Capacity:     100.0 -> nMAE = 4.25%
    Baseline MAE: 8.5 -> Skill Score = 1.0 - (4.25 / 8.5) = 0.50 (50% improvement)
    """
    y_true = np.array([10.0, 20.0, 0.0, 50.0])
    y_pred = np.array([12.0, 15.0, 0.0, 40.0])
    daylight = np.array([True, True, False, True])

    metrics = compute_forecast_metrics(
        y_true=y_true,
        y_pred=y_pred,
        capacity_mw=100.0,
        daylight_mask=daylight,
        baseline_mae=8.5,
    )

    assert metrics["mae_all"] == pytest.approx(4.25, abs=1e-3)
    assert metrics["rmse_all"] == pytest.approx(np.sqrt(32.25), abs=1e-3)
    assert metrics["bias_all"] == pytest.approx(-3.25, abs=1e-3)
    assert metrics["nmae_all_pct"] == pytest.approx(4.25, abs=1e-2)
    assert metrics["skill_score_vs_baseline"] == pytest.approx(0.50, abs=1e-4)

    # Daylight only: [10, 20, 50] vs [12, 15, 40] -> abs errors [2, 5, 10] -> MAE = 17 / 3 = 5.667
    assert metrics["mae_daylight"] == pytest.approx(17.0 / 3.0, abs=1e-3)


def test_conformal_coverage_hand_worked():
    """Validates empirical coverage calculation against hand-crafted intervals."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    low = np.array([8.0, 18.0, 25.0, 45.0, 48.0])   # 40 is missed (low=45 > 40)
    high = np.array([12.0, 22.0, 35.0, 55.0, 52.0]) # 10, 20, 30, 50 covered -> 4/5 = 80.0%

    cov = compute_coverage(y, low, high)
    assert cov["empirical_coverage_pct"] == pytest.approx(80.0, abs=1e-2)
    assert cov["mean_interval_width_mw"] == pytest.approx(6.4, abs=1e-2)

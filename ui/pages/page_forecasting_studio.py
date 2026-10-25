"""Page C: Forecasting Studio - Model Inference, Baselines, and Residual Diagnostics."""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go

from ui.theme import get_plotly_layout, COLOR_EMERALD, COLOR_CYAN, COLOR_BLUE, COLOR_AMBER, COLOR_RED, render_metric_card
from src.evaluation import compute_forecast_metrics, compute_error_by_hour


def render_page_forecasting_studio(config: dict, technical_mode: bool):
    """Renders the forecasting studio and model diagnostics page."""
    st.markdown("## Forecasting Studio")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: -10px;'>"
        "Evaluate day-ahead solar generation predictions, baseline comparisons, and uncertainty bands on the 2020 holdout test set."
        "</p>",
        unsafe_allow_html=True,
    )

    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    preds_file = reports_dir / "test_predictions.parquet"
    metrics_file = reports_dir / "forecast_metrics.json"

    if not preds_file.exists():
        st.error("Model predictions not found. Please train models first.")
        return

    preds_df = pd.read_parquet(preds_file)
    with open(metrics_file, "r") as f:
        all_metrics = json.load(f)

    # Model selector
    model_options = {
        "LightGBM (Gradient Boosting)": "model_lightgbm",
        "Ridge Regression (Linear ML)": "model_ridge",
        "Random Forest (Ensemble)": "model_random_forest",
        "Persistence Baseline (24h Lag)": "baseline_persistence_24h",
        "Seasonal Naive (168h Lag)": "baseline_seasonal_naive_168h",
    }

    col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
    with col_m1:
        selected_model_label = st.selectbox(
            "Select Forecasting Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose the model or baseline to inspect on the holdout test period.",
        )
        selected_col = model_options[selected_model_label]

    with col_m2:
        show_conformal = st.checkbox(
            "Show Conformal Bands (90%)",
            value=True,
            disabled=("lightgbm" not in selected_col),
            help="Display split-conformal prediction intervals calibrated on chronological 2019 validation data.",
        )

    with col_m3:
        filter_daylight = st.checkbox(
            "Daylight Hours Only",
            value=False,
            help="Filter error metrics and charts to exclude nighttime zero periods.",
        )

    # Date range within test set
    test_min = preds_df.index.min().date()
    test_max = preds_df.index.max().date()
    default_start = pd.to_datetime("2020-06-01").date()
    default_end = pd.to_datetime("2020-06-10").date()

    date_sel = st.date_input(
        "Forecast Inspection Window",
        value=(default_start, default_end),
        min_value=test_min,
        max_value=test_max,
        help="Select date range within the 2020 holdout test set.",
    )

    if isinstance(date_sel, tuple) and len(date_sel) == 2:
        start_t = pd.to_datetime(date_sel[0]).tz_localize("UTC")
        end_t = pd.to_datetime(date_sel[1]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sub_preds = preds_df.loc[start_t:end_t].copy()
    else:
        sub_preds = preds_df.iloc[:240].copy()

    # Metrics on the selected window
    y_true_win = sub_preds["actual"]
    y_pred_win = sub_preds[selected_col]
    daylight_win = sub_preds["is_daylight"].values.astype(bool)
    cap_mw = float(preds_df["actual"].max())

    win_metrics = compute_forecast_metrics(
        y_true=y_true_win,
        y_pred=y_pred_win,
        capacity_mw=cap_mw,
        daylight_mask=daylight_win,
        baseline_mae=all_metrics["Persistence_24h"]["mae_all"],
    )

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            title="MAE (All Hours)",
            value=f"{win_metrics['mae_all']:.2f} MW",
            sub_text=f"nMAE: {win_metrics['nmae_all_pct']:.2f}% of peak",
            pill_text="MAE",
            pill_color="cyan",
        )
    with c2:
        render_metric_card(
            title="RMSE (All Hours)",
            value=f"{win_metrics['rmse_all']:.2f} MW",
            sub_text=f"nRMSE: {win_metrics['nrmse_all_pct']:.2f}% of peak",
            pill_text="RMSE",
            pill_color="blue",
        )
    with c3:
        render_metric_card(
            title="Daylight MAE",
            value=f"{win_metrics.get('mae_daylight', 0.0):.2f} MW",
            sub_text="Evaluated when sun is above horizon",
            pill_text="Daylight",
            pill_color="emerald",
        )
    with c4:
        bias_val = win_metrics["bias_all"]
        b_color = "emerald" if abs(bias_val) < 5 else "amber"
        render_metric_card(
            title="Forecast Bias (ME)",
            value=f"{bias_val:+.2f} MW",
            sub_text="Mean Error (+ over / - under)",
            pill_text="Bias",
            pill_color=b_color,
        )

    # Actual vs Predicted Plot
    fig = go.Figure()

    # Conformal band
    if show_conformal and "lightgbm_lower_90" in sub_preds.columns and "lightgbm" in selected_col:
        fig.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["lightgbm_upper_90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["lightgbm_lower_90"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(34, 211, 238, 0.12)",
                name="90% Conformal Interval",
            )
        )

    # Ground Truth Actual
    fig.add_trace(
        go.Scatter(
            x=sub_preds.index,
            y=sub_preds["actual"],
            name="Actual Solar Generation (MW)",
            line=dict(color=COLOR_EMERALD, width=2.5),
        )
    )

    # Model Prediction
    fig.add_trace(
        go.Scatter(
            x=sub_preds.index,
            y=sub_preds[selected_col],
            name=f"{selected_model_label} (MW)",
            line=dict(color=COLOR_CYAN, width=2, dash="solid"),
        )
    )

    # Baseline comparison trace
    fig.add_trace(
        go.Scatter(
            x=sub_preds.index,
            y=sub_preds["baseline_persistence_24h"],
            name="Persistence Baseline 24h (MW)",
            line=dict(color=COLOR_AMBER, width=1, dash="dot"),
        )
    )

    fig.update_layout(
        get_plotly_layout(
            title=f"Actual vs Day-Ahead Forecast: {selected_model_label}",
            xaxis_title="UTC Timestamp",
            yaxis_title="Generation (MW)",
            height=430,
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error Diagnostics: Error by Hour of Day
    st.markdown("### Error by Hour of Day")
    err_by_hour = compute_error_by_hour(sub_preds["actual"], sub_preds[selected_col])

    fig_hour = go.Figure()
    fig_hour.add_trace(
        go.Bar(
            x=err_by_hour["hour"],
            y=err_by_hour["mae"],
            name="MAE (MW)",
            marker_color=COLOR_CYAN,
        )
    )
    fig_hour.add_trace(
        go.Scatter(
            x=err_by_hour["hour"],
            y=err_by_hour["mean_actual"],
            name="Mean Actual Solar (MW)",
            line=dict(color=COLOR_EMERALD, width=2),
            yaxis="y2",
        )
    )
    layout_hour = get_plotly_layout(
        title="Forecast MAE & Solar Generation by Hour of Day (UTC)",
        xaxis_title="Hour of Day (UTC)",
        yaxis_title="MAE (MW)",
        height=320,
    )
    layout_hour["yaxis2"] = dict(
        title=dict(text="Mean Solar (MW)", font=dict(color=COLOR_EMERALD, size=11)),
        tickfont=dict(color=COLOR_EMERALD, size=10),
        overlaying="y",
        side="right",
        showgrid=False,
    )
    fig_hour.update_layout(layout_hour)
    st.plotly_chart(fig_hour, use_container_width=True)

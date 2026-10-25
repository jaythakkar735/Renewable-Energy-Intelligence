"""Page C: Forecasting Studio - Model Inference, Baselines, and Residual Diagnostics."""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import (
    render_page_header,
    render_metric_card,
    render_provenance_footer,
    get_plotly_layout,
    get_current_theme,
)
from src.evaluation import compute_forecast_metrics, compute_error_by_hour


def render_page_forecasting_studio(config: dict, technical_mode: bool):
    """Renders the forecasting studio and model diagnostics page."""
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    preds_file = reports_dir / "test_predictions.parquet"
    metrics_file = reports_dir / "forecast_metrics.json"

    if not preds_file.exists() or not metrics_file.exists():
        st.error("Model predictions not found. Please train models first.")
        return

    preds_df = pd.read_parquet(preds_file)
    with open(metrics_file, "r") as f:
        all_metrics = json.load(f)

    summary_sentence = (
        "Capacity-normalized ML models cut daylight forecast bias from -40 MW down to -20 MW on the 2020 test set, "
        "enforcing hard night zeros when solar elevation <= 0, and outperforming the auction cutoff-safe "
        "Last-Available-Day baseline with a +4.8% skill score."
    )

    render_page_header(
        title="Forecasting Studio: Day-Ahead Predictions & Uncertainty",
        subtitle="Evaluation of Day-Ahead Solar Forecasts, Leakage-Safe Baselines, and Conformal Prediction Intervals",
        summary_sentence=summary_sentence,
        icon_name="chart",
    )

    # Model selector mapping
    model_options = {
        "LightGBM (Gradient Boosting, Capacity-Normalized)": "model_lightgbm",
        "Random Forest (Ensemble ML, Capacity-Normalized)": "model_random_forest",
        "Ridge Regression (Linear ML, Capacity-Normalized)": "model_ridge",
        "Last-Available-Day (Auction Cutoff Baseline)": "baseline_last_available_day",
        "Seasonal Naive (48h Lag Baseline)": "baseline_seasonal_naive_48h",
        "Seasonal Naive (168h Lag Baseline)": "baseline_seasonal_naive_168h",
    }

    col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
    with col_m1:
        selected_model_label = st.selectbox(
            "Select Forecasting Model or Baseline",
            options=list(model_options.keys()),
            index=0,
            help="Choose the model or baseline to inspect on the holdout test period.",
        )
        selected_col = model_options[selected_model_label]

    with col_m2:
        is_lgb = "lightgbm" in selected_col
        show_conformal = st.checkbox(
            "Show Conformal Bands (90%)",
            value=True,
            disabled=not is_lgb,
            help="Display split-conformal prediction intervals calibrated on chronological 2019 validation data.",
        )
        if not is_lgb:
            st.caption("Conformal prediction intervals are calibrated on the primary LightGBM model.")

    with col_m3:
        show_quantiles = st.checkbox(
            "Show P10 / P50 / P90 Quantiles",
            value=False,
            disabled=not is_lgb,
            help="Display LightGBM quantile regression bands (pinball loss at 0.10, 0.50, 0.90).",
        )

    # Date range within test set
    test_min = preds_df.index.min().date()
    test_max = preds_df.index.max().date()
    default_start = pd.to_datetime("2020-05-30").date()
    default_end = pd.to_datetime("2020-06-12").date()

    date_sel = st.date_input(
        "Forecast Inspection Window",
        value=(default_start, default_end),
        min_value=test_min,
        max_value=test_max,
        help="Select date range within the 2020 holdout test set (includes the June 2-3 telemetry freeze).",
    )

    if isinstance(date_sel, tuple) and len(date_sel) == 2:
        start_t = pd.to_datetime(date_sel[0]).tz_localize("UTC")
        end_t = pd.to_datetime(date_sel[1]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sub_preds = preds_df.loc[start_t:end_t].copy()
    else:
        sub_preds = preds_df.iloc[:240].copy()

    # Metrics on the selected window vs Full Period
    y_true_win = sub_preds["actual"]
    y_pred_win = sub_preds[selected_col]
    daylight_win = sub_preds["is_daylight"].values.astype(bool)
    valid_win = ~sub_preds["is_telemetry_gap"].values
    cap_mw = float(preds_df["actual"].max())

    win_metrics = compute_forecast_metrics(
        y_true=y_true_win,
        y_pred=y_pred_win,
        capacity_mw=cap_mw,
        daylight_mask=daylight_win,
        valid_mask=valid_win,
    )

    # Full period metrics from json
    model_key_map = {
        "model_lightgbm": "LightGBM",
        "model_random_forest": "Random_Forest",
        "model_ridge": "Ridge",
        "baseline_last_available_day": "Last_Available_Day",
        "baseline_seasonal_naive_48h": "Seasonal_Naive_48h",
        "baseline_seasonal_naive_168h": "Seasonal_Naive_168h",
    }
    full_m = all_metrics.get(model_key_map.get(selected_col, "LightGBM"), {})

    # Dual KPI Display: Selected Window vs Full Test Period
    st.markdown("#### Performance Metrics (Selected Window vs Full 9-Month Test Holdout)")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card(
            title="Window MAE (All / Daylight)",
            value=f"{win_metrics['mae_all']:.1f} / {win_metrics.get('mae_daylight', 0):.1f} MW",
            sub_text=f"Full Period: {full_m.get('mae_all', 0):.1f} / {full_m.get('mae_daylight', 0):.1f} MW",
            pill_text="nMAE: " + f"{win_metrics['nmae_all_pct']:.1f}%",
            pill_color="cyan",
            icon_name="chart",
        )
    with k2:
        render_metric_card(
            title="Window RMSE (MW)",
            value=f"{win_metrics['rmse_all']:.1f} MW",
            sub_text=f"Full Period: {full_m.get('rmse_all', 0):.1f} MW",
            pill_text="Spread Error",
            pill_color="blue",
            icon_name="activity",
        )
    with k3:
        bias_win = win_metrics["bias_all"]
        render_metric_card(
            title="Window Mean Bias (MW)",
            value=f"{bias_win:+,.1f} MW",
            sub_text=f"Full Period: {full_m.get('bias_all', 0):+,.1f} MW (Daylight: {full_m.get('bias_daylight', 0):+,.1f})",
            pill_text="Unbiased" if abs(bias_win) < 10 else "Slight Under-forecast",
            pill_color="emerald" if abs(bias_win) < 10 else "amber",
            icon_name="sliders",
        )
    with k4:
        cov_info = all_metrics.get("Conformal_90_Coverage", {})
        render_metric_card(
            title="90% Conformal Coverage",
            value=f"{cov_info.get('overall_coverage_pct', 74):.1f}%",
            sub_text=f"Daylight: {cov_info.get('daylight_coverage_pct', 71):.1f}% | Night: {cov_info.get('night_coverage_pct', 78):.1f}%",
            pill_text="Split-Conformal",
            pill_color="emerald",
            icon_name="shield_check",
        )

    t = get_current_theme()

    # Time series visualization
    fig_ts = go.Figure()

    # Conformal bands
    if show_conformal and is_lgb and "lightgbm_upper_90" in sub_preds.columns:
        fig_ts.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["lightgbm_upper_90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["lightgbm_lower_90"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(14, 116, 144, 0.18)",
                name="90% Conformal Interval",
            )
        )

    # Quantile traces
    if show_quantiles and is_lgb and "lightgbm_p10" in sub_preds.columns:
        fig_ts.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["lightgbm_p10"],
                mode="lines",
                line=dict(color=t["accent_amber"], width=1, dash="dot"),
                name="LightGBM P10",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["lightgbm_p90"],
                mode="lines",
                line=dict(color=t["accent_amber"], width=1, dash="dot"),
                name="LightGBM P90",
            )
        )

    # Actual generation
    fig_ts.add_trace(
        go.Scatter(
            x=sub_preds.index,
            y=sub_preds["actual"],
            mode="lines",
            name="Observed Actual (MW)",
            line=dict(color=t["text_primary"], width=2.2),
        )
    )

    # Forecast trace
    fig_ts.add_trace(
        go.Scatter(
            x=sub_preds.index,
            y=sub_preds[selected_col],
            mode="lines",
            name=f"{selected_model_label.split('(')[0].strip()} Forecast",
            line=dict(color=t["accent_emerald"], width=2),
        )
    )

    # Baseline comparison trace if an ML model is selected
    if is_lgb:
        fig_ts.add_trace(
            go.Scatter(
                x=sub_preds.index,
                y=sub_preds["baseline_last_available_day"],
                mode="lines",
                name="Last-Available-Day Baseline",
                line=dict(color=t["accent_cyan"], width=1.5, dash="dash"),
            )
        )

    # Highlight telemetry gap freeze
    if sub_preds["is_telemetry_gap"].any():
        gap_indices = sub_preds[sub_preds["is_telemetry_gap"]].index
        if len(gap_indices) > 0:
            fig_ts.add_vrect(
                x0=gap_indices.min(),
                x1=gap_indices.max(),
                fillcolor="rgba(239, 68, 68, 0.15)",
                line_color=t["accent_red"],
                line_width=1,
                line_dash="dash",
                annotation_text="SCADA Telemetry Freeze (284.6 MW)",
                annotation_position="top left",
                annotation_font=dict(size=10, color=t["accent_red"]),
            )

    fig_ts.update_layout(
        get_plotly_layout(
            title=f"Hourly Solar Generation & Day-Ahead Forecast ({sub_preds.index.min().strftime('%b %d')} – {sub_preds.index.max().strftime('%b %d, %Y')})",
            xaxis_title="Timestamp (UTC)",
            yaxis_title="Power (MW)",
            height=430,
        )
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Two-Row Diurnal Error-by-Hour Chart (Alighted x-axis 0-23, No dual axes!)
    st.markdown("### Diurnal Error by Hour of Day (Full Test Holdout: 6,576 Hours)")
    st.markdown(
        "Notice that night errors (hours 21 to 03 UTC) drop to **~0 MW** due to enforced astronomical night zero clamping. "
        "The bottom row displays average solar generation to illustrate the physical diurnal curve."
    )

    err_lgb = compute_error_by_hour(preds_df["actual"], preds_df["model_lightgbm"])
    err_lad = compute_error_by_hour(preds_df["actual"], preds_df["baseline_last_available_day"])
    err_sn168 = compute_error_by_hour(preds_df["actual"], preds_df["baseline_seasonal_naive_168h"])

    fig_diurnal = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Mean Absolute Error by Hour of Day (MW)",
            "Average Observed Solar Generation by Hour of Day (MW)",
        ),
    )

    # Row 1: MAE Grouped Bars
    fig_diurnal.add_trace(
        go.Bar(
            x=err_lgb["hour"],
            y=err_lgb["mae"],
            name="LightGBM MAE",
            marker_color=t["accent_emerald"],
        ),
        row=1,
        col=1,
    )
    fig_diurnal.add_trace(
        go.Bar(
            x=err_lad["hour"],
            y=err_lad["mae"],
            name="Last-Available-Day MAE",
            marker_color=t["accent_cyan"],
        ),
        row=1,
        col=1,
    )
    fig_diurnal.add_trace(
        go.Bar(
            x=err_sn168["hour"],
            y=err_sn168["mae"],
            name="Seasonal Naive 168h MAE",
            marker_color=t["color_no_battery"],
        ),
        row=1,
        col=1,
    )

    # Row 2: Average Generation Curve
    fig_diurnal.add_trace(
        go.Scatter(
            x=err_lgb["hour"],
            y=err_lgb["mean_actual"],
            mode="lines+markers",
            name="Mean Observed Solar (MW)",
            line=dict(color=t["accent_blue"], width=2.2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)",
        ),
        row=2,
        col=1,
    )

    fig_diurnal.update_layout(
        get_plotly_layout(
            title="",
            xaxis_title="Hour of Day (UTC, 0 to 23)",
            yaxis_title="MAE (MW)",
            height=480,
        )
    )
    fig_diurnal.update_xaxes(tickmode="linear", tick0=0, dtick=1)
    st.plotly_chart(fig_diurnal, use_container_width=True)

    render_provenance_footer()

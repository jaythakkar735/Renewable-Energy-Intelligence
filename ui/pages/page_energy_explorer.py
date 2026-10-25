"""Page B: Energy Explorer - Generation, Load, and Market Price Analytics."""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import get_clean_data
from ui.theme import (
    render_page_header,
    render_metric_card,
    render_provenance_footer,
    get_plotly_layout,
    get_current_theme,
)


def render_page_energy_explorer(config: dict, technical_mode: bool):
    """Renders the interactive energy exploration page."""
    clean_df = get_clean_data()
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    gaps_file = reports_dir / "telemetry_gaps.json"

    telemetry_gaps = []
    if gaps_file.exists():
        with open(gaps_file, "r") as f:
            telemetry_gaps = json.load(f)

    summary_sentence = (
        "Denmark West (DK_1) features strong diurnal solar generation peaking at 726.7 MW, "
        "intermittent negative day-ahead prices (down to -50.05 EUR/MWh), and 13 identified telemetry freezes "
        "including a 26-hour SCADA freeze on June 2–3, 2020."
    )

    render_page_header(
        title="Energy Explorer: Generation, Demand & Prices",
        subtitle="Exploratory Analysis of Historical Power Flows, Spot Prices, and Telemetry Quality",
        summary_sentence=summary_sentence,
        icon_name="sun",
    )

    # Filter controls
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        min_date = clean_df.index.min().date()
        max_date = clean_df.index.max().date()
        default_start = pd.to_datetime("2020-05-28").date()
        default_end = pd.to_datetime("2020-06-10").date()

        date_range = st.date_input(
            "Select Date Range",
            value=(default_start, default_end),
            min_value=min_date,
            max_value=max_date,
            help="Filter observations between the selected dates.",
        )

    with col_c2:
        tz_option = st.selectbox(
            "Display Timezone",
            options=["UTC (Universal Time)", "Copenhagen Local (CET / CEST)"],
            index=0,
            help="Toggle between standard market UTC and Danish local daylight saving time.",
        )

    with col_c3:
        highlight_gaps = st.checkbox(
            "Highlight Telemetry Freezes",
            value=True,
            help="Shade regions with identified SCADA sensor flatlines (>= 3 consecutive non-zero hours).",
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0]).tz_localize("UTC")
        end_dt = pd.to_datetime(date_range[1]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sub_df = clean_df.loc[start_dt:end_dt].copy()
    else:
        sub_df = clean_df.iloc[-336:].copy()

    if sub_df.empty:
        sub_df = clean_df.iloc[-336:].copy()

    # Apply timezone conversion if requested
    if "Copenhagen" in tz_option:
        plot_df = sub_df.copy()
        plot_df.index = plot_df.index.tz_convert("Europe/Copenhagen")
    else:
        plot_df = sub_df

    # Summary metrics for the filtered window
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            title="Peak Solar Generation",
            value=f"{plot_df['solar_generation_mw'].max():.1f} MW",
            sub_text=f"Mean: {plot_df['solar_generation_mw'].mean():.1f} MW",
            pill_text="Solar PV",
            pill_color="emerald",
            icon_name="sun",
        )
    with col2:
        render_metric_card(
            title="Peak System Load",
            value=f"{plot_df['load_actual_mw'].max():.1f} MW",
            sub_text=f"Mean: {plot_df['load_actual_mw'].mean():.1f} MW",
            pill_text="DK1 Demand",
            pill_color="blue",
            icon_name="activity",
        )
    with col3:
        render_metric_card(
            title="Mean Day-Ahead Price",
            value=f"{plot_df['price_day_ahead_eur_mwh'].mean():.2f} EUR",
            sub_text=f"Min: {plot_df['price_day_ahead_eur_mwh'].min():.1f} | Max: {plot_df['price_day_ahead_eur_mwh'].max():.1f}",
            pill_text="Spot Market",
            pill_color="cyan",
            icon_name="zap",
        )
    with col4:
        neg_hrs = int((plot_df["price_day_ahead_eur_mwh"] < 0).sum())
        freeze_hrs = int(plot_df["is_telemetry_gap"].sum())
        render_metric_card(
            title="Negative Price & Gap Hours",
            value=f"{neg_hrs} Neg / {freeze_hrs} Gap",
            sub_text=f"{neg_hrs / max(1, len(plot_df)) * 100:.1f}% negative price frequency",
            pill_text="Market Risk",
            pill_color="amber",
            icon_name="alert",
        )

    t = get_current_theme()

    # Stacked Two-Row Plotly Subplots (No dual axes!)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Power Generation & Load Demand (MW)", "Day-Ahead Spot Price (EUR/MWh)"),
    )

    # Subplot 1: Power flows
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["solar_generation_mw"],
            mode="lines",
            name="Solar Generation",
            line=dict(color=t["accent_emerald"], width=2),
            fill="tozeroy",
            fillcolor="rgba(5, 150, 105, 0.12)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["load_actual_mw"],
            mode="lines",
            name="System Load Demand",
            line=dict(color=t["accent_blue"], width=1.8),
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Day-ahead spot price
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["price_day_ahead_eur_mwh"],
            mode="lines",
            name="Day-Ahead Price",
            line=dict(color=t["accent_cyan"], width=1.8),
        ),
        row=2,
        col=1,
    )

    # Zero line on price subplot
    fig.add_hline(
        y=0.0,
        line_dash="dot",
        line_color=t["accent_amber"],
        line_width=1.2,
        row=2,
        col=1,
    )

    # Highlight telemetry freezes if toggled
    if highlight_gaps:
        for g in telemetry_gaps:
            g_start = pd.to_datetime(g["start"])
            g_end = pd.to_datetime(g["end"])
            # Check if gap intersects view window
            if g_end >= plot_df.index.min() and g_start <= plot_df.index.max():
                x0 = g_start if "Copenhagen" not in tz_option else g_start.tz_convert("Europe/Copenhagen")
                x1 = g_end if "Copenhagen" not in tz_option else g_end.tz_convert("Europe/Copenhagen")
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="rgba(239, 68, 68, 0.18)",
                    line_color=t["accent_red"],
                    line_width=1,
                    line_dash="dash",
                    annotation_text=f"Freeze {g['hours']}h ({g['stuck_value_mw']}MW)",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color=t["accent_red"]),
                    row=1,
                    col=1,
                )

    layout_dict = get_plotly_layout(height=520)
    layout_dict["title"] = dict(text="DK_1 Electricity System Dynamics", font=dict(color=t["text_primary"], size=14))
    layout_dict["yaxis"]["title"] = dict(text="Power (MW)", font=dict(color=t["text_muted"], size=12))
    layout_dict["yaxis2"] = dict(
        title=dict(text="Price (EUR/MWh)", font=dict(color=t["text_muted"], size=12)),
        tickfont=dict(color=t["text_muted"], size=11),
        gridcolor=t["border"],
        zerolinecolor=t["border"],
        showgrid=True,
    )

    fig.update_layout(**layout_dict)
    st.plotly_chart(fig, width="stretch")

    # Telemetry Freezes Table
    if telemetry_gaps:
        st.markdown("### Telemetry Freeze Audit Log (Identified SCADA Flatlines)")
        st.markdown(
            "The following periods contain non-zero stuck telemetry readings where Energinet SCADA reported identical "
            "values across multiple consecutive hours. These periods are flagged, excluded from ML loss training, "
            "and preserved transparently without synthetic interpolation."
        )
        df_gaps = pd.DataFrame(telemetry_gaps)
        st.dataframe(df_gaps, width="stretch", hide_index=True)

    render_provenance_footer()

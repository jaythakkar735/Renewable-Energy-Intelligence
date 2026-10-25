"""Page B: Energy Explorer - Generation, Load, and Market Price Analytics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.preprocessing import get_clean_data
from ui.theme import get_plotly_layout, COLOR_EMERALD, COLOR_CYAN, COLOR_AMBER, COLOR_TEXT_MUTED, render_metric_card


def render_page_energy_explorer(config: dict, technical_mode: bool):
    """Renders the interactive energy exploration page."""
    st.markdown("## Energy Explorer")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: -10px;'>"
        "Explore historical solar generation, electrical load demand, and day-ahead electricity prices in Denmark West (`DK_1`)."
        "</p>",
        unsafe_allow_html=True,
    )

    clean_df = get_clean_data()

    # Filter controls
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        min_date = clean_df.index.min().date()
        max_date = clean_df.index.max().date()
        default_start = pd.to_datetime("2020-05-01").date()
        default_end = pd.to_datetime("2020-05-14").date()

        date_range = st.date_input(
            "Select Date Range",
            value=(default_start, default_end),
            min_value=min_date,
            max_value=max_date,
            help="Filter observations between the selected dates (UTC).",
        )

    with col_c2:
        resample_freq = st.selectbox(
            "Temporal Aggregation",
            options=["Hourly (Raw)", "Daily Average", "Weekly Average"],
            index=0,
            help="Select resampling frequency for trend analysis.",
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0]).tz_localize("UTC")
        end_dt = pd.to_datetime(date_range[1]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sub_df = clean_df.loc[start_dt:end_dt].copy()
    else:
        sub_df = clean_df.iloc[-336:].copy()

    if resample_freq == "Daily Average":
        plot_df = sub_df.resample("1D").mean()
    elif resample_freq == "Weekly Average":
        plot_df = sub_df.resample("1W").mean()
    else:
        plot_df = sub_df

    # Summary metrics for the filtered window
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            title="Max Solar Generation",
            value=f"{plot_df['solar_generation_mw'].max():.1f} MW",
            sub_text=f"Mean: {plot_df['solar_generation_mw'].mean():.1f} MW",
            pill_text="Solar",
            pill_color="emerald",
        )
    with col2:
        render_metric_card(
            title="Max System Load",
            value=f"{plot_df['load_actual_mw'].max():.1f} MW",
            sub_text=f"Mean: {plot_df['load_actual_mw'].mean():.1f} MW",
            pill_text="Load",
            pill_color="cyan",
        )
    with col3:
        render_metric_card(
            title="Mean Day-Ahead Price",
            value=f"{plot_df['price_day_ahead_eur_mwh'].mean():.2f} EUR",
            sub_text=f"Min: {plot_df['price_day_ahead_eur_mwh'].min():.1f} | Max: {plot_df['price_day_ahead_eur_mwh'].max():.1f}",
            pill_text="Spot Price",
            pill_color="amber",
        )
    with col4:
        neg_hrs = int((plot_df['price_day_ahead_eur_mwh'] < 0).sum())
        render_metric_card(
            title="Negative Price Hours",
            value=f"{neg_hrs}",
            sub_text=f"{neg_hrs / max(1, len(plot_df)) * 100:.1f}% of selected span",
            pill_text="Curtailment Risk",
            pill_color="amber",
        )

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["solar_generation_mw"],
            name="Solar Generation (MW)",
            line=dict(color=COLOR_EMERALD, width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["load_actual_mw"],
            name="System Load (MW)",
            line=dict(color=COLOR_CYAN, width=1.5, dash="dot"),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["price_day_ahead_eur_mwh"],
            name="Day-Ahead Price (EUR/MWh)",
            line=dict(color=COLOR_AMBER, width=2),
            yaxis="y2",
        )
    )

    layout = get_plotly_layout(
        title=f"Denmark West (DK1) Dynamics ({plot_df.index.min().strftime('%Y-%m-%d')} to {plot_df.index.max().strftime('%Y-%m-%d')})",
        xaxis_title="UTC Timestamp",
        yaxis_title="Power (MW)",
        height=450,
    )
    layout["yaxis2"] = dict(
        title=dict(text="Spot Price (EUR/MWh)", font=dict(color=COLOR_AMBER, size=12)),
        tickfont=dict(color=COLOR_AMBER, size=11),
        overlaying="y",
        side="right",
        showgrid=False,
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

    # Technical Quality & Completeness Audit
    if technical_mode:
        st.markdown("### Data Quality & Gap Audit")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(
                """
                **OPSD Ingestion Audit Findings**:
                - **Raw File Rows**: 50,401 hourly entries
                - **Duplicate Timestamps**: 0
                - **Missing Grid Intervals**: 0
                - **Clean Index Verification**: Complete contiguous UTC hourly grid
                - **Solar Negative Values**: 0 (Clean physical non-negativity)
                """
            )
        with col_t2:
            st.markdown(
                """
                **Missing Value Imputation**:
                - `load_actual_mw`: 3 raw NaNs (< 0.01%) -> Time-interpolated
                - `price_day_ahead_eur_mwh`: 15 raw NaNs (< 0.03%) -> Forward-filled
                - `solar_generation_mw`: 12 raw NaNs (< 0.03%) -> Imputed with zero-nighttime preservation
                - Lookahead free: No future synthetic imputation applied.
                """
            )

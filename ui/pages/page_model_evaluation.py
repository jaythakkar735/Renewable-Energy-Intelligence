"""Page E: Model Evaluation - Holdout Benchmarks, Skill Scores, and Sensitivity Sweeps."""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go

from ui.theme import get_plotly_layout, COLOR_EMERALD, COLOR_CYAN, COLOR_BLUE, COLOR_AMBER, COLOR_RED


def render_page_model_evaluation(config: dict, technical_mode: bool):
    """Renders the comprehensive model evaluation and sensitivity analysis page."""
    st.markdown("## Model Evaluation & Sensitivity Analysis")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: -10px;'>"
        "Chronological holdout performance, benchmark comparisons, and offline multi-dimensional sensitivity sweeps."
        "</p>",
        unsafe_allow_html=True,
    )

    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    leaderboard_file = reports_dir / "model_leaderboard.csv"
    sens_file = reports_dir / "sensitivity_results.json"

    if not leaderboard_file.exists():
        st.error("Model leaderboard not found. Please train models first.")
        return

    df_lead = pd.read_csv(leaderboard_file)

    # 1. Model Leaderboard Table
    st.markdown("### 1. Forecasting Model Leaderboard (Test Holdout: Jan–Sep 2020)")
    st.dataframe(
        df_lead[[
            "model", "type", "mae_all", "rmse_all", "nmae_all_pct",
            "mae_daylight", "skill_score", "train_time_sec", "inference_time_ms", "size_kb"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # Leaderboard analysis callout
    st.markdown(
        """
        <div class="callout-box">
            <b>Evaluation Insights:</b><br>
            - <b>Point Error vs Outlier Dynamics</b>: While the 24h Persistence baseline achieves a competitive overall MAE (40.36 MW), the machine learning models (Ridge Regression: 77.03 MW RMSE; LightGBM: 82.49 MW RMSE) dramatically reduce large swing residuals, cutting RMSE by up to <b>11.3%</b> relative to persistence.<br>
            - <b>Operational Efficiency</b>: LightGBM trains in under <b>0.2 seconds</b> on CPU and requires only <b>231 KB</b> of storage, while generating predictions in <b>6.6 milliseconds</b> for the entire 6,576-hour test set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Sensitivity Analysis Section
    st.markdown("---")
    st.markdown("### 2. Multi-Dimensional Sensitivity Analysis")

    if sens_file.exists():
        with open(sens_file, "r") as f:
            sens = json.load(f)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Forecast Quality", "Battery Capacity", "Power Rating", "Efficiency (RTE)", "Retail Adder"
        ])

        with tab1:
            st.markdown("#### Impact of Forecast Quality on Electricity Procurement Cost")
            df_qual = pd.DataFrame(sens["forecast_quality"])
            st.dataframe(df_qual, use_container_width=True, hide_index=True)

            fig_q = go.Figure()
            fig_q.add_trace(
                go.Scatter(
                    x=df_qual["solar_forecast_rmse_mw"],
                    y=df_qual["simulated_total_cost_eur"],
                    mode="lines+markers+text",
                    text=df_qual["forecast_source"],
                    textposition="top right",
                    line=dict(color=COLOR_CYAN, width=2),
                    marker=dict(size=10, color=COLOR_EMERALD),
                )
            )
            fig_q.update_layout(
                get_plotly_layout(
                    title="Simulated Cost vs Solar Forecast RMSE (MW)",
                    xaxis_title="Forecast RMSE (MW)",
                    yaxis_title="Total Simulated Electricity Cost (EUR)",
                    height=350,
                )
            )
            st.plotly_chart(fig_q, use_container_width=True)

        with tab2:
            st.markdown("#### Sensitivity to Battery Energy Capacity (MWh)")
            df_cap = pd.DataFrame(sens["capacity"])
            st.dataframe(df_cap, use_container_width=True, hide_index=True)

            fig_cap = go.Figure()
            fig_cap.add_trace(
                go.Bar(
                    x=[f"{c} MWh" for c in df_cap["capacity_mwh"]],
                    y=df_cap["forecast_savings_rule_based_eur"],
                    name="Savings vs Rule-Based (EUR)",
                    marker_color=COLOR_CYAN,
                )
            )
            fig_cap.update_layout(
                get_plotly_layout(
                    title="Incremental Savings vs Rule-Based by Storage Capacity",
                    xaxis_title="Battery Capacity",
                    yaxis_title="Incremental Savings (EUR)",
                    height=320,
                )
            )
            st.plotly_chart(fig_cap, use_container_width=True)

        with tab3:
            st.markdown("#### Sensitivity to Inverter Power Rating (MW)")
            df_pwr = pd.DataFrame(sens["power"])
            st.dataframe(df_pwr, use_container_width=True, hide_index=True)

        with tab4:
            st.markdown("#### Sensitivity to Round-Trip Efficiency (RTE)")
            df_eff = pd.DataFrame(sens["efficiency"])
            st.dataframe(df_eff, use_container_width=True, hide_index=True)

        with tab5:
            st.markdown("#### Sensitivity to Retail Grid Adder (EUR/MWh)")
            df_add = pd.DataFrame(sens["adder"])
            st.dataframe(df_add, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3. Methodological Limitations & Realistic Perspective")
    st.markdown(
        """
        - **Retrospective Data Artifact**: All dispatch evaluations are conducted retrospectively on 2020 Danish bidding zone data.
        - **No Synthetic Lookahead**: All features and day-ahead market schedules strictly respect historical cutoffs, but real-world operations feature intra-day balancing markets (continuous 15-minute trading) not modeled in this day-ahead simulation.
        - **Tariff Assumptions**: The retail markup adder (20 EUR/MWh) is an assumed regulatory proxy. Real commercial facilities face peak demand capacity charges (EUR/kW-month) which would further heighten the economic value of battery peak shaving.
        """
    )

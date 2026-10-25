"""Page G: Data & Results Export matching active configuration."""
import streamlit as st
import pandas as pd
from pathlib import Path

from src.preprocessing import get_clean_data


def render_page_export(config: dict, technical_mode: bool):
    """Renders the data export and artifact download page."""
    st.markdown("## Data & Results Export")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: -10px;'>"
        "Download reproducible CSV files of clean historical data, day-ahead model forecasts, and simulation results matching current parameters."
        "</p>",
        unsafe_allow_html=True,
    )

    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    preds_file = reports_dir / "test_predictions.parquet"
    dispatch_file = reports_dir / "dispatch_summary.csv"
    lead_file = reports_dir / "model_leaderboard.csv"

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### 1. Model Predictions & Leaderboard")
        if preds_file.exists():
            preds_df = pd.read_parquet(preds_file)
            csv_preds = preds_df.to_csv().encode("utf-8")
            st.download_button(
                label="Download Forecast Predictions (CSV)",
                data=csv_preds,
                file_name="forecast_test_predictions_2020.csv",
                mime="text/csv",
                help="Download hourly actual generation and all model forecasts for the 2020 test period.",
            )

        if lead_file.exists():
            df_lead = pd.read_csv(lead_file)
            csv_lead = df_lead.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Model Leaderboard (CSV)",
                data=csv_lead,
                file_name="model_leaderboard.csv",
                mime="text/csv",
                help="Download accuracy, latency, and memory metrics across all models.",
            )

    with col_e2:
        st.markdown("### 2. Battery Simulation Results")
        # Check if live simulation summary exists in session state
        if "live_simulation_summary" in st.session_state:
            df_summ = st.session_state["live_simulation_summary"]
            csv_summ = df_summ.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Active Battery Dispatch Summary (CSV)",
                data=csv_summ,
                file_name="active_battery_dispatch_summary.csv",
                mime="text/csv",
                help="Download simulation comparative table matching active UI parameter sliders.",
            )
        elif dispatch_file.exists():
            df_summ = pd.read_csv(dispatch_file)
            csv_summ = df_summ.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Default Battery Dispatch Summary (CSV)",
                data=csv_summ,
                file_name="battery_dispatch_summary.csv",
                mime="text/csv",
                help="Download standard 10 MWh / 2.5 MW dispatch comparative table.",
            )

        if "live_simulation_results" in st.session_state:
            tl = st.session_state["live_simulation_results"]["forecast_informed"].timeline
            csv_tl = tl.to_csv().encode("utf-8")
            st.download_button(
                label="Download Forecast-Informed Dispatch Timeline (CSV)",
                data=csv_tl,
                file_name="forecast_informed_hourly_timeline.csv",
                mime="text/csv",
                help="Download hourly power flows, SoC, and cost settlements matching active sliders.",
            )

    st.markdown("---")
    st.markdown("### 3. Historical Clean Dataset")
    clean_df = get_clean_data()
    # Offer 2020 slice or full
    sub_2020 = clean_df.loc["2020-01-01":]
    csv_clean = sub_2020.to_csv().encode("utf-8")
    st.download_button(
        label="Download Denmark West (DK1) Clean 2020 Data (CSV)",
        data=csv_clean,
        file_name="dk1_clean_hourly_2020.csv",
        mime="text/csv",
        help="Download clean preprocessed hourly solar, wind, load, and day-ahead spot prices for 2020.",
    )

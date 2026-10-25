"""Renewable Energy Intelligence Streamlit Dashboard Application."""
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from src.config import load_config
from ui.theme import apply_custom_css
from ui.pages.page_overview import render_page_overview
from ui.pages.page_energy_explorer import render_page_energy_explorer
from ui.pages.page_forecasting_studio import render_page_forecasting_studio
from ui.pages.page_battery_laboratory import render_page_battery_laboratory
from ui.pages.page_model_evaluation import render_page_model_evaluation
from ui.pages.page_methodology import render_page_methodology
from ui.pages.page_export import render_page_export


def main():
    st.set_page_config(
        page_title="Renewable Energy Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply unified dark energy theme
    apply_custom_css()

    config = load_config()

    # Sidebar Header & Branding
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 15px 0;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #22D3EE; letter-spacing: -0.02em;">
                RENEWABLE ENERGY INTELLIGENCE
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
                Forecasting & Battery Storage Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audience Mode Toggle
    view_mode = st.sidebar.radio(
        "Audience View Mode",
        options=["Plain-Language (Executive)", "Technical (Energy Systems)"],
        index=0,
        help="Toggle between high-level executive summaries and in-depth engineering formulas and residual diagnostics.",
    )
    technical_mode = "Technical" in view_mode

    st.sidebar.markdown("---")

    # Page Navigation
    pages = {
        "Overview": render_page_overview,
        "Energy Explorer": render_page_energy_explorer,
        "Forecasting Studio": render_page_forecasting_studio,
        "Battery Laboratory": render_page_battery_laboratory,
        "Model Evaluation": render_page_model_evaluation,
        "Methodology & Explainability": render_page_methodology,
        "Data & Results Export": render_page_export,
    }

    selected_page_name = st.sidebar.radio(
        "Navigation",
        options=list(pages.keys()),
        index=0,
        help="Navigate between analytics, forecasting studio, battery dispatch, and exports.",
    )

    st.sidebar.markdown("---")

    # System Status & Metadata
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.5;">
            <b>Zone</b>: {config["data"]["selected_country"]} (Denmark West)<br>
            <b>Target</b>: Solar PV (Hourly UTC)<br>
            <b>Storage</b>: {config["battery"]["energy_capacity_mwh"]} MWh / {config["battery"]["max_charge_power_mw"]} MW<br>
            <b>Status</b>: <span style="color: #10B981; font-weight: 600;">Models Loaded</span><br>
            <b>Engine</b>: HiGHS Linear Solver
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render selected page
    render_func = pages[selected_page_name]
    render_func(config=config, technical_mode=technical_mode)


if __name__ == "__main__":
    main()

"""Renewable Energy Intelligence Streamlit Dashboard Application."""
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from src.config import load_config
from ui.theme import apply_custom_css, get_current_theme
from ui.icons import get_icon
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
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for theme
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"

    # Sidebar Header & Branding (No emojis, clean technical typography)
    st.sidebar.markdown(
        f"""
        <div style="padding: 10px 0 15px 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; letter-spacing: -0.01em; display: flex; align-items: center;">
                {get_icon("zap", color="#38BDF8", size=20)}
                RENEWABLE ENERGY INTELLIGENCE
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 3px;">
                Forecasting & Battery Storage Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Theme Toggle in Sidebar
    theme_choice = st.sidebar.radio(
        "Appearance Theme",
        options=["Light Mode (Default)", "Dark Mode"],
        index=0 if st.session_state["theme_mode"] == "light" else 1,
        help="Switch between light professional theme (default) and dark mode.",
    )
    st.session_state["theme_mode"] = "dark" if "Dark" in theme_choice else "light"

    # Apply CSS for chosen theme
    apply_custom_css()

    config = load_config()

    # Audience Mode Toggle
    view_mode = st.sidebar.radio(
        "Audience Mode",
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
        <div style="font-size: 0.76rem; color: #94A3B8; line-height: 1.55;">
            <b>Zone</b>: {config["data"]["selected_country"]} (Denmark West)<br>
            <b>Target</b>: Solar PV (Hourly UTC)<br>
            <b>Auction Cutoff</b>: 12:00 UTC Day-Ahead<br>
            <b>Storage</b>: {config["battery"]["energy_capacity_mwh"]} MWh / {config["battery"]["max_charge_power_mw"]} MW<br>
            <b>Engine</b>: HiGHS Linear Solver<br>
            <b>Contrast</b>: WCAG AA Compliant
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render selected page
    render_func = pages[selected_page_name]
    render_func(config=config, technical_mode=technical_mode)


if __name__ == "__main__":
    main()

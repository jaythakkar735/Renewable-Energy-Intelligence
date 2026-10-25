"""Page D: Battery Laboratory - Interactive Simulation, Optimization, and Physical Dispatch."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from src.preprocessing import get_clean_data
from src.battery_simulator import BatteryConfig
from src.strategies import scale_to_site, compare_dispatch_strategies
from ui.theme import get_plotly_layout, COLOR_EMERALD, COLOR_CYAN, COLOR_BLUE, COLOR_AMBER, COLOR_RED, render_metric_card


def render_page_battery_laboratory(config: dict, technical_mode: bool):
    """Renders the interactive battery dispatch simulation laboratory."""
    st.markdown("## Battery Laboratory")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: -10px;'>"
        "Configure battery storage parameters and market tariffs to simulate live dispatch and compare Rule-Based, Forecast-Informed, and Perfect Foresight strategies."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout-box callout-box-amber">
            <b>Theoretical Reference Notice:</b> <i>Perfect Foresight</i> is a mathematical upper bound calculated by solving the optimization problem using ground-truth future generation and load. It cannot be achieved in real-world operations and serves solely as a reference benchmark.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar / Top Parameters
    with st.expander("Interactive Storage & Tariff Parameters", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("**Battery Hardware Specs**")
            cap_mwh = st.slider(
                "Energy Capacity (MWh)",
                min_value=2.0,
                max_value=30.0,
                value=float(config["battery"]["energy_capacity_mwh"]),
                step=1.0,
                help="Nominal nameplate energy storage capacity in MWh.",
            )
            pwr_mw = st.slider(
                "Max Power Rating (MW)",
                min_value=0.5,
                max_value=10.0,
                value=float(config["battery"]["max_charge_power_mw"]),
                step=0.5,
                help="Maximum continuous charge and discharge power in MW.",
            )
            rte_pct = st.slider(
                "Round-Trip Efficiency (%)",
                min_value=65,
                max_value=98,
                value=int(config["battery"]["round_trip_efficiency"] * 100),
                step=1,
                help="AC-to-AC round-trip efficiency (losses split equally into charge/discharge).",
            )

        with col_p2:
            st.markdown("**Operational Limits**")
            soc_min_pct = st.slider(
                "Min SoC (%)",
                min_value=0,
                max_value=30,
                value=int(config["battery"]["soc_min"] * 100),
                step=5,
                help="Lower buffer limit to protect battery health.",
            )
            soc_max_pct = st.slider(
                "Max SoC (%)",
                min_value=70,
                max_value=100,
                value=int(config["battery"]["soc_max"] * 100),
                step=5,
                help="Upper buffer limit to avoid accelerated calendar aging.",
            )
            deg_eur = st.number_input(
                "Degradation Cost (EUR/MWh)",
                min_value=0.0,
                max_value=25.0,
                value=float(config["battery"]["degradation_cost_eur_mwh"]),
                step=1.0,
                help="Throughput degradation penalty per MWh charged or discharged.",
            )

        with col_p3:
            st.markdown("**Tariffs & Grid Policy**")
            adder_eur = st.slider(
                "Retail Grid Adder (EUR/MWh)",
                min_value=0.0,
                max_value=60.0,
                value=float(config["tariffs"]["retail_adder_eur_mwh"]),
                step=5.0,
                help="Network tariffs, levies, and supplier retail markup on grid import.",
            )
            export_ratio = st.slider(
                "Export Remuneration Ratio",
                min_value=0.0,
                max_value=1.5,
                value=float(config["tariffs"]["export_ratio"]),
                step=0.1,
                help="Ratio of day-ahead price credited when feeding surplus energy into the grid.",
            )
            allow_neg = st.checkbox(
                "Allow Negative Price Export",
                value=bool(config["tariffs"]["allow_negative_pricing"]),
                help="If False, surplus solar is curtailed during negative spot price hours to prevent paying for feed-in.",
            )

    # Load data & run live simulation
    clean_df = get_clean_data()
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
    preds_file = reports_dir / "test_predictions.parquet"
    preds_df = pd.read_parquet(preds_file)

    test_slice = clean_df.loc[preds_df.index]
    site_df = scale_to_site(
        test_slice,
        peak_load_mw=config["site"]["peak_load_mw"],
        solar_capacity_mw=config["site"]["solar_capacity_mw"],
    )

    solar_fc_scaled = preds_df["model_lightgbm"].values * site_df["solar_scale_factor"].values

    live_battery_cfg = BatteryConfig(
        energy_capacity_mwh=float(cap_mwh),
        max_charge_power_mw=float(pwr_mw),
        max_discharge_power_mw=float(pwr_mw),
        round_trip_efficiency=float(rte_pct) / 100.0,
        soc_min=float(soc_min_pct) / 100.0,
        soc_max=float(soc_max_pct) / 100.0,
        soc_initial=0.50,
        soc_terminal=0.50,
        grid_import_limit_mw=float(config["site"]["grid_import_limit_mw"]),
        grid_export_limit_mw=float(config["site"]["grid_export_limit_mw"]),
        retail_adder_eur_mwh=float(adder_eur),
        export_ratio=float(export_ratio),
        degradation_cost_eur_mwh=float(deg_eur),
        allow_negative_pricing=allow_neg,
    )

    results, summary_df = compare_dispatch_strategies(
        site_df=site_df,
        solar_forecast_mw=solar_fc_scaled,
        load_forecast_mw=site_df["load_forecast_mw"],
        battery_cfg=live_battery_cfg,
    )

    # Store current simulation results in session state for Export page
    st.session_state["live_simulation_summary"] = summary_df
    st.session_state["live_simulation_results"] = results
    st.session_state["live_battery_cfg"] = live_battery_cfg

    # Display Comparative Results Table
    st.markdown("### Comparative Strategy Performance (Holdout Test Period)")
    st.dataframe(
        summary_df[[
            "Strategy", "Total Cost (EUR)", "Savings vs No Battery (EUR)",
            "Savings vs No Battery (%)", "Savings vs Rule-Based (EUR)",
            "Energy Charged (MWh)", "Energy Discharged (MWh)", "Eq. Full Cycles",
            "Max Balance Residual (MW)", "Violations"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # Cost Comparison Bar Chart
    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Bar(
            x=summary_df["Strategy"],
            y=summary_df["Total Cost (EUR)"],
            marker_color=[COLOR_AMBER, COLOR_BLUE, COLOR_CYAN, COLOR_EMERALD],
            text=[f"EUR {v:,.0f}" for v in summary_df["Total Cost (EUR)"]],
            textposition="auto",
        )
    )
    fig_cost.update_layout(
        get_plotly_layout(
            title="Total Electricity Procurement Cost Comparison",
            xaxis_title="Strategy",
            yaxis_title="Total Cost (EUR)",
            height=340,
        )
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Detailed Timeline Inspection
    st.markdown("### Detailed Dispatch & State of Charge Timeline")
    inspect_window = st.selectbox(
        "Select Inspection Window",
        options=["First 7 Days of June 2020", "First 7 Days of August 2020", "Full First 14 Days"],
        index=0,
    )

    if inspect_window == "First 7 Days of June 2020":
        start_w = "2020-06-01 00:00:00+00:00"
        end_w = "2020-06-07 23:00:00+00:00"
    elif inspect_window == "First 7 Days of August 2020":
        start_w = "2020-08-01 00:00:00+00:00"
        end_w = "2020-08-07 23:00:00+00:00"
    else:
        start_w = "2020-01-01 00:00:00+00:00"
        end_w = "2020-01-14 23:00:00+00:00"

    tl_fc = results["forecast_informed"].timeline.loc[start_w:end_w]
    tl_rule = results["rule_based"].timeline.loc[start_w:end_w]
    tl_perf = results["perfect_foresight"].timeline.loc[start_w:end_w]

    # SoC Comparison Plot
    fig_soc = go.Figure()
    fig_soc.add_trace(
        go.Scatter(
            x=tl_fc.index,
            y=tl_fc["soc"] * 100.0,
            name="Forecast-Informed SoC (%)",
            line=dict(color=COLOR_CYAN, width=2.5),
        )
    )
    fig_soc.add_trace(
        go.Scatter(
            x=tl_rule.index,
            y=tl_rule["soc"] * 100.0,
            name="Rule-Based SoC (%)",
            line=dict(color=COLOR_AMBER, width=1.5, dash="dash"),
        )
    )
    fig_soc.add_trace(
        go.Scatter(
            x=tl_perf.index,
            y=tl_perf["soc"] * 100.0,
            name="Perfect Foresight SoC (%)",
            line=dict(color=COLOR_EMERALD, width=1.5, dash="dot"),
        )
    )
    fig_soc.update_layout(
        get_plotly_layout(
            title=f"Battery State of Charge (SoC) Dynamics ({start_w[:10]} to {end_w[:10]})",
            xaxis_title="UTC Timestamp",
            yaxis_title="State of Charge (%)",
            height=320,
        )
    )
    st.plotly_chart(fig_soc, use_container_width=True)

    # Power Flows Plot for Forecast-Informed Strategy
    fig_flow = go.Figure()
    fig_flow.add_trace(
        go.Scatter(
            x=tl_fc.index,
            y=tl_fc["p_charge_mw"],
            name="Battery Charging (MW)",
            line=dict(color=COLOR_EMERALD, width=2),
        )
    )
    fig_flow.add_trace(
        go.Scatter(
            x=tl_fc.index,
            y=tl_fc["p_discharge_mw"],
            name="Battery Discharging (MW)",
            line=dict(color=COLOR_CYAN, width=2),
        )
    )
    fig_flow.add_trace(
        go.Scatter(
            x=tl_fc.index,
            y=tl_fc["p_import_mw"],
            name="Grid Import (MW)",
            line=dict(color=COLOR_AMBER, width=1.5, dash="dot"),
        )
    )
    fig_flow.update_layout(
        get_plotly_layout(
            title="Realized Power Flows: Forecast-Informed Dispatch (MW)",
            xaxis_title="UTC Timestamp",
            yaxis_title="Power (MW)",
            height=340,
        )
    )
    st.plotly_chart(fig_flow, use_container_width=True)

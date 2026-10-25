"""Page F: Methodology, Permutation Feature Importance, and Mathematical Formulations."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from ui.theme import get_plotly_layout, COLOR_CYAN, COLOR_EMERALD, COLOR_AMBER


def render_page_methodology(config: dict, technical_mode: bool):
    """Renders the methodology, explainability, and glossary documentation."""
    st.markdown("## Methodology & Explainability")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: -10px;'>"
        "Detailed mathematical formulations, feature importance, battery dispatch equations, and project glossary."
        "</p>",
        unsafe_allow_html=True,
    )

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs([
        "Feature Importance", "Optimization Formulation", "Physical Constraints", "Glossary"
    ])

    with tab_m1:
        st.markdown("### Permutation Feature Importance on Holdout Test Set")
        st.markdown(
            "Permutation feature importance measures the increase in model prediction error after shuffling each feature on the chronological test set (Jan–Sep 2020)."
        )

        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        lgb_file = models_dir / "lightgbm_model.joblib"
        clean_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "DK_1_clean.parquet"

        if lgb_file.exists() and clean_file.exists():
            from src.features import build_forecasting_features
            from src.forecasting import split_train_val_test

            clean_df = pd.read_parquet(clean_file)
            feat_df = build_forecasting_features(
                clean_df,
                target_col=config["forecasting"]["target"],
                min_lag_hours=24,
                latitude_deg=config["data"]["latitude"],
                longitude_deg=config["data"]["longitude"],
            )
            _, _, test_df = split_train_val_test(feat_df, config)

            feature_cols = [c for c in feat_df.columns if c != "target"]
            X_test = test_df[feature_cols].values
            y_test = test_df["target"].values

            model = joblib.load(lgb_file)

            # Permutation importance (subsample 1500 rows for instant rendering)
            sub_idx = np.linspace(0, len(X_test) - 1, 1500, dtype=int)
            perm = permutation_importance(
                model, X_test[sub_idx], y_test[sub_idx],
                n_repeats=3, random_state=42, scoring="neg_mean_absolute_error"
            )

            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance (MAE Drop MW)": perm.importances_mean,
            }).sort_values("Importance (MAE Drop MW)", ascending=True)

            fig_imp = go.Figure()
            fig_imp.add_trace(
                go.Bar(
                    y=imp_df["Feature"],
                    x=imp_df["Importance (MAE Drop MW)"],
                    orientation="h",
                    marker_color=COLOR_CYAN,
                )
            )
            fig_imp.update_layout(
                get_plotly_layout(
                    title="Permutation Feature Importance (LightGBM on Test Set)",
                    xaxis_title="Increase in MAE when Shuffled (MW)",
                    yaxis_title="",
                    height=450,
                )
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab_m2:
        st.markdown("### Linear Programming Optimization Formulation")
        st.markdown(
            r"""
            At day-ahead decision time ($t=0$), the controller optimizes over horizon $H=24$ hours:

            $$\min_{\{p_{\text{ch}}, p_{\text{dis}}, p_{\text{imp}}, p_{\text{exp}}\}} \sum_{t=0}^{H-1} \left( (\lambda_{\text{da}}(t) + \text{Adder}) p_{\text{imp}}(t) - (\lambda_{\text{da}}(t) \cdot \text{ExportRatio}) p_{\text{exp}}(t) + C_{\text{deg}}(p_{\text{ch}}(t) + p_{\text{dis}}(t)) \right) \Delta t$$

            **Subject to:**
            1. **Power Balance Equality**:
               $$G_{\text{fc}}(t) + p_{\text{dis}}(t) + p_{\text{imp}}(t) - p_{\text{curt}}(t) = L_{\text{fc}}(t) + p_{\text{ch}}(t) + p_{\text{exp}}(t)$$
            2. **Dynamic State-of-Charge Transitions**:
               $$\text{SoC}(t) = \text{SoC}(t-1) + \frac{\eta_{\text{ch}} p_{\text{ch}}(t) \Delta t}{E_{\text{cap}}} - \frac{p_{\text{dis}}(t) \Delta t}{\eta_{\text{dis}} E_{\text{cap}}}$$
            3. **Operational Bounds**:
               $$0 \le p_{\text{ch}}(t) \le P_{\text{max,ch}}, \quad 0 \le p_{\text{dis}}(t) \le P_{\text{max,dis}}$$
               $$0 \le p_{\text{imp}}(t) \le P_{\text{grid,imp,max}}, \quad 0 \le p_{\text{exp}}(t) \le P_{\text{grid,exp,max}}$$
               $$\text{SoC}_{\min} \le \text{SoC}(t) \le \text{SoC}_{\max}, \quad \text{SoC}(H-1) \ge \text{SoC}_{\text{term}}$$
            """
        )

    with tab_m3:
        st.markdown("### Physical Invariants & Non-Simultaneous Dispatch")
        st.markdown(
            r"""
            - **No Simultaneous Charging and Discharging**:
              Because round-trip efficiency is strictly below 100% ($\eta_{\text{ch}}\eta_{\text{dis}} < 1.0$), and the grid retail price adder ensures that buying power is strictly more expensive than selling power, any simultaneous charging and discharging incurs net physical losses and economic cost. Therefore, the optimal Linear Programming vertex strictly satisfies:
              $$\min(p_{\text{ch}}(t), p_{\text{dis}}(t)) = 0$$
            - **Energy Balance Residual**:
              $$\text{Residual}(t) = |(G(t) + P_{\text{dis}}(t) + P_{\text{imp}}(t)) - (L(t) + P_{\text{ch}}(t) + P_{\text{exp}}(t) + P_{\text{curt}}(t))| \equiv 0.0$$
            """
        )

    with tab_m4:
        st.markdown("### Technical Energy Systems Glossary")
        glossary = [
            ("MAE (Mean Absolute Error)", "Average magnitude of forecast errors in MW without considering their sign."),
            ("RMSE (Root Mean Squared Error)", "Square root of the average squared errors; penalizes large outlier errors more heavily than MAE."),
            ("nMAE (Normalized MAE)", "MAE divided by installed peak capacity, expressed as a percentage."),
            ("Skill Score", "Fractional error reduction of a candidate model relative to a naive persistence benchmark: 1 - (MAE_model / MAE_baseline)."),
            ("SoC (State of Charge)", "The current energy content of the battery expressed as a percentage of total capacity (0% to 100%)."),
            ("Round-Trip Efficiency (RTE)", "The ratio of energy retrieved during discharging to energy supplied during charging (typically 85% to 92% for lithium-ion)."),
            ("Curtailment", "Deliberate reduction in renewable generator output below available capability to prevent overgeneration or negative price penalties."),
            ("Day-Ahead Spot Price", "Wholesale electricity price cleared hourly in auction markets for delivery on the following day."),
            ("Perfect Foresight", "A theoretical benchmark assuming future weather and load demand are known with 100% accuracy."),
            ("Equivalent Full Cycle (EFC)", "Cumulative throughput normalized by battery capacity: (Energy Charged + Energy Discharged) / (2 * Capacity)."),
        ]
        for term, desc in glossary:
            st.markdown(f"**{term}**: {desc}")

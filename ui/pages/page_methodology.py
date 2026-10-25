"""Page F: Methodology, Permutation Feature Importance, and Mathematical Formulations."""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from ui.theme import (
    render_page_header,
    render_provenance_footer,
    get_plotly_layout,
    get_current_theme,
)


def render_page_methodology(config: dict, technical_mode: bool):
    """Renders the methodology, explainability, and glossary documentation."""
    summary_sentence = (
        "The modeling pipeline strictly enforces the 12:00 UTC day-ahead gate closure, "
        "normalizes targets with a causal trailing capacity proxy, solves a rolling 24-hour Linear Program "
        "via the HiGHS solver, and validates uncertainty with split-conformal coverage."
    )

    render_page_header(
        title="Methodology, Formulations & Explainability",
        subtitle="Mathematical Optimization Specs, Astronomical Ephemeris, Permutation Importance, and Energy Glossary",
        summary_sentence=summary_sentence,
        icon_name="book",
    )

    t = get_current_theme()

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs([
        "Feature Importance", "Optimization Formulation", "Capacity & Solar Ephemeris", "Glossary"
    ])

    with tab_m1:
        st.markdown("### Permutation Feature Importance (Holdout Test Period)")
        st.markdown(
            "Permutation feature importance measures the increase in model prediction error (MAE in MW) after shuffling "
            "each feature column on the test set. Features with highest impact are the strictly cutoff-safe lags and astronomical geometry."
        )

        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"
        perm_file = reports_dir / "permutation_importance.json"
        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        lgb_file = models_dir / "lightgbm_model.joblib"
        clean_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "DK_1_clean.parquet"

        imp_df = None
        if perm_file.exists():
            with open(perm_file, "r") as f:
                perm_json = json.load(f)
            imp_df = pd.DataFrame({
                "Feature": perm_json["features"],
                "Importance": perm_json["importance_mean"],
            }).sort_values("Importance", ascending=True).tail(12)
        elif lgb_file.exists() and clean_file.exists():
            from src.features import build_forecasting_features
            from src.forecasting import split_train_val_test

            clean_df = pd.read_parquet(clean_file)
            feat_df = build_forecasting_features(
                clean_df,
                target_col=config["forecasting"]["target"],
                latitude_deg=config["data"]["latitude"],
                longitude_deg=config["data"]["longitude"],
            )
            _, _, test_df = split_train_val_test(feat_df, config)

            non_feature_cols = {
                "target", "target_normalized", "is_telemetry_gap",
                "target_lag_last_available", "target_lag_24h", "target_lag_48h", "target_lag_168h",
                "target_rolling_mean_24h", "target_rolling_std_24h", "target_rolling_max_24h", "target_rolling_mean_168h",
            }
            feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]
            X_test = test_df[feature_cols].values
            y_test_norm = test_df["target_normalized"].values

            model = joblib.load(lgb_file)

            # Fast subsampled permutation importance
            sub_idx = np.linspace(0, len(X_test) - 1, 1200, dtype=int)
            perm = permutation_importance(
                model, X_test[sub_idx], y_test_norm[sub_idx],
                n_repeats=3, random_state=42, scoring="neg_mean_absolute_error"
            )

            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": perm.importances_mean,
            }).sort_values("Importance", ascending=True).tail(12)

        if imp_df is not None:
            fig_imp = go.Figure()
            fig_imp.add_trace(
                go.Bar(
                    y=imp_df["Feature"],
                    x=imp_df["Importance"],
                    orientation="h",
                    marker_color=t["accent_emerald"],
                )
            )
            fig_imp.update_layout(
                get_plotly_layout(
                    title="Top 12 Features by Permutation Importance (LightGBM)",
                    xaxis_title="MAE Increase when Feature Shuffled (Normalized)",
                    yaxis_title="",
                    height=420,
                )
            )
            st.plotly_chart(fig_imp, width="stretch")

    with tab_m2:
        st.markdown("### Linear Programming Optimization Formulation")
        st.markdown(
            r"""
            At day-ahead decision cutoff ($t=0$), the system controller solves a rolling Linear Program over horizon $H=24$ hours:

            $$\min_{\{p_{\text{ch}}, p_{\text{dis}}, p_{\text{imp}}, p_{\text{exp}}, p_{\text{curt}}\}} \sum_{t=0}^{H-1} \left( (\lambda_{\text{da}}(t) + \lambda_{\text{adder}}) p_{\text{imp}}(t) - (\lambda_{\text{da}}(t) \cdot r_{\text{exp}}) p_{\text{exp}}(t) + C_{\text{deg}}(p_{\text{ch}}(t) + p_{\text{dis}}(t)) \right) \Delta t$$

            **Subject to:**
            1. **Power Balance Equality ($\forall t$):**
               $$G_{\text{fc}}(t) + p_{\text{dis}}(t) + p_{\text{imp}}(t) - p_{\text{curt}}(t) = L_{\text{fc}}(t) + p_{\text{ch}}(t) + p_{\text{exp}}(t)$$
            2. **Dynamic State-of-Charge Transitions ($\forall t$):**
               $$\text{SoC}(t) = \text{SoC}(t-1) + \frac{\eta_{\text{ch}} \cdot p_{\text{ch}}(t) \cdot \Delta t}{E_{\text{cap}}} - \frac{p_{\text{dis}}(t) \cdot \Delta t}{\eta_{\text{dis}} \cdot E_{\text{cap}}}$$
            3. **Power Rating Limits:**
               $$0 \le p_{\text{ch}}(t) \le P_{\text{max,ch}}, \quad 0 \le p_{\text{dis}}(t) \le P_{\text{max,dis}}$$
            4. **Grid Headroom Interconnect Limits:**
               $$0 \le p_{\text{imp}}(t) \le P_{\text{grid,imp}}, \quad 0 \le p_{\text{exp}}(t) \le P_{\text{grid,exp}}$$
            5. **Physical Buffer Constraints:**
               $$\text{SoC}_{\text{min}} \le \text{SoC}(t) \le \text{SoC}_{\text{max}}, \quad \text{SoC}(H-1) \ge \text{SoC}_{\text{terminal}}$$
            6. **Negative Price & Energy Burning Protection:**
               $$0 \le p_{\text{curt}}(t) \le \max(0, G_{\text{fc}}(t)), \quad p_{\text{ch}}(t) \cdot p_{\text{dis}}(t) = 0$$
            """
        )

    with tab_m3:
        st.markdown("### Capacity Normalization & Astronomical Solar Geometry")
        st.markdown(
            r"""
            #### 1. Causal Trailing Capacity Normalization
            To eliminate the negative bias from Danish solar capacity expansion (+107% from 2015 to 2020), the target is normalized by a strictly causal trailing capacity proxy:
            $$\tilde{y}_t = \frac{y_t}{C_t}, \quad \text{where} \quad C_t = \max_{k \in [48, 8760]} y_{t-k}$$
            At forecast inference time:
            $$\hat{y}_t = \text{clip}\left(\hat{\tilde{y}}_t, 0, 1.15\right) \times C_t, \quad \text{with hard night zero: } \hat{y}_t = 0 \text{ when } \alpha_t \le 0$$

            #### 2. Astronomical Solar Position (Spencer 1971 / Iqbal)
            The solar elevation angle $\alpha_t$ and clear-sky proxy are computed deterministically from UTC timestamps:
            $$\gamma = \frac{2\pi}{365.25} (d_{\text{oy}} - 1 + \frac{h - 12}{24})$$
            $$\delta = 0.006918 - 0.399912\cos\gamma + 0.070257\sin\gamma - 0.006758\cos 2\gamma + \dots$$
            $$\sin\alpha_t = \sin\phi\sin\delta + \cos\phi\cos\delta\cos\omega_t$$
            """
        )

    with tab_m4:
        st.markdown("### Technical Glossary of Terms")
        glossary = [
            ("Day-Ahead Gate Closure", "The 12:00 UTC market cutoff when electricity bids for delivery day D+1 are submitted to Nord Pool / EPEX Spot."),
            ("Last-Available-Day Baseline", "The strictly non-anticipative baseline using lag 24h for hours < 12 UTC and lag 48h for hours >= 12 UTC."),
            ("Round-Trip Efficiency (RTE)", "The percentage of energy retrieved from the battery relative to the energy supplied to charge it."),
            ("Equivalent Full Cycles (EFC)", "Cumulative throughput normalized by battery capacity: EFC = (Total Charged + Total Discharged) / (2 * Capacity)."),
            ("Split-Conformal Prediction", "A distribution-free method that provides finite-sample coverage guarantees without parametric distribution assumptions."),
            ("HiGHS Solver", "A state-of-the-art open-source dual simplex and interior-point solver used via scipy.optimize.linprog."),
            ("Negative Electricity Pricing", "Hours where wholesale spot prices fall below 0 EUR/MWh due to high renewable output and low demand."),
            ("Energy Burning", "An unphysical pathology where an optimizer imports negative-priced power and immediately curtails it; prevented by bounding curtailment to available solar."),
        ]
        df_g = pd.DataFrame(glossary, columns=["Term", "Definition"])
        st.dataframe(df_g, width="stretch", hide_index=True)

    render_provenance_footer()

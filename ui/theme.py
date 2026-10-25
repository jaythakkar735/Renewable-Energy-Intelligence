"""Theme definitions, color tokens, Plotly templates, and UI components."""
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go
from ui.icons import get_icon

# Light Theme Tokens (Default)
THEME_LIGHT = {
    "name": "light",
    "bg_app": "#F4F7FB",
    "bg_surface": "#FFFFFF",
    "bg_card": "#FFFFFF",
    "border": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_muted": "#475569",
    "text_subtle": "#64748B",
    # Strategy Colors
    "no_battery": "#64748B",
    "rule_based": "#2563EB",
    "lp_naive": "#7C3AED",
    "lp_ml": "#059669",
    "perfect_foresight": "#10B981",
    "color_no_battery": "#64748B",
    "color_rule_based": "#2563EB",
    "color_lp_naive": "#7C3AED",
    "color_lp_ml": "#059669",
    "color_perfect_foresight": "#10B981",
    # Accents (WCAG AA Compliant >= 4.5:1)
    "accent_emerald": "#047857",
    "accent_cyan": "#0E7490",
    "accent_amber": "#92400E",
    "accent_red": "#B91C1C",
    "accent_blue": "#1D4ED8",
}

# Dark Theme Tokens (Option)
THEME_DARK = {
    "name": "dark",
    "bg_app": "#0B1220",
    "bg_surface": "#111B2E",
    "bg_card": "#111B2E",
    "border": "#1E293B",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "text_subtle": "#64748B",
    # Strategy Colors
    "no_battery": "#94A3B8",
    "rule_based": "#3B82F6",
    "lp_naive": "#8B5CF6",
    "lp_ml": "#10B981",
    "perfect_foresight": "#34D399",
    "color_no_battery": "#94A3B8",
    "color_rule_based": "#3B82F6",
    "color_lp_naive": "#8B5CF6",
    "color_lp_ml": "#10B981",
    "color_perfect_foresight": "#34D399",
    # Accents
    "accent_emerald": "#34D399",
    "accent_cyan": "#38BDF8",
    "accent_amber": "#FBBF24",
    "accent_red": "#F87171",
    "accent_blue": "#60A5FA",
}


def get_current_theme() -> Dict[str, str]:
    """Returns the currently active theme token dictionary."""
    mode = st.session_state.get("theme_mode", "light")
    return THEME_DARK if mode == "dark" else THEME_LIGHT


def apply_custom_css():
    """Injects custom CSS stylesheets and sets dark/light mode class."""
    css_file = Path(__file__).resolve().parent.parent / "assets" / "css" / "styles.css"
    mode = st.session_state.get("theme_mode", "light")
    theme_cls = "dark-mode" if mode == "dark" else "light-mode"

    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(
            f"""
            <style>
            {css}
            </style>
            <script>
            document.body.className = "{theme_cls}";
            document.documentElement.setAttribute('data-theme', '{mode}');
            </script>
            """,
            unsafe_allow_html=True,
        )


def get_plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 400,
) -> dict:
    """Returns a standardized Plotly layout dictionary honoring the active theme."""
    t = get_current_theme()
    return dict(
        title=dict(text=title, font=dict(family="sans-serif", size=14, color=t["text_primary"])),
        paper_bgcolor=t["bg_surface"],
        plot_bgcolor=t["bg_surface"],
        height=height,
        margin=dict(l=45, r=20, t=45, b=40),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=t["text_muted"], size=12)),
            tickfont=dict(color=t["text_muted"], size=11),
            gridcolor=t["border"],
            zerolinecolor=t["border"],
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=t["text_muted"], size=12)),
            tickfont=dict(color=t["text_muted"], size=11),
            gridcolor=t["border"],
            zerolinecolor=t["border"],
            showgrid=True,
        ),
        legend=dict(
            font=dict(color=t["text_primary"], size=11),
            bgcolor=t["bg_surface"],
            bordercolor=t["border"],
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hoverlabel=dict(
            bgcolor=t["bg_surface"],
            font_size=12,
            font_family="sans-serif",
            bordercolor=t["border"],
        ),
    )


def render_page_header(title: str, subtitle: str, summary_sentence: str, icon_name: str = "activity"):
    """Renders the top banner on every page with SVG icon, title, subtitle, and auto-generated summary."""
    icon_svg = get_icon(icon_name, color="var(--accent-blue-text)", size=24)
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-title">{icon_svg}{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="page-summary-banner">
                {get_icon("info", color="var(--accent-cyan-text)", size=16)}
                <strong>Key Takeaway:</strong> {summary_sentence}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: str,
    value: str,
    sub_text: str = "",
    pill_text: str = "",
    pill_color: str = "cyan",
    icon_name: Optional[str] = None,
):
    """Renders a styled KPI card with optional SVG icon and status pill."""
    pill_html = ""
    if pill_text:
        pill_html = f"<span class='status-pill status-pill-{pill_color}'>{pill_text}</span>"

    icon_html = ""
    if icon_name:
        icon_html = get_icon(icon_name, size=15)

    html = f"""
    <div class="analytics-card">
        <div class="analytics-card-title">
            <span>{icon_html}{title}</span>
            {pill_html}
        </div>
        <div class="analytics-card-value tabular-nums">{value}</div>
        <div class="analytics-card-sub">{sub_text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_provenance_footer():
    """Renders data provenance footer at bottom of page."""
    html = """
    <div class="provenance-footer">
        <div><strong>Dataset:</strong> Open Power System Data (OPSD) Time Series (DK_1 Bidding Zone)</div>
        <div><strong>Evaluation Period:</strong> 2020-01-01 to 2020-09-30 (Holdout Test Set)</div>
        <div><strong>Auction Cutoff:</strong> 12:00 UTC Day-Ahead Gate Closure</div>
        <div><strong>Code Version:</strong> Git Commit <code>Upgrade-Pass-2026</code></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

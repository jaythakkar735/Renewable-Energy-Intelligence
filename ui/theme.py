"""Theme definitions, color tokens, and Plotly templates."""
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go


COLOR_NAVY_BG = "#0B1220"
COLOR_NAVY_CARD = "#111B2E"
COLOR_BORDER = "#1E293B"
COLOR_EMERALD = "#10B981"
COLOR_CYAN = "#22D3EE"
COLOR_BLUE = "#3B82F6"
COLOR_AMBER = "#F59E0B"
COLOR_RED = "#EF4444"
COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"


def apply_custom_css():
    """Injects custom CSS stylesheets into the Streamlit app."""
    css_file = Path(__file__).resolve().parent.parent / "assets" / "css" / "styles.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 400,
) -> dict:
    """Returns a standardized dark Plotly layout dict."""
    return dict(
        title=dict(text=title, font=dict(family="sans-serif", size=15, color=COLOR_TEXT_PRIMARY)),
        paper_bgcolor=COLOR_NAVY_CARD,
        plot_bgcolor=COLOR_NAVY_CARD,
        height=height,
        margin=dict(l=45, r=20, t=45, b=40),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=COLOR_TEXT_MUTED, size=12)),
            tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=COLOR_TEXT_MUTED, size=12)),
            tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            showgrid=True,
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT_PRIMARY, size=11),
            bgcolor="rgba(17, 27, 46, 0.7)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hoverlabel=dict(
            bgcolor=COLOR_NAVY_BG,
            font_size=12,
            font_family="sans-serif",
            bordercolor=COLOR_BORDER,
        ),
    )


def render_metric_card(
    title: str,
    value: str,
    sub_text: str = "",
    pill_text: str = "",
    pill_color: str = "cyan",
):
    """Renders a styled KPI card."""
    pill_html = ""
    if pill_text:
        pill_html = f"<span class='status-pill status-pill-{pill_color}' style='float: right;'>{pill_text}</span>"

    html = f"""
    <div class="analytics-card">
        <div class="analytics-card-title">{title} {pill_html}</div>
        <div class="analytics-card-value">{value}</div>
        <div class="analytics-card-sub">{sub_text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

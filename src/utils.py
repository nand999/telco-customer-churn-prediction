"""
Utility functions and theme configuration for the Churn Prediction Dashboard.
"""

import streamlit as st
import pandas as pd
import os

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "primary_light": "#8B83FF",
    "primary_dark": "#4A42CC",
    "accent": "#00D4AA",
    "accent2": "#FF6B6B",
    "accent3": "#FFD93D",
    "accent4": "#4ECDC4",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1D23",
    "bg_card_hover": "#22252D",
    "text_primary": "#FAFAFA",
    "text_secondary": "#B0B0B0",
    "success": "#00D4AA",
    "warning": "#FFD93D",
    "danger": "#FF6B6B",
    "info": "#6C63FF",
    "gradient_start": "#6C63FF",
    "gradient_end": "#00D4AA",
}

CHART_COLORS = [
    "#6C63FF", "#00D4AA", "#FF6B6B", "#FFD93D",
    "#4ECDC4", "#FF8A5C", "#A78BFA", "#F472B6",
    "#34D399", "#FBBF24", "#60A5FA", "#FB923C",
]

def base_layout(**overrides):
    """
    Build a Plotly layout dict with dark-theme defaults.
    Any key passed as an override replaces (not duplicates) the default.
    """
    defaults = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#FAFAFA", "family": "Inter, sans-serif"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.06)",
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.06)",
        },
        "colorway": CHART_COLORS,
        "margin": {"t": 40, "b": 40, "l": 40, "r": 20},
    }
    # Deep-merge dicts one level (for xaxis, yaxis, margin, font)
    merged = {}
    for key, val in defaults.items():
        if key in overrides:
            if isinstance(val, dict) and isinstance(overrides[key], dict):
                merged[key] = {**val, **overrides[key]}
            else:
                merged[key] = overrides[key]
        else:
            merged[key] = val
    # Add any extra keys from overrides not in defaults
    for key, val in overrides.items():
        if key not in merged:
            merged[key] = val
    return merged


def get_data_path():
    """Return the absolute path to the dataset CSV."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def get_models_dir():
    """Return the path to the models directory, creating it if needed."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


@st.cache_data
def load_data():
    """Load and return the raw telco customer churn dataset."""
    df = pd.read_csv(get_data_path())
    return df


def apply_custom_css():
    """Inject premium CSS styling into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ── Gradient Page Header ── */
    .page-header {
        background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .page-subtitle {
        color: #B0B0B0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, #1A1D23 0%, #22252D 100%);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6C63FF, #00D4AA);
        border-radius: 16px 16px 0 0;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(108, 99, 255, 0.4);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.3rem 0;
    }

    .metric-label {
        color: #B0B0B0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(26, 29, 35, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .glass-card h3 {
        color: #FAFAFA;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* ── Section Header ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(108, 99, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Risk Badges ── */
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    .risk-low {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }

    .risk-medium {
        background: rgba(255, 217, 61, 0.15);
        color: #FFD93D;
        border: 1px solid rgba(255, 217, 61, 0.3);
    }

    .risk-high {
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }

    .risk-critical {
        background: rgba(255, 56, 56, 0.2);
        color: #FF3838;
        border: 1px solid rgba(255, 56, 56, 0.4);
    }

    /* ── Insight Box ── */
    .insight-box {
        background: linear-gradient(145deg, rgba(108, 99, 255, 0.08) 0%, rgba(0, 212, 170, 0.05) 100%);
        border-left: 4px solid #6C63FF;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #FAFAFA;
    }

    .insight-box .insight-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #8B83FF;
        margin-bottom: 0.4rem;
    }

    .insight-box .insight-text {
        color: #B0B0B0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Recommendation Card ── */
    .rec-card {
        background: linear-gradient(145deg, #1A1D23 0%, #22252D 100%);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }

    .rec-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }

    .rec-title {
        font-weight: 700;
        font-size: 1rem;
        color: #FAFAFA;
        margin-bottom: 0.3rem;
    }

    .rec-desc {
        color: #B0B0B0;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Styled Table ── */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #4A42CC) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(108, 99, 255, 0.35) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151820 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.1);
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6C63FF, #00D4AA) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        background: transparent;
        border: 1px solid rgba(108, 99, 255, 0.15);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(108, 99, 255, 0.15) !important;
        border-color: #6C63FF !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(26, 29, 35, 0.6);
        border-radius: 10px;
    }

    /* ── Divider ── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* ── Selectbox / Input ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div {
        border-radius: 10px !important;
    }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .slide-in {
        animation: slideIn 0.4s ease-out forwards;
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title, subtitle):
    """Render a styled gradient page header."""
    st.markdown(f"""
    <div class="fade-in">
        <div class="page-header">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(icon, value, label):
    """Render a single metric card with icon, value, and label."""
    return f"""
    <div class="metric-card fade-in">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_insight_box(title, text):
    """Render a styled insight box."""
    st.markdown(f"""
    <div class="insight-box slide-in">
        <div class="insight-title">💡 {title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(risk_level):
    """Return HTML for a risk badge."""
    css_class = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
        "Critical": "risk-critical",
    }.get(risk_level, "risk-medium")
    return f'<span class="risk-badge {css_class}">{risk_level} Risk</span>'


def get_risk_level(probability):
    """Map a churn probability to a risk level string."""
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Critical"


def plotly_config():
    """Return standard Plotly chart display config."""
    return {
        "displayModeBar": False,
        "staticPlot": False,
    }

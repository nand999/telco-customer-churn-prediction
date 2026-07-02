"""
Customer Churn Prediction Dashboard — Home Page
A premium Streamlit dashboard for telco customer churn analysis and prediction.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import apply_custom_css, load_data, render_page_header, render_metric_card

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🔮</div>
        <div style="font-size: 1.1rem; font-weight: 700; 
             background: linear-gradient(135deg, #6C63FF, #00D4AA);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text;">Churn Predictor</div>
        <div style="color: #666; font-size: 0.75rem; margin-top: 0.2rem;">v1.0 • Telco Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="padding: 0.5rem;">
        <div style="color: #B0B0B0; font-size: 0.8rem; margin-bottom: 0.8rem;">NAVIGATION</div>
        <div style="color: #FAFAFA; font-size: 0.9rem; line-height: 2.2;">
            📊 Exploratory Data Analysis<br>
            🤖 Model Training Pipeline<br>
            📈 Model Performance<br>
            🔮 Real-Time Prediction<br>
            📁 Batch Prediction<br>
            💡 Business Insights
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
render_page_header(
    "Customer Churn Prediction",
    "AI-powered analytics to predict and prevent customer churn in telecom industry"
)

# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_data()

# ─── Hero Metrics ─────────────────────────────────────────────────────────────
churn_count = df["Churn"].value_counts()
churn_rate = (churn_count.get("Yes", 0) / len(df)) * 100
avg_tenure = df["tenure"].mean()
avg_monthly = df["MonthlyCharges"].mean()
total_revenue = df["TotalCharges"].replace(" ", "0").astype(float).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(render_metric_card("👥", f"{len(df):,}", "Total Customers"), unsafe_allow_html=True)
with col2:
    st.markdown(render_metric_card("📉", f"{churn_rate:.1f}%", "Churn Rate"), unsafe_allow_html=True)
with col3:
    st.markdown(render_metric_card("📅", f"{avg_tenure:.0f} mo", "Avg Tenure"), unsafe_allow_html=True)
with col4:
    st.markdown(render_metric_card("💰", f"${avg_monthly:.0f}", "Avg Monthly"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Project Overview ────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Project Overview</h3>
        <p style="color: #B0B0B0; line-height: 1.8; font-size: 0.95rem;">
            This dashboard provides a comprehensive machine learning pipeline for predicting 
            customer churn in the telecommunications industry. Using the IBM Telco Customer Churn 
            dataset with <strong style="color: #6C63FF;">7,043 customers</strong> and 
            <strong style="color: #00D4AA;">21 features</strong>, we train, evaluate, and deploy 
            multiple classification models to identify customers at risk of leaving.
        </p>
        <br>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;">
            <div style="background: rgba(108,99,255,0.08); border-radius: 10px; padding: 0.8rem;">
                <div style="color: #8B83FF; font-weight: 600; font-size: 0.85rem;">🤖 ML Models</div>
                <div style="color: #B0B0B0; font-size: 0.82rem; margin-top: 0.3rem;">
                    Logistic Regression, Random Forest, XGBoost, SVM
                </div>
            </div>
            <div style="background: rgba(0,212,170,0.08); border-radius: 10px; padding: 0.8rem;">
                <div style="color: #00D4AA; font-weight: 600; font-size: 0.85rem;">🔍 Validation</div>
                <div style="color: #B0B0B0; font-size: 0.82rem; margin-top: 0.3rem;">
                    GridSearchCV with 5-Fold Cross Validation
                </div>
            </div>
            <div style="background: rgba(255,107,107,0.08); border-radius: 10px; padding: 0.8rem;">
                <div style="color: #FF6B6B; font-weight: 600; font-size: 0.85rem;">📊 Analysis</div>
                <div style="color: #B0B0B0; font-size: 0.82rem; margin-top: 0.3rem;">
                    EDA, Feature Importance, Business Insights
                </div>
            </div>
            <div style="background: rgba(255,217,61,0.08); border-radius: 10px; padding: 0.8rem;">
                <div style="color: #FFD93D; font-weight: 600; font-size: 0.85rem;">🔮 Prediction</div>
                <div style="color: #B0B0B0; font-size: 0.82rem; margin-top: 0.3rem;">
                    Real-time & Batch Prediction with Risk Scoring
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <h3>📋 Dataset Summary</h3>
    </div>
    """, unsafe_allow_html=True)

    # Quick dataset stats
    st.markdown(f"""
    <div style="padding: 0 1rem;">
        <div style="display: flex; justify-content: space-between; padding: 0.6rem 0; 
             border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="color: #B0B0B0;">Rows</span>
            <span style="color: #FAFAFA; font-weight: 600;">{len(df):,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.6rem 0;
             border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="color: #B0B0B0;">Features</span>
            <span style="color: #FAFAFA; font-weight: 600;">{len(df.columns)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.6rem 0;
             border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="color: #B0B0B0;">Numeric Features</span>
            <span style="color: #FAFAFA; font-weight: 600;">3</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.6rem 0;
             border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="color: #B0B0B0;">Categorical Features</span>
            <span style="color: #FAFAFA; font-weight: 600;">17</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.6rem 0;
             border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="color: #B0B0B0;">Churned Customers</span>
            <span style="color: #FF6B6B; font-weight: 600;">{churn_count.get("Yes", 0):,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.6rem 0;">
            <span style="color: #B0B0B0;">Retained Customers</span>
            <span style="color: #00D4AA; font-weight: 600;">{churn_count.get("No", 0):,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Quick Start Guide ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">
    <h3>🚀 Quick Start Guide</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.5rem;">
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">1️⃣</div>
            <div style="color: #FAFAFA; font-weight: 600; margin-bottom: 0.3rem;">Explore Data</div>
            <div style="color: #B0B0B0; font-size: 0.82rem;">
                Navigate to the EDA page to understand distributions, correlations, and churn patterns
            </div>
        </div>
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">2️⃣</div>
            <div style="color: #FAFAFA; font-weight: 600; margin-bottom: 0.3rem;">Train Models</div>
            <div style="color: #B0B0B0; font-size: 0.82rem;">
                Go to Model Training to run the full ML pipeline with hyperparameter tuning
            </div>
        </div>
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">3️⃣</div>
            <div style="color: #FAFAFA; font-weight: 600; margin-bottom: 0.3rem;">Predict & Act</div>
            <div style="color: #B0B0B0; font-size: 0.82rem;">
                Use Prediction pages for real-time scoring and Business Insights for retention strategies
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Data Preview ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True,
)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;">
    Built with Streamlit • Machine Learning Pipeline • © 2025 Customer Churn Prediction
</div>
""", unsafe_allow_html=True)

"""
Page 4: Real-Time Prediction
Interactive form to input customer data and predict churn probability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    apply_custom_css, render_page_header, render_metric_card, render_risk_badge,
    get_risk_level, COLORS, base_layout, plotly_config
)
from src.model_training import load_model_artifacts
from src.data_preprocessing import preprocess_single_customer

st.set_page_config(page_title="Prediction | Churn Prediction", page_icon="🔮", layout="wide")
apply_custom_css()

render_page_header(
    "Real-Time Prediction",
    "Enter customer details to predict churn probability with confidence scoring"
)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
artifacts = load_model_artifacts()

if artifacts.get("best_model") is None:
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #FFD93D; text-align: center; padding: 3rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
        <div style="color: #FFD93D; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.5rem;">
            No Trained Model Found
        </div>
        <div style="color: #B0B0B0; font-size: 0.9rem;">
            Please go to the <strong>Model Training</strong> page first to train the models.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model = artifacts["best_model"]
scaler = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
feature_names = artifacts["feature_names"]
best_model_name = artifacts.get("best_model_name", "Best Model")

st.markdown(f"""
<div class="glass-card" style="border-left: 4px solid #6C63FF;">
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.2rem;">🤖</span>
        <span style="color: #B0B0B0;">Active Model:</span>
        <span style="color: #6C63FF; font-weight: 700;">{best_model_name}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Customer Input Form ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Customer Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="pred_senior")
    partner = st.selectbox("Partner", ["Yes", "No"], key="pred_partner")
    dependents = st.selectbox("Dependents", ["Yes", "No"], key="pred_dependents")

with col2:
    st.markdown("**Account Info**")
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="pred_tenure")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="pred_contract")
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="pred_paperless")
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], key="pred_payment")

with col3:
    st.markdown("**Charges**")
    monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5, key="pred_monthly")
    total = st.slider("Total Charges ($)", 18.0, 9000.0, float(monthly * tenure), 10.0, key="pred_total")

st.markdown("<br>", unsafe_allow_html=True)

# Services
st.markdown('<div class="section-header">📡 Services</div>', unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    phone = st.selectbox("Phone Service", ["Yes", "No"], key="pred_phone")
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="pred_multiple")

with col_s2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="pred_internet")
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key="pred_security")

with col_s3:
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="pred_backup")
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="pred_protection")

with col_s4:
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="pred_tech")
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="pred_tv")

streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="pred_movies")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Predict Button ──────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔮 Predict Churn", use_container_width=True, key="predict_btn")

if predict_clicked:
    # Build input dict
    input_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    # Preprocess
    input_processed = preprocess_single_customer(input_data, label_encoders, scaler, feature_names)

    # Predict
    churn_proba = model.predict_proba(input_processed)[0][1]
    churn_pred = model.predict(input_processed)[0]
    risk_level = get_risk_level(churn_proba)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

    # Results row
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown(render_metric_card(
            "🎯",
            f"{churn_proba*100:.1f}%",
            "Churn Probability"
        ), unsafe_allow_html=True)

    with col_r2:
        confidence = max(churn_proba, 1 - churn_proba) * 100
        st.markdown(render_metric_card(
            "🛡️",
            f"{confidence:.1f}%",
            "Confidence Score"
        ), unsafe_allow_html=True)

    with col_r3:
        prediction_text = "WILL CHURN" if churn_pred == 1 else "WILL STAY"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div class="metric-icon">⚡</div>
            <div style="margin: 0.5rem 0;">
                {render_risk_badge(risk_level)}
            </div>
            <div class="metric-label">{prediction_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart
    col_gauge, col_factors = st.columns([1, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_proba * 100,
            number=dict(suffix="%", font=dict(size=40, color="#FAFAFA")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#B0B0B0", tickfont=dict(color="#B0B0B0")),
                bar=dict(color=COLORS["primary"]),
                bgcolor="rgba(26,29,35,0.8)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 25], color="rgba(0,212,170,0.15)"),
                    dict(range=[25, 50], color="rgba(255,217,61,0.15)"),
                    dict(range=[50, 75], color="rgba(255,107,107,0.15)"),
                    dict(range=[75, 100], color="rgba(255,56,56,0.2)"),
                ],
                threshold=dict(
                    line=dict(color="#FF6B6B", width=3),
                    thickness=0.8,
                    value=churn_proba * 100
                ),
            ),
        ))
        fig_gauge.update_layout(
            **base_layout(margin=dict(t=30, b=0, l=30, r=30)),
            height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config=plotly_config())

    with col_factors:
        # Key contributing factors
        st.markdown("""
        <div class="glass-card">
            <h3 style="font-size: 1rem;">🔍 Key Risk Factors</h3>
        </div>
        """, unsafe_allow_html=True)

        factors = []
        if contract == "Month-to-month":
            factors.append(("⚠️", "Month-to-month contract", "High risk factor — no commitment lock-in"))
        if internet == "Fiber optic":
            factors.append(("⚠️", "Fiber optic internet", "Historically higher churn segment"))
        if payment == "Electronic check":
            factors.append(("⚠️", "Electronic check payment", "Less committed payment method"))
        if tenure < 12:
            factors.append(("⚠️", f"Short tenure ({tenure} months)", "New customers churn more"))
        if senior == "Yes":
            factors.append(("⚠️", "Senior citizen", "Higher churn demographic"))
        if tech == "No":
            factors.append(("⚠️", "No tech support", "Missing retention anchor"))
        if security == "No":
            factors.append(("⚠️", "No online security", "Missing retention anchor"))
        if monthly > 80:
            factors.append(("⚠️", f"High monthly charges (${monthly:.0f})", "Price sensitivity factor"))

        # Positive factors
        if contract in ["One year", "Two year"]:
            factors.append(("✅", f"{contract} contract", "Strong retention commitment"))
        if tenure >= 36:
            factors.append(("✅", f"Long tenure ({tenure} months)", "Loyal customer signal"))
        if tech == "Yes":
            factors.append(("✅", "Has tech support", "Service stickiness"))

        for icon, title, desc in factors[:8]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.8rem; padding: 0.5rem 0;
                 border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="font-size: 1.1rem;">{icon}</span>
                <div>
                    <div style="color: #FAFAFA; font-weight: 600; font-size: 0.88rem;">{title}</div>
                    <div style="color: #666; font-size: 0.78rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💡 Retention Recommendations</div>', unsafe_allow_html=True)

    if risk_level in ["High", "Critical"]:
        st.markdown("""
        <div class="rec-card" style="border-left: 4px solid #FF6B6B;">
            <div class="rec-title">🚨 Immediate Action Required</div>
            <div class="rec-desc">
                This customer is at <strong>high risk</strong> of churning. Consider offering a personalized 
                retention package: contract upgrade discount, loyalty rewards, or complimentary service add-ons.
                Assign a dedicated account manager for proactive outreach.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif risk_level == "Medium":
        st.markdown("""
        <div class="rec-card" style="border-left: 4px solid #FFD93D;">
            <div class="rec-title">⚡ Proactive Engagement Needed</div>
            <div class="rec-desc">
                This customer shows moderate churn risk. Schedule a satisfaction check-in call, 
                offer service bundles at a discount, and highlight the value of long-term contracts.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rec-card" style="border-left: 4px solid #00D4AA;">
            <div class="rec-title">✅ Low Risk — Upsell Opportunity</div>
            <div class="rec-desc">
                This customer is likely to stay. Focus on upselling premium services, 
                referral programs, and long-term contract renewals to maximize lifetime value.
            </div>
        </div>
        """, unsafe_allow_html=True)

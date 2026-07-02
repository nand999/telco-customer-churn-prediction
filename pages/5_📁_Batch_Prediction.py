"""
Page 5: Batch Prediction
Upload CSV for bulk churn prediction and download results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    apply_custom_css, render_page_header, render_metric_card,
    render_risk_badge, get_risk_level,
    COLORS, CHART_COLORS, base_layout, plotly_config
)
from src.model_training import load_model_artifacts
from src.data_preprocessing import preprocess_batch

st.set_page_config(page_title="Batch Prediction | Churn Prediction", page_icon="📁", layout="wide")
apply_custom_css()

render_page_header(
    "Batch Prediction",
    "Upload a CSV file with customer data to predict churn probability in bulk"
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

# ─── Upload Section ──────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <h3>📤 Upload Customer Data</h3>
    <p style="color: #B0B0B0; font-size: 0.9rem;">
        Upload a CSV file with the same format as the training data. 
        Required columns: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
        TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
        MonthlyCharges, TotalCharges.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="batch_upload",
    help="Upload a CSV with customer data matching the training format."
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # ─── Data Preview ────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(render_metric_card("📊", f"{len(batch_df):,}", "Total Rows"), unsafe_allow_html=True)
        with col_info2:
            st.markdown(render_metric_card("📐", f"{len(batch_df.columns)}", "Columns"), unsafe_allow_html=True)
        with col_info3:
            missing = batch_df.isnull().sum().sum()
            st.markdown(render_metric_card("❓", f"{missing}", "Missing Values"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

        # ─── Validate columns ────────────────────────────────────────────
        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]

        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        # ─── Run Predictions ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            predict_clicked = st.button("🚀 Run Batch Predictions", use_container_width=True, key="batch_predict_btn")

        if predict_clicked:
            with st.spinner("Processing predictions..."):
                # Save customer IDs if present
                customer_ids = None
                if "customerID" in batch_df.columns:
                    customer_ids = batch_df["customerID"].copy()

                # Preprocess
                processed = preprocess_batch(batch_df, label_encoders, scaler, feature_names)

                # Predict
                probabilities = model.predict_proba(processed)[:, 1]
                predictions = model.predict(processed)

            # Build results
            results_df = pd.DataFrame()
            if customer_ids is not None:
                results_df["CustomerID"] = customer_ids
            results_df["Churn_Probability"] = np.round(probabilities * 100, 2)
            results_df["Prediction"] = ["Churn" if p == 1 else "Retain" for p in predictions]
            results_df["Risk_Level"] = [get_risk_level(p) for p in probabilities]
            results_df["Monthly_Charges"] = batch_df["MonthlyCharges"].values
            results_df["Tenure"] = batch_df["tenure"].values
            results_df["Contract"] = batch_df["Contract"].values

            # ─── Results Summary ──────────────────────────────────────────
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

            churn_predicted = (predictions == 1).sum()
            retain_predicted = (predictions == 0).sum()
            avg_proba = probabilities.mean() * 100
            high_risk = sum(1 for p in probabilities if p >= 0.5)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(render_metric_card("📉", f"{churn_predicted}", "Predicted Churn"), unsafe_allow_html=True)
            with col_m2:
                st.markdown(render_metric_card("✅", f"{retain_predicted}", "Predicted Retain"), unsafe_allow_html=True)
            with col_m3:
                st.markdown(render_metric_card("📊", f"{avg_proba:.1f}%", "Avg Churn Prob"), unsafe_allow_html=True)
            with col_m4:
                st.markdown(render_metric_card("🚨", f"{high_risk}", "High Risk"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk distribution chart
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                risk_counts = results_df["Risk_Level"].value_counts()
                risk_order = ["Low", "Medium", "High", "Critical"]
                risk_colors = [COLORS["accent"], COLORS["accent3"], COLORS["accent2"], "#FF3838"]

                fig_risk = go.Figure(data=[go.Pie(
                    labels=[r for r in risk_order if r in risk_counts.index],
                    values=[risk_counts.get(r, 0) for r in risk_order if r in risk_counts.index],
                    hole=0.55,
                    marker=dict(colors=[risk_colors[risk_order.index(r)] for r in risk_order if r in risk_counts.index]),
                    textinfo="percent+value",
                    textfont=dict(size=13, color="white"),
                )])
                fig_risk.update_layout(
                    **base_layout(),
                    title=dict(text="Risk Distribution", font=dict(size=14, color="#FAFAFA")),
                    height=380,
                )
                st.plotly_chart(fig_risk, use_container_width=True, config=plotly_config())

            with col_chart2:
                fig_hist = px.histogram(
                    results_df, x="Churn_Probability",
                    nbins=30,
                    color_discrete_sequence=[COLORS["primary"]],
                )
                fig_hist.update_layout(
                    **base_layout(),
                    title=dict(text="Churn Probability Distribution", font=dict(size=14, color="#FAFAFA")),
                    xaxis_title="Churn Probability (%)",
                    yaxis_title="Count",
                    height=380,
                )
                st.plotly_chart(fig_hist, use_container_width=True, config=plotly_config())

            # Results table
            st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)

            # Sort by probability descending
            results_display = results_df.sort_values("Churn_Probability", ascending=False)
            st.dataframe(results_display, use_container_width=True, hide_index=True)

            # ─── Download ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)

            csv_output = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_output,
                file_name="churn_predictions.csv",
                mime="text/csv",
                key="download_btn",
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

else:
    # Show sample format
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3 style="font-size: 1rem;">📝 Expected CSV Format</h3>
        <p style="color: #B0B0B0; font-size: 0.85rem; margin-top: 0.5rem;">
            Your CSV should contain the following columns (same as the training dataset):
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample_data = pd.DataFrame({
        "customerID": ["1234-ABCDE"],
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [24],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [70.50],
        "TotalCharges": [1692.00],
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    st.info("💡 **Tip:** You can also use the original training dataset file to test batch predictions.")

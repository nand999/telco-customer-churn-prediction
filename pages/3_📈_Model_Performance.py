"""
Page 3: Model Performance
Confusion matrix, ROC-AUC curve, precision-recall curve, feature importance, classification report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    apply_custom_css, render_page_header, render_metric_card,
    COLORS, CHART_COLORS, base_layout, plotly_config
)
from src.model_training import load_model_artifacts

st.set_page_config(page_title="Performance | Churn Prediction", page_icon="📈", layout="wide")
apply_custom_css()

render_page_header(
    "Model Performance",
    "In-depth evaluation metrics, curves, and feature importance for all trained models"
)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
artifacts = load_model_artifacts()

if artifacts.get("evaluations") is None:
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #FFD93D; text-align: center; padding: 3rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
        <div style="color: #FFD93D; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.5rem;">
            No Trained Models Found
        </div>
        <div style="color: #B0B0B0; font-size: 0.9rem;">
            Please go to the <strong>Model Training</strong> page first to train the models.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

evaluations = artifacts["evaluations"]
all_models = artifacts.get("all_models", {})
best_model_name = artifacts.get("best_model_name", "")
comparison_df = artifacts.get("comparison_df")

# ─── Model Selector ──────────────────────────────────────────────────────────
model_names = list(evaluations.keys())
selected_model = st.selectbox(
    "Select Model to Analyze",
    model_names,
    index=model_names.index(best_model_name) if best_model_name in model_names else 0,
    key="model_selector"
)

eval_data = evaluations[selected_model]

# ─── Key Metrics ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(render_metric_card("🎯", f"{eval_data['accuracy']:.3f}", "Accuracy"), unsafe_allow_html=True)
with col2:
    st.markdown(render_metric_card("🔍", f"{eval_data['precision']:.3f}", "Precision"), unsafe_allow_html=True)
with col3:
    st.markdown(render_metric_card("📡", f"{eval_data['recall']:.3f}", "Recall"), unsafe_allow_html=True)
with col4:
    st.markdown(render_metric_card("⚡", f"{eval_data['f1']:.3f}", "F1-Score"), unsafe_allow_html=True)
with col5:
    st.markdown(render_metric_card("📊", f"{eval_data['roc_auc']:.3f}", "ROC-AUC"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC & PR Curves", "🏗️ Feature Importance", "📋 Classification Report"])

# ─── TAB 1: Confusion Matrix ─────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-header">📊 Confusion Matrix — {selected_model}</div>', unsafe_allow_html=True)

    col_cm1, col_cm2 = st.columns([2, 1])

    with col_cm1:
        cm = eval_data["confusion_matrix"]
        labels = ["Retained (0)", "Churned (1)"]

        # Annotated heatmap
        text = [[f"TN<br><span style='font-size:1.8rem;font-weight:800'>{cm[0][0]}</span>",
                 f"FP<br><span style='font-size:1.8rem;font-weight:800'>{cm[0][1]}</span>"],
                [f"FN<br><span style='font-size:1.8rem;font-weight:800'>{cm[1][0]}</span>",
                 f"TP<br><span style='font-size:1.8rem;font-weight:800'>{cm[1][1]}</span>"]]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#1A1D23"], [0.5, "#4A42CC"], [1, "#6C63FF"]],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        fig_cm.update_layout(
            **base_layout(yaxis=dict(autorange="reversed")),
            title=dict(text="Prediction Results Matrix", font=dict(size=16, color="#FAFAFA")),
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            height=450,
        )
        st.plotly_chart(fig_cm, use_container_width=True, config=plotly_config())

    with col_cm2:
        total = cm.sum()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        st.markdown(f"""
        <div class="glass-card">
            <h3 style="font-size: 1rem;">📐 Matrix Breakdown</h3>
            <div style="line-height: 2.2; font-size: 0.9rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #B0B0B0;">True Negatives</span>
                    <span style="color: #00D4AA; font-weight: 700;">{tn}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #B0B0B0;">True Positives</span>
                    <span style="color: #00D4AA; font-weight: 700;">{tp}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #B0B0B0;">False Positives</span>
                    <span style="color: #FF6B6B; font-weight: 700;">{fp}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #B0B0B0;">False Negatives</span>
                    <span style="color: #FF6B6B; font-weight: 700;">{fn}</span>
                </div>
                <hr style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #B0B0B0;">Specificity</span>
                    <span style="color: #6C63FF; font-weight: 700;">{tn/(tn+fp):.3f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #B0B0B0;">Sensitivity</span>
                    <span style="color: #6C63FF; font-weight: 700;">{tp/(tp+fn):.3f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: ROC & PR Curves ──────────────────────────────────────────────────
with tab2:
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown('<div class="section-header">📈 ROC Curve (All Models)</div>', unsafe_allow_html=True)

        fig_roc = go.Figure()

        # Diagonal reference
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="rgba(255,255,255,0.2)", width=1),
            name="Random (AUC=0.5)",
            showlegend=True,
        ))

        for i, (name, ev) in enumerate(evaluations.items()):
            line_width = 3 if name == selected_model else 1.5
            opacity = 1.0 if name == selected_model else 0.5
            fig_roc.add_trace(go.Scatter(
                x=ev["fpr"], y=ev["tpr"],
                mode="lines",
                name=f"{name} (AUC={ev['roc_auc']:.3f})",
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=line_width),
                opacity=opacity,
            ))

        fig_roc.update_layout(
            **base_layout(),
            title=dict(text="Receiver Operating Characteristic", font=dict(size=14, color="#FAFAFA")),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=480,
            legend=dict(x=0.4, y=0.05, font=dict(size=10)),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config=plotly_config())

    with col_pr:
        st.markdown('<div class="section-header">📈 Precision-Recall Curve</div>', unsafe_allow_html=True)

        fig_pr = go.Figure()

        for i, (name, ev) in enumerate(evaluations.items()):
            line_width = 3 if name == selected_model else 1.5
            opacity = 1.0 if name == selected_model else 0.5
            fig_pr.add_trace(go.Scatter(
                x=ev["pr_recall"], y=ev["pr_precision"],
                mode="lines",
                name=f"{name} (AP={ev['avg_precision']:.3f})",
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=line_width),
                opacity=opacity,
            ))

        fig_pr.update_layout(
            **base_layout(),
            title=dict(text="Precision-Recall Curve", font=dict(size=14, color="#FAFAFA")),
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=480,
            legend=dict(x=0.0, y=0.05, font=dict(size=10)),
        )
        st.plotly_chart(fig_pr, use_container_width=True, config=plotly_config())

# ─── TAB 3: Feature Importance ───────────────────────────────────────────────
with tab3:
    st.markdown(f'<div class="section-header">🏗️ Feature Importance — {selected_model}</div>', unsafe_allow_html=True)

    fi_df = eval_data.get("feature_importance")

    if fi_df is not None and len(fi_df) > 0 and fi_df["importance"].sum() > 0:
        top_n = min(15, len(fi_df))
        top_features = fi_df.head(top_n).sort_values("importance", ascending=True)

        fig_fi = go.Figure(data=[go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h",
            marker=dict(
                color=top_features["importance"],
                colorscale=[[0, COLORS["primary_dark"]], [0.5, COLORS["primary"]], [1, COLORS["accent"]]],
                cornerradius=6,
            ),
            text=[f"{v:.4f}" for v in top_features["importance"]],
            textposition="outside",
            textfont=dict(color="#FAFAFA", size=11),
        )])
        fig_fi.update_layout(
            **base_layout(margin=dict(l=200)),
            title=dict(text=f"Top {top_n} Features", font=dict(size=16, color="#FAFAFA")),
            xaxis_title="Importance Score",
            height=500,
        )
        st.plotly_chart(fig_fi, use_container_width=True, config=plotly_config())

        # Feature importance table
        with st.expander("📋 Full Feature Importance Table"):
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
    else:
        st.info("⚠️ Feature importance is not available for this model type (e.g., SVM with RBF kernel).")

# ─── TAB 4: Classification Report ────────────────────────────────────────────
with tab4:
    st.markdown(f'<div class="section-header">📋 Classification Report — {selected_model}</div>', unsafe_allow_html=True)

    report = eval_data["classification_report"]

    # Build report dataframe
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            report_data.append({
                "Class": label,
                "Precision": f"{metrics.get('precision', 0):.4f}",
                "Recall": f"{metrics.get('recall', 0):.4f}",
                "F1-Score": f"{metrics.get('f1-score', 0):.4f}",
                "Support": int(metrics.get('support', 0)),
            })

    report_df = pd.DataFrame(report_data)

    # Rename classes
    report_df["Class"] = report_df["Class"].replace({
        "0": "Retained (0)",
        "1": "Churned (1)",
    })

    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Comparison across models
    if comparison_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Model Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Visual comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Visual Model Comparison</div>', unsafe_allow_html=True)

    metrics_to_compare = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_metrics = {}
    for name, ev in evaluations.items():
        model_metrics[name] = [ev["accuracy"], ev["precision"], ev["recall"], ev["f1"], ev["roc_auc"]]

    fig_compare = go.Figure()
    for i, (name, values) in enumerate(model_metrics.items()):
        fig_compare.add_trace(go.Bar(
            name=name,
            x=metrics_to_compare,
            y=values,
            marker=dict(color=CHART_COLORS[i % len(CHART_COLORS)], cornerradius=6),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig_compare.update_layout(
        **base_layout(yaxis=dict(range=[0, 1.1])),
        barmode="group",
        title=dict(text="Model Performance Comparison", font=dict(size=16, color="#FAFAFA")),
        height=450,
        yaxis_title="Score",
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
    )
    st.plotly_chart(fig_compare, use_container_width=True, config=plotly_config())

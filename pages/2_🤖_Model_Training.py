"""
Page 2: Model Training Pipeline
Data preprocessing, train multiple models with GridSearchCV, 5-fold CV, model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    apply_custom_css, load_data, render_page_header, render_metric_card,
    render_insight_box, COLORS, CHART_COLORS, plotly_config
)
from src.data_preprocessing import preprocess_pipeline, scale_features, save_preprocessing_artifacts
from src.model_training import (
    train_all_models, build_comparison_table, get_best_model,
    save_model_artifacts, load_model_artifacts
)

st.set_page_config(page_title="Model Training | Churn Prediction", page_icon="🤖", layout="wide")
apply_custom_css()

render_page_header(
    "Model Training Pipeline",
    "Train, tune, and compare multiple ML models with automated hyperparameter optimization"
)

# ─── Check if models already exist ───────────────────────────────────────────
artifacts = load_model_artifacts()
models_exist = artifacts.get("best_model") is not None

if models_exist:
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #00D4AA;">
        <div style="display: flex; align-items: center; gap: 0.8rem;">
            <span style="font-size: 1.5rem;">✅</span>
            <div>
                <div style="color: #00D4AA; font-weight: 700;">Models Already Trained</div>
                <div style="color: #B0B0B0; font-size: 0.85rem;">
                    Trained models found in the models/ directory. You can retrain or view existing results.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ─── Pipeline Steps Overview ─────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Pipeline Steps</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; text-align: center;">
        <div>
            <div style="font-size: 1.5rem;">🧹</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 0.85rem; margin-top: 0.3rem;">Data Cleaning</div>
            <div style="color: #666; font-size: 0.75rem;">Handle missing values</div>
        </div>
        <div>
            <div style="font-size: 1.5rem;">🔤</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 0.85rem; margin-top: 0.3rem;">Encoding</div>
            <div style="color: #666; font-size: 0.75rem;">Label + One-Hot</div>
        </div>
        <div>
            <div style="font-size: 1.5rem;">⚖️</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 0.85rem; margin-top: 0.3rem;">Scaling</div>
            <div style="color: #666; font-size: 0.75rem;">StandardScaler</div>
        </div>
        <div>
            <div style="font-size: 1.5rem;">🧠</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 0.85rem; margin-top: 0.3rem;">Training</div>
            <div style="color: #666; font-size: 0.75rem;">GridSearchCV + 5-Fold</div>
        </div>
        <div>
            <div style="font-size: 1.5rem;">📊</div>
            <div style="color: #FAFAFA; font-weight: 600; font-size: 0.85rem; margin-top: 0.3rem;">Evaluation</div>
            <div style="color: #666; font-size: 0.75rem;">Compare & Select Best</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Configuration ───────────────────────────────────────────────────────────
col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    test_size = st.slider("Test Set Size", 0.15, 0.35, 0.20, 0.05, key="test_size_slider")
with col_cfg2:
    random_state = st.number_input("Random State", value=42, min_value=0, max_value=9999, key="random_state_input")

# ─── Train Button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

col_btn, col_space = st.columns([1, 3])
with col_btn:
    train_clicked = st.button("🚀 Train All Models", use_container_width=True, key="train_btn")

if train_clicked:
    df = load_data()

    # Step 1: Preprocessing
    st.markdown('<div class="section-header">🧹 Step 1: Data Preprocessing</div>', unsafe_allow_html=True)

    with st.spinner("Cleaning and encoding data..."):
        X, y, label_encoders, feature_names = preprocess_pipeline(df)
        time.sleep(0.3)  # brief pause for UX

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(render_metric_card("📐", f"{X.shape[1]}", "Features"), unsafe_allow_html=True)
    with col_p2:
        st.markdown(render_metric_card("📊", f"{X.shape[0]:,}", "Samples"), unsafe_allow_html=True)
    with col_p3:
        st.markdown(render_metric_card("🎯", f"{y.sum()}", "Positive (Churn)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2: Split
    st.markdown('<div class="section-header">✂️ Step 2: Train-Test Split</div>', unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    X_train, X_test, scaler = scale_features(X_train.copy(), X_test.copy(), fit=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="color: #6C63FF; font-size: 1.8rem; font-weight: 800;">{len(X_train):,}</div>
            <div style="color: #B0B0B0; font-size: 0.85rem;">Training Samples ({100-test_size*100:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="color: #00D4AA; font-size: 1.8rem; font-weight: 800;">{len(X_test):,}</div>
            <div style="color: #B0B0B0; font-size: 0.85rem;">Test Samples ({test_size*100:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 3: Training
    st.markdown('<div class="section-header">🧠 Step 3: Model Training with GridSearchCV</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(progress, text):
        progress_bar.progress(progress)
        status_text.markdown(f"""
        <div style="color: #B0B0B0; font-size: 0.9rem;">
            <span class="pulse">⏳</span> {text}
        </div>
        """, unsafe_allow_html=True)

    results = train_all_models(X_train, y_train, progress_callback=progress_callback)

    progress_bar.progress(1.0)
    status_text.markdown("""
    <div style="color: #00D4AA; font-weight: 600;">
        ✅ All models trained successfully!
    </div>
    """, unsafe_allow_html=True)

    # Step 4: Evaluation & Comparison
    st.markdown('<div class="section-header">📊 Step 4: Model Comparison</div>', unsafe_allow_html=True)

    comparison_df, evaluations = build_comparison_table(results, X_test, y_test)
    best_result = get_best_model(results, X_test, y_test)

    # Highlight best model
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #00D4AA;">
        <div style="display: flex; align-items: center; gap: 0.8rem;">
            <span style="font-size: 2rem;">🏆</span>
            <div>
                <div style="color: #00D4AA; font-weight: 700; font-size: 1.1rem;">
                    Best Model: {best_result['name']}
                </div>
                <div style="color: #B0B0B0; font-size: 0.85rem;">
                    CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f} | 
                    Training Time: {best_result['training_time']:.2f}s
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # Best params per model
    with st.expander("🔧 Best Hyperparameters per Model", expanded=False):
        for result in results:
            st.markdown(f"**{result['name']}**: `{result['best_params']}`")

    # Step 5: Save
    st.markdown('<div class="section-header">💾 Step 5: Save Artifacts</div>', unsafe_allow_html=True)

    with st.spinner("Saving models and artifacts..."):
        save_model_artifacts(
            results, best_result, scaler, label_encoders, feature_names,
            comparison_df, evaluations
        )
        time.sleep(0.3)

    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #6C63FF;">
        <div style="color: #6C63FF; font-weight: 700; margin-bottom: 0.5rem;">💾 Artifacts Saved</div>
        <div style="color: #B0B0B0; font-size: 0.85rem; line-height: 1.8;">
            ✅ <code>models/best_model.pkl</code> — Best performing model<br>
            ✅ <code>models/all_models.pkl</code> — All trained models<br>
            ✅ <code>models/scaler.pkl</code> — StandardScaler<br>
            ✅ <code>models/label_encoders.pkl</code> — Label encoders<br>
            ✅ <code>models/feature_names.pkl</code> — Feature names<br>
            ✅ <code>models/model_comparison.pkl</code> — Comparison table<br>
            ✅ <code>models/model_evaluations.pkl</code> — Evaluation metrics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Training pipeline completed successfully! Navigate to **Model Performance** to explore detailed results.")

# ─── Show existing results if models exist ────────────────────────────────────
elif models_exist and not train_clicked:
    st.markdown('<div class="section-header">📊 Previous Training Results</div>', unsafe_allow_html=True)

    comparison_df = artifacts.get("comparison_df")
    best_name = artifacts.get("best_model_name")

    if comparison_df is not None:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #00D4AA;">
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span style="font-size: 2rem;">🏆</span>
                <div>
                    <div style="color: #00D4AA; font-weight: 700; font-size: 1.1rem;">
                        Best Model: {best_name}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.info("💡 Click **Train All Models** above to retrain with new settings.")

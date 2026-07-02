"""
Page 1: Exploratory Data Analysis
Distribution plots, correlation heatmap, churn rate by feature, business insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    apply_custom_css, load_data, render_page_header, render_metric_card,
    render_insight_box, COLORS, CHART_COLORS, base_layout, plotly_config
)

st.set_page_config(page_title="EDA | Churn Prediction", page_icon="📊", layout="wide")
apply_custom_css()

render_page_header("Exploratory Data Analysis", "Deep dive into customer data patterns, distributions, and churn drivers")

df = load_data()

# ─── Key Metrics ──────────────────────────────────────────────────────────────
churn_counts = df["Churn"].value_counts()
churn_rate = churn_counts.get("Yes", 0) / len(df) * 100
avg_tenure_churn = df[df["Churn"] == "Yes"]["tenure"].mean()
avg_tenure_stay = df[df["Churn"] == "No"]["tenure"].mean()
avg_monthly_churn = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(render_metric_card("📉", f"{churn_rate:.1f}%", "Overall Churn"), unsafe_allow_html=True)
with col2:
    st.markdown(render_metric_card("⏳", f"{avg_tenure_churn:.0f} mo", "Avg Tenure (Churned)"), unsafe_allow_html=True)
with col3:
    st.markdown(render_metric_card("⏳", f"{avg_tenure_stay:.0f} mo", "Avg Tenure (Retained)"), unsafe_allow_html=True)
with col4:
    st.markdown(render_metric_card("💸", f"${avg_monthly_churn:.0f}", "Avg Monthly (Churned)"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab layout
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "📉 Churn Analysis", "💡 Insights"])

# ─── TAB 1: Distributions ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📊 Target Variable Distribution</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Donut chart for churn
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Retained", "Churned"],
            values=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
            hole=0.65,
            marker=dict(colors=[COLORS["accent"], COLORS["accent2"]]),
            textinfo="percent+value",
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])
        fig_donut.update_layout(
            **base_layout(),
            title=dict(text="Churn Distribution", font=dict(size=16, color="#FAFAFA")),
            height=400,
            showlegend=True,
            legend=dict(font=dict(color="#B0B0B0")),
            annotations=[dict(
                text=f"{churn_rate:.1f}%<br><span style='font-size:12px;color:#B0B0B0'>Churn</span>",
                x=0.5, y=0.5, font_size=28, font_color="#FF6B6B",
                showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config=plotly_config())

    with col_b:
        # Bar chart for churn counts
        fig_bar = go.Figure(data=[
            go.Bar(
                x=["Retained", "Churned"],
                y=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
                marker=dict(
                    color=[COLORS["accent"], COLORS["accent2"]],
                    line=dict(width=0),
                    cornerradius=8,
                ),
                text=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
                textposition="outside",
                textfont=dict(color="#FAFAFA", size=14, family="Inter"),
            )
        ])
        fig_bar.update_layout(
            **base_layout(),
            title=dict(text="Customer Count by Status", font=dict(size=16, color="#FAFAFA")),
            height=400,
            yaxis_title="Count",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config=plotly_config())

    # Numeric distributions
    st.markdown('<div class="section-header">📈 Numeric Feature Distributions</div>', unsafe_allow_html=True)

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    df_plot = df.copy()
    df_plot["TotalCharges"] = pd.to_numeric(df_plot["TotalCharges"], errors="coerce")

    cols = st.columns(3)
    for i, feat in enumerate(numeric_features):
        with cols[i]:
            fig = px.histogram(
                df_plot, x=feat, color="Churn",
                nbins=40,
                color_discrete_map={"No": COLORS["accent"], "Yes": COLORS["accent2"]},
                barmode="overlay",
                opacity=0.75,
            )
            fig.update_layout(
                **base_layout(),
                title=dict(text=f"{feat} Distribution", font=dict(size=14, color="#FAFAFA")),
                height=350,
                xaxis_title=feat,
                yaxis_title="Count",
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    # Box plots
    st.markdown('<div class="section-header">📦 Box Plots by Churn Status</div>', unsafe_allow_html=True)
    cols2 = st.columns(3)
    for i, feat in enumerate(numeric_features):
        with cols2[i]:
            fig = px.box(
                df_plot, x="Churn", y=feat,
                color="Churn",
                color_discrete_map={"No": COLORS["accent"], "Yes": COLORS["accent2"]},
            )
            fig.update_layout(
                **base_layout(),
                title=dict(text=f"{feat} by Churn", font=dict(size=14, color="#FAFAFA")),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

# ─── TAB 2: Correlations ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)

    # Prepare numeric data
    df_corr = df.copy()
    df_corr["TotalCharges"] = pd.to_numeric(df_corr["TotalCharges"], errors="coerce")
    df_corr["Churn_Numeric"] = (df_corr["Churn"] == "Yes").astype(int)

    # Encode some categoricals for correlation
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df_corr.columns:
            df_corr[col] = df_corr[col].map(binary_map).fillna(df_corr[col])

    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    # Remove customerID-like columns
    numeric_cols = [c for c in numeric_cols if c != "customerID"]

    corr_matrix = df_corr[numeric_cols].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, "#FF6B6B"],
            [0.5, "#1A1D23"],
            [1, "#6C63FF"]
        ],
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#FAFAFA"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="Corr",
            title_font=dict(color="#B0B0B0"),
            tickfont=dict(color="#B0B0B0"),
        )
    ))
    fig_heatmap.update_layout(
        **base_layout(
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
        ),
        title=dict(text="Feature Correlation Matrix", font=dict(size=16, color="#FAFAFA")),
        height=600,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, config=plotly_config())

    # Top correlations with Churn
    st.markdown('<div class="section-header">🎯 Top Correlations with Churn</div>', unsafe_allow_html=True)

    churn_corr = corr_matrix["Churn_Numeric"].drop("Churn_Numeric").sort_values(key=abs, ascending=False)

    fig_corr_bar = go.Figure(data=[go.Bar(
        x=churn_corr.values,
        y=churn_corr.index,
        orientation="h",
        marker=dict(
            color=[COLORS["accent2"] if v > 0 else COLORS["accent"] for v in churn_corr.values],
            cornerradius=4,
        ),
        text=[f"{v:.3f}" for v in churn_corr.values],
        textposition="outside",
        textfont=dict(color="#FAFAFA", size=11),
    )])
    fig_corr_bar.update_layout(
        **base_layout(yaxis=dict(autorange="reversed")),
        title=dict(text="Correlation with Churn", font=dict(size=16, color="#FAFAFA")),
        height=450,
        xaxis_title="Correlation Coefficient",
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True, config=plotly_config())

# ─── TAB 3: Churn Rate by Feature ────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📉 Churn Rate by Categorical Feature</div>', unsafe_allow_html=True)

    cat_features = [
        "Contract", "InternetService", "PaymentMethod", "gender",
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"
    ]

    # Key features (2x2 grid)
    key_features = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
    cols_key = st.columns(2)

    for idx, feat in enumerate(key_features):
        with cols_key[idx % 2]:
            churn_by_feat = df.groupby(feat)["Churn"].value_counts(normalize=True).unstack()
            if "Yes" in churn_by_feat.columns:
                churn_rates = (churn_by_feat["Yes"] * 100).sort_values(ascending=True)
            else:
                continue

            fig = go.Figure(data=[go.Bar(
                x=churn_rates.values,
                y=churn_rates.index,
                orientation="h",
                marker=dict(
                    color=churn_rates.values,
                    colorscale=[[0, COLORS["accent"]], [1, COLORS["accent2"]]],
                    cornerradius=6,
                ),
                text=[f"{v:.1f}%" for v in churn_rates.values],
                textposition="outside",
                textfont=dict(color="#FAFAFA", size=12),
            )])
            fig.update_layout(
                **base_layout(margin=dict(l=120)),
                title=dict(text=f"Churn Rate by {feat}", font=dict(size=14, color="#FAFAFA")),
                height=300,
                xaxis_title="Churn Rate (%)",
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    # Remaining features with expander
    with st.expander("🔍 View All Categorical Features", expanded=False):
        remaining = [f for f in cat_features if f not in key_features]
        cols_rem = st.columns(3)
        for idx, feat in enumerate(remaining):
            with cols_rem[idx % 3]:
                churn_by_feat = df.groupby(feat)["Churn"].value_counts(normalize=True).unstack()
                if "Yes" in churn_by_feat.columns:
                    churn_rates = (churn_by_feat["Yes"] * 100).sort_values(ascending=True)
                else:
                    continue

                fig = go.Figure(data=[go.Bar(
                    x=churn_rates.values,
                    y=churn_rates.index,
                    orientation="h",
                    marker=dict(
                        color=churn_rates.values,
                        colorscale=[[0, COLORS["accent"]], [1, COLORS["accent2"]]],
                        cornerradius=4,
                    ),
                    text=[f"{v:.1f}%" for v in churn_rates.values],
                    textposition="outside",
                    textfont=dict(color="#FAFAFA", size=10),
                )])
                fig.update_layout(
                    **base_layout(margin=dict(l=120, t=40, b=30)),
                    title=dict(text=f"{feat}", font=dict(size=12, color="#FAFAFA")),
                    height=250,
                )
                st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    # SeniorCitizen analysis
    st.markdown('<div class="section-header">👴 Senior Citizen Analysis</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        senior_churn = df.groupby("SeniorCitizen")["Churn"].value_counts(normalize=True).unstack()
        labels = ["Non-Senior", "Senior"]
        fig_senior = go.Figure(data=[
            go.Bar(name="Retained", x=labels, y=[senior_churn.loc[0, "No"]*100, senior_churn.loc[1, "No"]*100],
                   marker=dict(color=COLORS["accent"], cornerradius=6)),
            go.Bar(name="Churned", x=labels, y=[senior_churn.loc[0, "Yes"]*100, senior_churn.loc[1, "Yes"]*100],
                   marker=dict(color=COLORS["accent2"], cornerradius=6)),
        ])
        fig_senior.update_layout(
            **base_layout(),
            barmode="stack",
            title=dict(text="Churn Rate: Senior vs Non-Senior", font=dict(size=14, color="#FAFAFA")),
            height=350,
            yaxis_title="Percentage (%)",
        )
        st.plotly_chart(fig_senior, use_container_width=True, config=plotly_config())

    with col_s2:
        # Tenure distribution by churn
        fig_tenure = px.violin(
            df, x="Churn", y="tenure", color="Churn",
            color_discrete_map={"No": COLORS["accent"], "Yes": COLORS["accent2"]},
            box=True,
        )
        fig_tenure.update_layout(
            **base_layout(),
            title=dict(text="Tenure Distribution by Churn", font=dict(size=14, color="#FAFAFA")),
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_tenure, use_container_width=True, config=plotly_config())

# ─── TAB 4: Business Insights ────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">💡 Key Business Insights</div>', unsafe_allow_html=True)

    # Calculate insights
    mtm_churn = df[df["Contract"] == "Month-to-month"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    fiber_churn = df[df["InternetService"] == "Fiber optic"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    echeck_churn = df[df["PaymentMethod"] == "Electronic check"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    no_tech_churn = df[df["TechSupport"] == "No"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    no_security_churn = df[df["OnlineSecurity"] == "No"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100

    df_temp = df.copy()
    df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
    revenue_at_risk = df_temp[df_temp["Churn"] == "Yes"]["MonthlyCharges"].sum()

    render_insight_box(
        "Contract Type is the Strongest Churn Predictor",
        f"Month-to-month customers churn at <strong>{mtm_churn:.1f}%</strong> — significantly higher than "
        f"one-year or two-year contract holders. Incentivizing longer contracts could dramatically reduce churn."
    )

    render_insight_box(
        "Fiber Optic Users Show Higher Churn",
        f"Fiber optic internet subscribers churn at <strong>{fiber_churn:.1f}%</strong>. "
        f"This may indicate service quality issues, pricing concerns, or competitive alternatives in fiber markets."
    )

    render_insight_box(
        "Payment Method Matters",
        f"Customers using electronic check payments churn at <strong>{echeck_churn:.1f}%</strong>. "
        f"Encouraging automatic payment methods (bank transfer, credit card) could improve retention."
    )

    render_insight_box(
        "Tech Support & Security are Retention Anchors",
        f"Customers without tech support churn at <strong>{no_tech_churn:.1f}%</strong>, and those without "
        f"online security churn at <strong>{no_security_churn:.1f}%</strong>. Bundling these services can "
        f"increase customer stickiness."
    )

    render_insight_box(
        "Revenue at Risk",
        f"The company risks losing <strong>${revenue_at_risk:,.0f}</strong> in monthly recurring revenue "
        f"from customers who have already churned. Proactive retention programs targeting high-risk segments "
        f"can significantly impact the bottom line."
    )

    render_insight_box(
        "Early Tenure is Critical",
        f"Churned customers have an average tenure of <strong>{avg_tenure_churn:.0f} months</strong> vs. "
        f"<strong>{avg_tenure_stay:.0f} months</strong> for retained customers. "
        f"The first 12 months are the most critical window for churn prevention."
    )

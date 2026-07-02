"""
Page 6: Business Insights
Customer segmentation, risk scoring, retention recommendations, revenue impact analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    apply_custom_css, load_data, render_page_header, render_metric_card,
    render_insight_box, render_risk_badge, get_risk_level,
    COLORS, CHART_COLORS, base_layout, plotly_config
)
from src.model_training import load_model_artifacts
from src.data_preprocessing import preprocess_batch

st.set_page_config(page_title="Insights | Churn Prediction", page_icon="💡", layout="wide")
apply_custom_css()

render_page_header(
    "Business Insights",
    "Customer segmentation, risk analytics, and data-driven retention strategies"
)

df = load_data()

# ─── Try to load model for risk scoring ──────────────────────────────────────
artifacts = load_model_artifacts()
has_model = artifacts.get("best_model") is not None

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Customer Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👥 Customer Segmentation by Tenure</div>', unsafe_allow_html=True)

df_seg = df.copy()
df_seg["TotalCharges"] = pd.to_numeric(df_seg["TotalCharges"], errors="coerce")

# Create tenure segments
def tenure_segment(t):
    if t <= 12:
        return "New (0-12 mo)"
    elif t <= 36:
        return "Growing (13-36 mo)"
    elif t <= 60:
        return "Mature (37-60 mo)"
    else:
        return "Loyal (61+ mo)"

df_seg["Segment"] = df_seg["tenure"].apply(tenure_segment)

segment_order = ["New (0-12 mo)", "Growing (13-36 mo)", "Mature (37-60 mo)", "Loyal (61+ mo)"]
segment_stats = []

for seg in segment_order:
    seg_df = df_seg[df_seg["Segment"] == seg]
    churn_rate = (seg_df["Churn"] == "Yes").mean() * 100
    count = len(seg_df)
    avg_monthly = seg_df["MonthlyCharges"].mean()
    avg_total = seg_df["TotalCharges"].mean()
    segment_stats.append({
        "Segment": seg,
        "Customers": count,
        "Churn Rate": f"{churn_rate:.1f}%",
        "Avg Monthly ($)": f"{avg_monthly:.2f}",
        "Avg Total ($)": f"{avg_total:.2f}",
        "churn_rate_val": churn_rate,
    })

seg_df_display = pd.DataFrame(segment_stats)

# Metric cards for segments
cols_seg = st.columns(4)
seg_icons = ["🌱", "📈", "🏢", "⭐"]
seg_colors = ["#FF6B6B", "#FFD93D", "#6C63FF", "#00D4AA"]

for i, seg in enumerate(segment_stats):
    with cols_seg[i]:
        st.markdown(f"""
        <div class="metric-card fade-in" style="border-top: 3px solid {seg_colors[i]};">
            <div class="metric-icon">{seg_icons[i]}</div>
            <div style="color: #FAFAFA; font-weight: 700; font-size: 0.9rem; margin: 0.3rem 0;">
                {seg['Segment']}
            </div>
            <div class="metric-value" style="font-size: 1.5rem;">{seg['Customers']:,}</div>
            <div style="color: {'#FF6B6B' if seg['churn_rate_val'] > 30 else '#00D4AA'}; 
                 font-weight: 600; font-size: 0.9rem; margin-top: 0.3rem;">
                {seg['Churn Rate']} churn
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Segmentation charts
col_seg1, col_seg2 = st.columns(2)

with col_seg1:
    fig_seg_bar = go.Figure(data=[
        go.Bar(
            x=segment_order,
            y=[s["Customers"] for s in segment_stats],
            marker=dict(color=seg_colors, cornerradius=8),
            text=[f"{s['Customers']:,}" for s in segment_stats],
            textposition="outside",
            textfont=dict(color="#FAFAFA", size=13),
        )
    ])
    fig_seg_bar.update_layout(
        **base_layout(),
        title=dict(text="Customers by Segment", font=dict(size=14, color="#FAFAFA")),
        height=380,
        yaxis_title="Count",
    )
    st.plotly_chart(fig_seg_bar, use_container_width=True, config=plotly_config())

with col_seg2:
    fig_seg_churn = go.Figure(data=[
        go.Bar(
            x=segment_order,
            y=[s["churn_rate_val"] for s in segment_stats],
            marker=dict(
                color=[s["churn_rate_val"] for s in segment_stats],
                colorscale=[[0, COLORS["accent"]], [1, COLORS["accent2"]]],
                cornerradius=8,
            ),
            text=[s["Churn Rate"] for s in segment_stats],
            textposition="outside",
            textfont=dict(color="#FAFAFA", size=13),
        )
    ])
    fig_seg_churn.update_layout(
        **base_layout(),
        title=dict(text="Churn Rate by Segment", font=dict(size=14, color="#FAFAFA")),
        height=380,
        yaxis_title="Churn Rate (%)",
    )
    st.plotly_chart(fig_seg_churn, use_container_width=True, config=plotly_config())

# Segment table
st.dataframe(
    seg_df_display[["Segment", "Customers", "Churn Rate", "Avg Monthly ($)", "Avg Total ($)"]],
    use_container_width=True,
    hide_index=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Risk Scoring (if model available)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🎯 Risk Scoring Distribution</div>', unsafe_allow_html=True)

if has_model:
    model = artifacts["best_model"]
    scaler = artifacts["scaler"]
    label_encoders_loaded = artifacts["label_encoders"]
    feat_names = artifacts["feature_names"]

    with st.spinner("Scoring all customers..."):
        try:
            processed = preprocess_batch(df, label_encoders_loaded, scaler, feat_names)
            probabilities = model.predict_proba(processed)[:, 1]
            df_seg["Churn_Probability"] = probabilities * 100
            df_seg["Risk_Level"] = [get_risk_level(p) for p in probabilities]
        except Exception as e:
            st.warning(f"Could not score all customers: {e}")
            has_model = False

if has_model and "Risk_Level" in df_seg.columns:
    # Risk tier metrics
    risk_order = ["Low", "Medium", "High", "Critical"]
    risk_colors_map = {"Low": COLORS["accent"], "Medium": COLORS["accent3"],
                       "High": COLORS["accent2"], "Critical": "#FF3838"}

    risk_counts = df_seg["Risk_Level"].value_counts()
    cols_risk = st.columns(4)
    for i, risk in enumerate(risk_order):
        with cols_risk[i]:
            count = risk_counts.get(risk, 0)
            pct = count / len(df_seg) * 100
            dot_color = risk_colors_map[risk]
            st.markdown(f"""
            <div class="metric-card fade-in" style="min-height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="width: 28px; height: 28px; border-radius: 50%; background: {dot_color};
                     box-shadow: 0 0 12px {dot_color}80; margin-bottom: 0.5rem;"></div>
                <div class="metric-value" style="font-size: 1.5rem;">{count:,}</div>
                <div class="metric-label">{risk} Risk ({pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_risk1, col_risk2 = st.columns(2)

    with col_risk1:
        fig_risk_dist = px.histogram(
            df_seg, x="Churn_Probability",
            nbins=50,
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_risk_dist.add_vrect(x0=50, x1=100, fillcolor="rgba(255,107,107,0.08)",
                                line=dict(width=0), annotation_text="High Risk Zone",
                                annotation_position="top right",
                                annotation_font=dict(color="#FF6B6B", size=11))
        fig_risk_dist.update_layout(
            **base_layout(),
            title=dict(text="Churn Probability Distribution", font=dict(size=14, color="#FAFAFA")),
            xaxis_title="Churn Probability (%)",
            yaxis_title="Customer Count",
            height=400,
        )
        st.plotly_chart(fig_risk_dist, use_container_width=True, config=plotly_config())

    with col_risk2:
        fig_risk_pie = go.Figure(data=[go.Pie(
            labels=[r for r in risk_order if r in risk_counts.index],
            values=[risk_counts.get(r, 0) for r in risk_order if r in risk_counts.index],
            hole=0.6,
            marker=dict(colors=[risk_colors_map[r] for r in risk_order if r in risk_counts.index]),
            textinfo="percent+value",
            textfont=dict(size=12, color="white"),
        )])
        fig_risk_pie.update_layout(
            **base_layout(),
            title=dict(text="Risk Tier Distribution", font=dict(size=14, color="#FAFAFA")),
            height=400,
        )
        st.plotly_chart(fig_risk_pie, use_container_width=True, config=plotly_config())

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Revenue Impact
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💰 Revenue Impact Analysis</div>', unsafe_allow_html=True)

    high_risk_df = df_seg[df_seg["Risk_Level"].isin(["High", "Critical"])]
    monthly_at_risk = high_risk_df["MonthlyCharges"].sum()
    annual_at_risk = monthly_at_risk * 12
    avg_clv = df_seg["TotalCharges"].mean()

    col_rev1, col_rev2, col_rev3 = st.columns(3)
    with col_rev1:
        st.markdown(render_metric_card("💸", f"${monthly_at_risk:,.0f}", "Monthly Revenue at Risk"), unsafe_allow_html=True)
    with col_rev2:
        st.markdown(render_metric_card("📅", f"${annual_at_risk:,.0f}", "Annual Revenue at Risk"), unsafe_allow_html=True)
    with col_rev3:
        st.markdown(render_metric_card("👤", f"${avg_clv:,.0f}", "Avg Customer Lifetime Value"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue by risk tier
    revenue_by_risk = df_seg.groupby("Risk_Level")["MonthlyCharges"].agg(["sum", "mean", "count"]).reset_index()
    revenue_by_risk.columns = ["Risk_Level", "Total_Monthly", "Avg_Monthly", "Count"]
    revenue_by_risk["Risk_Level"] = pd.Categorical(revenue_by_risk["Risk_Level"], categories=risk_order, ordered=True)
    revenue_by_risk = revenue_by_risk.sort_values("Risk_Level")

    fig_rev = go.Figure(data=[go.Bar(
        x=revenue_by_risk["Risk_Level"],
        y=revenue_by_risk["Total_Monthly"],
        marker=dict(
            color=[risk_colors_map.get(r, COLORS["primary"]) for r in revenue_by_risk["Risk_Level"]],
            cornerradius=8,
        ),
        text=[f"${v:,.0f}" for v in revenue_by_risk["Total_Monthly"]],
        textposition="outside",
        textfont=dict(color="#FAFAFA", size=12),
    )])
    fig_rev.update_layout(
        **base_layout(),
        title=dict(text="Monthly Revenue by Risk Tier", font=dict(size=14, color="#FAFAFA")),
        height=380,
        yaxis_title="Monthly Revenue ($)",
    )
    st.plotly_chart(fig_rev, use_container_width=True, config=plotly_config())

else:
    st.info("💡 Train the model first (Model Training page) to enable risk scoring and revenue impact analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Churn Drivers Summary
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🔍 Churn Drivers Summary</div>', unsafe_allow_html=True)

# Calculate churn rates for key drivers
drivers = {}
for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]:
    for val in df[col].unique():
        sub = df[df[col] == val]
        cr = (sub["Churn"] == "Yes").mean() * 100
        drivers[f"{col}: {val}"] = cr

# Sort and get top churn drivers
drivers_sorted = sorted(drivers.items(), key=lambda x: x[1], reverse=True)[:10]

fig_drivers = go.Figure(data=[go.Bar(
    x=[v for _, v in drivers_sorted],
    y=[k for k, _ in drivers_sorted],
    orientation="h",
    marker=dict(
        color=[v for _, v in drivers_sorted],
        colorscale=[[0, COLORS["accent"]], [1, COLORS["accent2"]]],
        cornerradius=6,
    ),
    text=[f"{v:.1f}%" for _, v in drivers_sorted],
    textposition="outside",
    textfont=dict(color="#FAFAFA", size=11),
)])
fig_drivers.update_layout(
    **base_layout(margin=dict(l=250), yaxis=dict(autorange="reversed")),
    title=dict(text="Top 10 Churn Drivers", font=dict(size=16, color="#FAFAFA")),
    xaxis_title="Churn Rate (%)",
    height=450,
)
st.plotly_chart(fig_drivers, use_container_width=True, config=plotly_config())

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Retention Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🛡️ Retention Recommendations</div>', unsafe_allow_html=True)

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.markdown("""
    <div class="rec-card" style="border-left: 4px solid #FF6B6B;">
        <div class="rec-title">🔴 High & Critical Risk Customers</div>
        <div class="rec-desc" style="line-height: 1.8;">
            <strong>Immediate Actions:</strong><br>
            • Assign dedicated account managers for personal outreach<br>
            • Offer 20-30% discount on contract upgrade to annual plan<br>
            • Provide complimentary premium services (Tech Support, Security) for 3 months<br>
            • Implement proactive service quality monitoring<br>
            • Send personalized "We Value You" campaigns
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rec-card" style="border-left: 4px solid #00D4AA;">
        <div class="rec-title">🟢 Low Risk Customers</div>
        <div class="rec-desc" style="line-height: 1.8;">
            <strong>Growth Actions:</strong><br>
            • Introduce referral program with incentives<br>
            • Upsell premium services (Streaming, Security bundles)<br>
            • Offer family/household plans for cross-selling<br>
            • Encourage long-term contract renewals with loyalty rewards<br>
            • Collect testimonials and reviews for marketing
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_rec2:
    st.markdown("""
    <div class="rec-card" style="border-left: 4px solid #FFD93D;">
        <div class="rec-title">🟡 Medium Risk Customers</div>
        <div class="rec-desc" style="line-height: 1.8;">
            <strong>Proactive Engagement:</strong><br>
            • Schedule quarterly satisfaction check-in calls<br>
            • Offer bundled service discounts (Internet + Security + Support)<br>
            • Provide early access to new features and services<br>
            • Implement automated retention triggers (usage dip alerts)<br>
            • Create loyalty milestone rewards program
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rec-card" style="border-left: 4px solid #6C63FF;">
        <div class="rec-title">📊 Strategic Recommendations</div>
        <div class="rec-desc" style="line-height: 1.8;">
            <strong>Company-Wide Initiatives:</strong><br>
            • Review fiber optic pricing competitiveness<br>
            • Improve first-year onboarding experience<br>
            • Transition customers from electronic check to auto-pay<br>
            • Invest in tech support quality and availability<br>
            • Develop predictive churn alerts for account teams
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Feature importance from model (if available)
# ═══════════════════════════════════════════════════════════════════════════════
if has_model and artifacts.get("evaluations"):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏗️ Model Feature Importance</div>', unsafe_allow_html=True)

    best_name = artifacts.get("best_model_name", "")
    if best_name in artifacts["evaluations"]:
        fi_df = artifacts["evaluations"][best_name].get("feature_importance")
        if fi_df is not None and fi_df["importance"].sum() > 0:
            top_10 = fi_df.head(10).sort_values("importance", ascending=True)

            fig_fi = go.Figure(data=[go.Bar(
                x=top_10["importance"],
                y=top_10["feature"],
                orientation="h",
                marker=dict(
                    color=top_10["importance"],
                    colorscale=[[0, COLORS["primary_dark"]], [1, COLORS["accent"]]],
                    cornerradius=6,
                ),
                text=[f"{v:.4f}" for v in top_10["importance"]],
                textposition="outside",
                textfont=dict(color="#FAFAFA", size=11),
            )])
            fig_fi.update_layout(
                **base_layout(margin=dict(l=200)),
                title=dict(text=f"Top 10 Features — {best_name}", font=dict(size=14, color="#FAFAFA")),
                xaxis_title="Importance",
                height=420,
            )
            st.plotly_chart(fig_fi, use_container_width=True, config=plotly_config())

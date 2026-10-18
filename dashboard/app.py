"""
Coach Dashboard: Athlete Injury-Risk Monitoring Platform
========================================================
Implements Flowchart 4:
- Athlete Risk Overview (High / Moderate / Low, Calibrated Probability %)
- Contributing Factors & Deep Neural Attention attribution
- Historical Workload and Wellness Trends (interactive charts)
- Configured Condition Alerts (ACWR spikes, extreme soreness, sleep deficits)
"""

import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Athlete Injury-Risk Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        margin-bottom: 10px;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-moderate {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #198754;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Athlete Injury-Risk Monitoring Platform")
st.caption("Multimodal Decision-Support System · Workload Analytics & Calibrated Deep Sequential Modeling")

# Verify API connection
try:
    athletes_res = requests.get(f"{API_URL}/athletes", timeout=5)
    athletes = athletes_res.json() if athletes_res.status_code == 200 else []
except requests.RequestException:
    st.error(f"Cannot reach Backend API at `{API_URL}`. Make sure it's running via `uvicorn app.main:app --reload`.")
    st.stop()

if not athletes:
    st.warning("No athletes found in the database. Use the API or data ingestion to register athletes.")
    st.stop()

# Sidebar: Athlete Selector & Platform Info
with st.sidebar:
    st.header("Player Selection")
    athlete_map = {f"{a['name']} ({a['external_id']})": a for a in athletes}
    selected_label = st.selectbox("Select Athlete", list(athlete_map.keys()))
    selected_athlete = athlete_map[selected_label]

    st.divider()
    st.markdown("### Model Status")
    st.markdown("- **Engine**: GRU + Temporal Attention & Workload Rules")
    st.markdown("- **Lookback Sequence**: 14 Days")
    st.markdown("- **Horizon**: Next 7 Days")
    st.caption("Training notebook: `notebooks/Athlete_Injury_Deep_Learning_Colab.ipynb`")

# Fetch risk assessment for selected athlete
try:
    assessment_res = requests.get(f"{API_URL}/assessments/{selected_athlete['id']}", timeout=5)
    assessment = assessment_res.json()
except requests.RequestException as e:
    st.error(f"Failed to fetch assessment: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# SECTION 1: ATHLETE RISK OVERVIEW
# ---------------------------------------------------------------------------
risk_level = assessment.get("risk_level", "low").upper()
risk_score = assessment.get("risk_score", 0.0)
model_prob = assessment.get("model_probability")

color_map = {"LOW": "#198754", "MODERATE": "#ffc107", "HIGH": "#dc3545"}
badge_color = color_map.get(risk_level, "#6c757d")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div style="background-color: {badge_color}15; border: 2px solid {badge_color}; border-radius: 10px; padding: 15px; text-align: center;">
            <span style="font-size: 0.9em; text-transform: uppercase; color: #555;">Current Risk Level</span>
            <h2 style="color: {badge_color}; margin: 5px 0;">{risk_level}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric("Synthesized Risk Score", f"{risk_score:.0f} / 100")

with col3:
    if model_prob is not None:
        st.metric("Calibrated 7-Day Risk", f"{model_prob * 100:.1f}%")
    else:
        st.metric("Deep Model Probability", "Pending Colab Run")

with col4:
    st.metric("Position / Sport", f"{selected_athlete.get('position') or 'Player'} · {selected_athlete.get('sport', 'Football').capitalize()}")

if assessment.get("data_quality_note"):
    st.info(f"ℹ️ {assessment['data_quality_note']}")

st.divider()

# ---------------------------------------------------------------------------
# SECTION 2: CONTRIBUTING FACTORS & ALERTS
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("⚠️ Contributing Factors & Triggers")
    factors = assessment.get("factors", [])
    if factors:
        for f in factors:
            name = f.get("name", "")
            val = f.get("value", 0)
            msg = f.get("message", "")
            st.markdown(
                f"""
                <div style="background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin-bottom: 8px; border-radius: 4px;">
                    <strong>{name.replace('_', ' ').title()}:</strong> {msg}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("No elevated risk factors detected in current observations.")

with col_right:
    st.subheader("🔔 Active Condition Alerts")
    active_alerts = []
    for f in factors:
        fname = f.get("name", "")
        if "acute_chronic" in fname:
            active_alerts.append(("Workload Spike", "Acute:Chronic ratio above 1.3 - monitor training volume."))
        elif "soreness" in fname:
            active_alerts.append(("High Muscle Soreness", "Player reports severe localized muscle tightness."))
        elif "fatigue" in fname:
            active_alerts.append(("Systemic Fatigue", "Fatigue elevated above safe recovery threshold."))
        elif "sleep" in fname:
            active_alerts.append(("Sleep Restriction", "Recent average sleep below 7 hours."))
        elif "deep_sequence" in fname:
            active_alerts.append(("Recurrent Neural Alert", "Deep sequence pattern flags multi-day accumulation."))

    if active_alerts:
        for title, desc in active_alerts:
            st.error(f"🚨 **{title}**: {desc}")
    else:
        st.info("All configured physiological conditions are within normal limits.")

st.divider()

# ---------------------------------------------------------------------------
# SECTION 3: HISTORICAL WORKLOAD & WELLNESS TRENDS
# ---------------------------------------------------------------------------
st.subheader("📈 Historical Workload & Wellness Trends (Last 28 Days)")

try:
    obs_res = requests.get(f"{API_URL}/athletes/{selected_athlete['id']}/observations", timeout=5)
    records = obs_res.json() if obs_res.status_code == 200 else []
except requests.RequestException:
    records = []

if records:
    df_obs = pd.DataFrame(records)
    df_obs["observed_on"] = pd.to_datetime(df_obs["observed_on"])
    df_obs = df_obs.sort_values("observed_on").set_index("observed_on")

    tab1, tab2, tab3 = st.tabs(["Training Load", "Wellness & Recovery", "Raw Data Table"])

    with tab1:
        load_cols = [c for c in ["daily_load", "duration_minutes", "total_distance_m"] if c in df_obs.columns]
        if load_cols:
            st.line_chart(df_obs[load_cols], height=300)
        else:
            st.info("No load metrics recorded yet.")

    with tab2:
        wellness_cols = [c for c in ["fatigue", "soreness", "sleep_hours"] if c in df_obs.columns]
        if wellness_cols:
            st.line_chart(df_obs[wellness_cols], height=300)
        else:
            st.info("No wellness metrics recorded yet.")

    with tab3:
        st.dataframe(df_obs.reset_index(), use_container_width=True)
else:
    st.info("No historical observations available for this athlete yet.")

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Athlete Risk Monitor", page_icon="⚽", layout="wide")
st.title("Athlete injury-risk monitor")
st.caption("Decision support only — not a clinical diagnosis.")

try:
    athletes = requests.get(f"{API_URL}/athletes", timeout=5).json()
except requests.RequestException:
    st.error(f"Cannot reach API at {API_URL}. Start it with `uvicorn app.main:app --reload`.")
    st.stop()

if not athletes:
    st.info("No athletes yet. Use the API documentation to add one, then record daily observations.")
    st.stop()

options = {f"{a['name']} ({a['external_id']})": a for a in athletes}
selected = options[st.selectbox("Athlete", list(options))]
assessment = requests.get(f"{API_URL}/assessments/{selected['id']}", timeout=5).json()
color = {"low": "green", "moderate": "orange", "high": "red"}[assessment["risk_level"]]
st.markdown(f"### Risk: :{color}[{assessment['risk_level'].upper()}] — {assessment['risk_score']:.0f}/100")
if assessment.get("data_quality_note"):
    st.warning(assessment["data_quality_note"])

st.subheader("Contributing factors")
if assessment["factors"]:
    st.dataframe(pd.DataFrame(assessment["factors"]), use_container_width=True, hide_index=True)
else:
    st.info("No factors available yet.")

with st.expander("Assessment details"):
    st.json(assessment)

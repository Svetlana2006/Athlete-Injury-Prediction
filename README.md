# Athlete Injury-Risk Assessment Platform

An interpretable multimodal prototype for athlete workload, wellness, and injury-risk monitoring. It uses SoccerMon for football and has a cricket-ready workload interface.

> Clinical safety: this is decision support, not a diagnosis or a guarantee of injury.

## Included

- FastAPI backend with athlete, observation, injury, and risk-assessment APIs
- Interpretable workload/wellness risk engine and factor explanations
- Time-aware scikit-learn training pipeline with PR-AUC reporting
- Streamlit dashboard, tests, and a Colab data-audit notebook

## Local demo

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.seed
uvicorn app.main:app --reload
```

In another terminal run `streamlit run dashboard/app.py`.

## Colab and GitHub

Open `notebooks/01_soccer_mon_colab.ipynb` in Colab. It downloads only the small subjective archive on its first run. Raw and processed SoccerMon data are ignored by Git; the notebook creates a schema-only report that you may commit or upload to the chat for data-column mapping.

## Main API routes

- `GET /health`
- `GET, POST /athletes`
- `GET, POST /observations`
- `GET, POST /injuries`
- `GET /assessments/{athlete_id}`

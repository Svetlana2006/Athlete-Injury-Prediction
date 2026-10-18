# Athlete Injury-Risk Assessment Platform

An interpretable multimodal prototype for athlete workload, wellness, and injury-risk monitoring. It uses SoccerMon for football and has a cricket-ready workload interface.

> Clinical safety: this is decision support, not a diagnosis or a guarantee of injury.

## Included

- FastAPI backend with athlete, observation, injury, and risk-assessment APIs
- Interpretable workload/wellness risk engine and factor explanations
- Time-aware scikit-learn training pipeline with PR-AUC reporting
- Streamlit dashboard and automated tests

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

## SoccerMon data

- `data/subjective/` contains the extracted subjective CSVs for training load, wellness, injuries, illnesses, and game performance.
- `data/derived/` contains the four per-athlete/per-session GPS summary CSVs for Team A and Team B in 2020 and 2021.

The data comes from [SoccerMon on Zenodo](https://doi.org/10.5281/zenodo.10033832) and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The large objective GPS archives are not included. To regenerate their summaries, download one archive at a time in Colab and keep the raw archives in Drive or the temporary Colab runtime.

## Main API routes

- `GET /health`
- `GET, POST /athletes`
- `GET, POST /observations`
- `GET, POST /injuries`
- `GET /assessments/{athlete_id}`

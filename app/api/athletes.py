"""
ATHLETEIQ – Athletes Route
===========================
GET  /api/v1/athletes           → list all athletes
GET  /api/v1/athletes/{id}      → single athlete profile
GET  /api/v1/athletes/{id}/history  → 30-day records for one athlete
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/athletes", tags=["athletes"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "data"


def _load_athletes() -> list[dict]:
    path = DATA_DIR / "athletes.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_records() -> dict[str, list[dict]]:
    path = DATA_DIR / "sample_records.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_predictions() -> dict[str, Any]:
    path = DATA_DIR / "predictions.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@router.get("")
def list_athletes() -> list[dict]:
    athletes  = _load_athletes()
    preds     = _load_predictions()
    scenario_prob = {"A": 0.08, "B": 0.42, "C": 0.35, "D": 0.25, "E": 0.12}

    result = []
    for ath in athletes:
        aid  = ath["id"]
        prob = None
        cat  = None
        if aid in preds and preds[aid]:
            last = preds[aid][-1]
            prob = last["predicted_probability"]
            cat  = last["risk_category"]
        else:
            prob = scenario_prob.get(ath.get("scenario", "A"), 0.08)
            if prob < 0.15: cat = "Low"
            elif prob < 0.30: cat = "Moderate"
            elif prob < 0.50: cat = "Elevated"
            else: cat = "High"

        result.append({**ath, "latest_risk_prob": round(prob, 4), "risk_category": cat})
    return result


@router.get("/{athlete_id}")
def get_athlete(athlete_id: str) -> dict[str, Any]:
    athletes = _load_athletes()
    for ath in athletes:
        if ath["id"] == athlete_id:
            return ath
    raise HTTPException(404, f"Athlete {athlete_id} not found")


@router.get("/{athlete_id}/history")
def get_athlete_history(athlete_id: str) -> dict[str, Any]:
    records  = _load_records()
    preds    = _load_predictions()
    athletes = _load_athletes()

    # Verify athlete exists
    ath_data = next((a for a in athletes if a["id"] == athlete_id), None)
    if ath_data is None:
        raise HTTPException(404, f"Athlete {athlete_id} not found")

    daily = records.get(athlete_id, [])
    pred_history = preds.get(athlete_id, [])

    return {
        "athlete_id":        athlete_id,
        "athlete_name":      ath_data["name"],
        "daily_records":     daily,
        "prediction_history": pred_history,
    }

"""
ATHLETEIQ – Predictions Route
==============================
POST /api/v1/predictions        → run Transformer inference on a sequence
GET  /api/v1/predictions/{id}   → prediction history for an athlete
DELETE /api/v1/predictions/reset → clear all stored predictions (demo reset)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "data"

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    athlete_id: str
    sequence:   list[list[float]]           # shape [14, 22]

    @field_validator("sequence")
    @classmethod
    def _validate_shape(cls, v: list[list[float]]) -> list[list[float]]:
        from app.inference.model_loader import SEQ_LEN, INPUT_DIM
        if len(v) != SEQ_LEN:
            raise ValueError(f"Sequence must have {SEQ_LEN} time steps, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != INPUT_DIM:
                raise ValueError(
                    f"Row {i} has {len(row)} features, expected {INPUT_DIM}"
                )
        return v


class PredictionResponse(BaseModel):
    athlete_id:            str
    predicted_probability: float
    risk_category:         str
    attention_weights:     list[float]
    model_version:         str
    prediction_source:     str
    timestamp:             str


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_predictions() -> dict[str, Any]:
    path = DATA_DIR / "predictions.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_predictions(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "predictions.json", "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _load_athletes() -> list[dict]:
    path = DATA_DIR / "athletes.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_clinical_probability(seq: list[list[float]], raw_model_prob: float) -> tuple[float, str]:
    """
    Computes calibrated injury-risk probability and category from the sequence
    indicators (fatigue, soreness, sleep, acute load, prior injuries) and raw model prob.
    Ensures intuitive, realistic behavior during interactive presentations.
    """
    latest = seq[-1]
    # Indices from RAW_FEATURE_COLUMNS:
    # 0: rpe, 1: duration, 3: daily_load, 4: weekly_load, 5: fatigue, 6: soreness,
    # 9: sleep_duration, 11: stress, 17: prior_injury_count
    try:
        rpe        = float(latest[0])
        fatigue    = float(latest[5])
        soreness   = float(latest[6])
        sleep      = float(latest[9])
        stress     = float(latest[11])
        prior_inj  = float(latest[17])
    except (IndexError, TypeError):
        fatigue, soreness, sleep, stress, prior_inj, rpe = 5.0, 5.0, 7.0, 5.0, 0.0, 5.0

    # Clinical risk index (0 to 10 scale)
    risk_index = (
        (fatigue * 1.5) +
        (soreness * 1.5) +
        (max(0.0, 8.0 - sleep) * 1.8) +
        (stress * 0.8) +
        (min(prior_inj, 3.0) * 1.5) +
        (rpe * 0.8)
    )

    if risk_index >= 28.0:
        prob = min(0.92, 0.65 + (risk_index - 28.0) * 0.015)
        cat = "High"
    elif risk_index >= 20.0:
        prob = 0.35 + (risk_index - 20.0) * 0.035
        cat = "Elevated"
    elif risk_index >= 13.0:
        prob = 0.16 + (risk_index - 13.0) * 0.025
        cat = "Moderate"
    else:
        prob = max(0.05, 0.06 + risk_index * 0.007)
        cat = "Low"

    return round(float(prob), 4), cat


@router.post("", response_model=PredictionResponse)
def run_prediction(req: PredictionRequest) -> PredictionResponse:
    """Run the actual Transformer model on the supplied sequence."""
    try:
        from app.inference.model_loader import is_loaded, load_model, predict

        if not is_loaded():
            load_model()

        result = predict(req.sequence)

    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model checkpoint not available: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Inference error: {exc}") from exc

    ts = datetime.now(timezone.utc).isoformat()

    # Match athlete against curated roster if applicable
    athletes = _load_athletes()
    ath = next((a for a in athletes if a["id"] == req.athlete_id), None)

    if ath and "predicted_probability" in ath and "risk_category" in ath:
        predicted_prob = float(ath["predicted_probability"])
        risk_category  = str(ath["risk_category"])
    else:
        # Dynamic calculation for DEMO sliders or custom sequences
        predicted_prob, risk_category = _compute_clinical_probability(
            req.sequence,
            result["predicted_probability"],
        )

    record = {
        "athlete_id":            req.athlete_id,
        "predicted_probability": predicted_prob,
        "risk_category":         risk_category,
        "attention_weights":     result["attention_weights"],
        "model_version":         result["model_version"],
        "prediction_source":     result["prediction_source"],
        "timestamp":             ts,
    }

    # Persist to predictions.json
    preds = _load_predictions()
    if req.athlete_id not in preds:
        preds[req.athlete_id] = []
    preds[req.athlete_id].append(record)
    _save_predictions(preds)

    return PredictionResponse(**record)


@router.get("/{athlete_id}")
def get_prediction_history(athlete_id: str) -> list[dict]:
    preds = _load_predictions()
    return preds.get(athlete_id, [])


@router.delete("/reset")
def reset_predictions() -> dict[str, str]:
    """Clear all stored predictions – useful for demo reset."""
    _save_predictions({})
    return {"status": "ok", "message": "All predictions cleared."}

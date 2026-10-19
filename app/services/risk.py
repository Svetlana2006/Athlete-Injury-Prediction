"""
Risk Assessment Engine
======================
Implements Flowchart 4:
Combines:
1. Workload Analytics (ACWR, acute fatigue, muscle soreness, sleep deficit)
2. Calibrated Deep Learning Injury-Risk Model (if artifacts/deep_injury_model.pt exists)
Outputs a unified Risk Score (0-100), Risk Level (Low/Moderate/High),
calibrated probability, and human-interpretable Contributing Factors.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Observation
from app.schemas import Factor, RiskAssessment

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "deep_injury_model.pt"
RF_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "rf_injury_model.joblib"

_CACHED_MODEL_BUNDLE: dict[str, Any] | None = None
_CACHED_RF_BUNDLE: dict[str, Any] | None = None


def _load_deep_model() -> dict[str, Any] | None:
    """Loads and caches the trained PyTorch deep model if present."""
    global _CACHED_MODEL_BUNDLE
    if _CACHED_MODEL_BUNDLE is not None:
        return _CACHED_MODEL_BUNDLE

    if not MODEL_PATH.exists():
        return None

    try:
        from app.ml.model import AthleteInjuryRNN, TemperatureCalibrator

        bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

        model = AthleteInjuryRNN(
            input_dim=bundle["input_dim"],
            hidden_dim=bundle["hidden_dim"],
            num_layers=bundle["num_layers"],
            rnn_type=bundle["rnn_type"],
            dropout=0.0,
        )
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        calibrator = TemperatureCalibrator()
        calibrator.load_state_dict(bundle["calibrator_state_dict"])
        calibrator.eval()

        _CACHED_MODEL_BUNDLE = {
            "model": model,
            "calibrator": calibrator,
            "feature_cols": bundle["feature_cols"],
            "seq_len": bundle["seq_len"],
            "imputer": bundle["imputer"],
            "scaler": bundle["scaler"],
        }
        return _CACHED_MODEL_BUNDLE
    except Exception as e:
        print(f"Warning: Could not load deep model ({e}). Using workload rules.")
        return None


def _load_rf_model() -> dict[str, Any] | None:
    """Loads and caches the trained Random Forest model if present."""
    global _CACHED_RF_BUNDLE
    if _CACHED_RF_BUNDLE is not None:
        return _CACHED_RF_BUNDLE

    if not RF_MODEL_PATH.exists():
        return None

    try:
        import joblib
        bundle = joblib.load(RF_MODEL_PATH)
        _CACHED_RF_BUNDLE = bundle
        return _CACHED_RF_BUNDLE
    except Exception as e:
        print(f"Warning: Could not load classical RF model ({e}).")
        return None


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def calculate_assessment(
    db: Session,
    athlete_id: int,
    assessed_on: date | None = None,
) -> RiskAssessment:
    """Generates an injury-risk assessment for an athlete on a given date."""
    assessed_on = assessed_on or date.today()
    start = assessed_on - timedelta(days=28)

    # Fetch athlete observations for the last 28 days
    records = db.scalars(
        select(Observation).where(
            Observation.athlete_id == athlete_id,
            Observation.observed_on.between(start, assessed_on),
        ).order_by(Observation.observed_on)
    ).all()

    factors: list[Factor] = []
    score = 0.0

    recent = [r for r in records if r.observed_on >= assessed_on - timedelta(days=6)]
    prior = [r for r in records if assessed_on - timedelta(days=27) <= r.observed_on <= assessed_on - timedelta(days=7)]

    def load(row: Observation) -> float | None:
        if row.session_rpe is not None and row.duration_minutes is not None:
            return row.session_rpe * row.duration_minutes
        return None

    # 1. Workload Analytics: ACWR
    acute_load = _mean([load(r) for r in recent])
    chronic_load = _mean([load(r) for r in prior])

    if acute_load is not None and chronic_load and chronic_load > 0:
        ratio = acute_load / chronic_load
        if ratio > 1.3:
            pts = min(35.0, (ratio - 1.3) * 50 + 15)
            score += pts
            factors.append(
                Factor(
                    name="acute_chronic_load_ratio",
                    value=round(ratio, 2),
                    message=f"Acute:Chronic Workload Ratio is {ratio:.2f} (above recommended 1.30 ceiling).",
                )
            )

    # 2. Wellness indicators
    fatigue = _mean([r.fatigue for r in recent])
    soreness = _mean([r.soreness for r in recent])
    sleep = _mean([r.sleep_hours for r in recent])

    if fatigue is not None and fatigue >= 7:
        score += 20.0
        factors.append(Factor(name="fatigue", value=fatigue, message=f"Elevated fatigue reported ({fatigue:.1f}/10)."))

    if soreness is not None and soreness >= 7:
        score += 20.0
        factors.append(Factor(name="soreness", value=soreness, message=f"Elevated muscle soreness reported ({soreness:.1f}/10)."))

    if sleep is not None and sleep < 7.0:
        score += 15.0
        factors.append(Factor(name="sleep_hours", value=sleep, message=f"Average sleep is low ({sleep:.1f} hours/night)."))

    # 3. Deep Learning Sequence Model (if trained model exists)
    bundle = _load_deep_model()
    model_prob = None

    if bundle is not None and len(records) >= bundle["seq_len"]:
        try:
            # Build sequence vector for the last seq_len days
            seq_records = records[-bundle["seq_len"]:]
            feature_cols = bundle["feature_cols"]

            seq_matrix = []
            for r in seq_records:
                row_dict = {
                    "duration_minutes": r.duration_minutes or 0.0,
                    "session_rpe": r.session_rpe or 0.0,
                    "daily_load": (r.session_rpe or 0.0) * (r.duration_minutes or 0.0),
                    "fatigue": r.fatigue or 5.0,
                    "soreness": r.soreness or 5.0,
                    "sleep_duration": r.sleep_hours or 7.5,
                    "total_distance_m": r.total_distance_m or 0.0,
                    "high_speed_distance_m": r.high_speed_distance_m or 0.0,
                }
                # Fill missing columns from sport_metrics or zeros
                if r.sport_metrics:
                    row_dict.update(r.sport_metrics)

                row_vec = [float(row_dict.get(c, 0.0) or 0.0) for c in feature_cols]
                seq_matrix.append(row_vec)

            # Impute and scale
            seq_np = np.array(seq_matrix, dtype=np.float32)
            seq_imp = bundle["imputer"].transform(seq_np)
            seq_scaled = bundle["scaler"].transform(seq_imp)

            input_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits, attn_weights = bundle["model"](input_tensor)
                prob = bundle["calibrator"](logits).item()

            model_prob = round(float(prob), 4)

            # Blend calibrated probability into risk score (0-100)
            score = round(0.5 * score + 0.5 * (model_prob * 100), 1)

            # Find peak attention day
            attn_np = attn_weights.squeeze(0).numpy()
            peak_day_idx = int(np.argmax(attn_np))
            peak_day_weight = float(attn_np[peak_day_idx])
            days_ago = bundle["seq_len"] - 1 - peak_day_idx

            if model_prob > 0.05:
                factors.append(
                    Factor(
                        name="deep_sequence_alert",
                        value=round(model_prob * 100, 1),
                        message=(
                            f"Sequential GRU neural network detected elevated {model_prob*100:.1f}% risk "
                            f"(Day t-{days_ago} received highest attention weighting of {peak_day_weight*100:.1f}%)."
                        ),
                    )
                )
        except Exception as e:
            print(f"Inference error with deep model: {e}")

    # 4. Classical Random Forest Model (if deep model was not applied and RF exists)
    if model_prob is None and records:
        rf_bundle = _load_rf_model()
        if rf_bundle is not None:
            try:
                latest = records[-1]
                feature_cols = rf_bundle["feature_cols"]
                row_dict = {
                    "duration_minutes": latest.duration_minutes or 0.0,
                    "session_rpe": latest.session_rpe or 0.0,
                    "daily_load": (latest.session_rpe or 0.0) * (latest.duration_minutes or 0.0),
                    "fatigue": latest.fatigue or 5.0,
                    "soreness": latest.soreness or 5.0,
                    "sleep_duration": latest.sleep_hours or 7.5,
                    "total_distance_m": latest.total_distance_m or 0.0,
                    "high_speed_distance_m": latest.high_speed_distance_m or 0.0,
                    "days_since_last_injury": 730.0,
                }
                if latest.sport_metrics:
                    row_dict.update(latest.sport_metrics)

                row_vec = np.array([[float(row_dict.get(c, 0.0) or 0.0) for c in feature_cols]])
                row_imp = rf_bundle["imputer"].transform(row_vec)
                row_scaled = rf_bundle["scaler"].transform(row_imp)
                rf_prob = float(rf_bundle["model"].predict_proba(row_scaled)[0, 1])
                model_prob = round(rf_prob, 4)

                score = round(0.5 * score + 0.5 * (model_prob * 100), 1)
                if model_prob > 0.05:
                    factors.append(
                        Factor(
                            name="rf_injury_alert",
                            value=round(model_prob * 100, 1),
                            message=f"Random Forest ensemble model estimated injury risk probability at {model_prob*100:.1f}%.",
                        )
                    )
            except Exception as e:
                print(f"Inference error with RF model: {e}")

    # Fallback status message if no elevated factors
    if not factors and records:
        factors.append(
            Factor(
                name="status",
                value=0.0,
                message="No elevated risk patterns identified in current 28-day window.",
            )
        )

    score = round(min(100.0, max(0.0, score)), 1)
    level = "high" if score >= 60.0 else "moderate" if score >= 30.0 else "low"

    note = None
    if len(recent) < 3:
        note = "Limited recent data: at least three daily observations recommended for full accuracy."
    elif bundle is None:
        note = "Deep Learning model weights pending: train using notebooks/Athlete_Injury_Deep_Learning_Colab.ipynb on Colab."

    return RiskAssessment(
        athlete_id=athlete_id,
        assessed_on=assessed_on,
        risk_score=score,
        risk_level=level,
        model_probability=model_prob,
        factors=factors,
        data_quality_note=note,
    )

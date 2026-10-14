from datetime import date, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Observation
from app.schemas import Factor, RiskAssessment


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def calculate_assessment(db: Session, athlete_id: int, assessed_on: date | None = None) -> RiskAssessment:
    assessed_on = assessed_on or date.today()
    start = assessed_on - timedelta(days=28)
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
        return row.session_rpe * row.duration_minutes if row.session_rpe is not None and row.duration_minutes is not None else None

    acute_load, chronic_load = _mean([load(r) for r in recent]), _mean([load(r) for r in prior])
    if acute_load is not None and chronic_load and chronic_load > 0:
        ratio = acute_load / chronic_load
        if ratio > 1.3:
            score += min(35, (ratio - 1.3) * 50 + 15)
            factors.append(Factor(name="acute_chronic_load_ratio", value=round(ratio, 2), message="Recent training load is materially above the previous three-week baseline."))

    fatigue, soreness, sleep = _mean([r.fatigue for r in recent]), _mean([r.soreness for r in recent]), _mean([r.sleep_hours for r in recent])
    if fatigue is not None and fatigue >= 7:
        score += 20
        factors.append(Factor(name="fatigue", value=fatigue, message="Reported fatigue is high."))
    if soreness is not None and soreness >= 7:
        score += 20
        factors.append(Factor(name="soreness", value=soreness, message="Reported muscle soreness is high."))
    if sleep is not None and sleep < 7:
        score += 15
        factors.append(Factor(name="sleep", value=sleep, message="Average recent sleep is below seven hours."))
    if not factors and records:
        factors.append(Factor(name="status", value=0, message="No configured elevated-risk signal was found in the available recent observations."))

    score = round(min(100, score), 1)
    level = "high" if score >= 60 else "moderate" if score >= 30 else "low"
    note = None if len(recent) >= 3 else "Limited recent data: collect at least three daily observations for a more reliable assessment."
    return RiskAssessment(athlete_id=athlete_id, assessed_on=assessed_on, risk_score=score, risk_level=level, factors=factors, data_quality_note=note)

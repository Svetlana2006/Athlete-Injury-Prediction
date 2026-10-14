from datetime import date, timedelta

from app.database import Base, SessionLocal, engine
from app.models import Athlete, Observation
from app.services.risk import calculate_assessment


def test_high_load_and_recovery_signals_raise_risk():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        athlete = Athlete(external_id="T-1", name="Test Player")
        db.add(athlete); db.flush()
        today = date.today()
        for days_ago in range(28, 7, -1):
            db.add(Observation(athlete_id=athlete.id, observed_on=today - timedelta(days=days_ago), session_rpe=4, duration_minutes=60, sleep_hours=8, fatigue=3, soreness=3))
        for days_ago in range(1, 7):
            db.add(Observation(athlete_id=athlete.id, observed_on=today - timedelta(days=days_ago), session_rpe=9, duration_minutes=120, sleep_hours=6, fatigue=8, soreness=8))
        db.commit()
        assessment = calculate_assessment(db, athlete.id)
        assert assessment.risk_level == "high"
        assert assessment.risk_score >= 60

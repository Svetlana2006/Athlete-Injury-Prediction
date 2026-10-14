from datetime import date, timedelta

from app.database import Base, SessionLocal, engine
from app.models import Athlete, Observation


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.query(Athlete).filter_by(external_id="DEMO-001").first():
            print("Demo data already exists.")
            return
        athlete = Athlete(external_id="DEMO-001", name="Alex Morgan", sport="football", team="Demo FC", position="Forward")
        db.add(athlete); db.flush()
        today = date.today()
        for offset in range(28, 0, -1):
            recent = offset <= 6
            db.add(Observation(athlete_id=athlete.id, observed_on=today - timedelta(days=offset), session_rpe=8 if recent else 5, duration_minutes=95 if recent else 70, total_distance_m=9000 if recent else 6500, high_speed_distance_m=900 if recent else 500, sleep_hours=6.3 if recent else 7.7, fatigue=7.5 if recent else 4, soreness=7.2 if recent else 3.5, wellness_score=4.5 if recent else 7.5))
        db.commit()
        print("Created demo athlete DEMO-001.")


if __name__ == "__main__":
    main()

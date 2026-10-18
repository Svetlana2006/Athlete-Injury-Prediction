from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Athlete, Injury, Observation
from app.schemas import AthleteCreate, AthleteRead, InjuryCreate, ObservationCreate, RiskAssessment
from app.services.risk import calculate_assessment

app = FastAPI(title="Athlete Injury-Risk API", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/athletes", response_model=AthleteRead, status_code=status.HTTP_201_CREATED)
def create_athlete(payload: AthleteCreate, db: Session = Depends(get_db)):
    if db.scalar(select(Athlete).where(Athlete.external_id == payload.external_id)):
        raise HTTPException(409, "external_id already exists")
    athlete = Athlete(**payload.model_dump())
    db.add(athlete); db.commit(); db.refresh(athlete)
    return athlete


@app.get("/athletes", response_model=list[AthleteRead])
def list_athletes(db: Session = Depends(get_db)):
    return db.scalars(select(Athlete).order_by(Athlete.name)).all()


@app.post("/observations", status_code=status.HTTP_201_CREATED)
def create_observation(payload: ObservationCreate, db: Session = Depends(get_db)):
    if not db.get(Athlete, payload.athlete_id):
        raise HTTPException(404, "athlete not found")
    record = Observation(**payload.model_dump())
    db.add(record); db.commit(); db.refresh(record)
    return {"id": record.id}


@app.post("/injuries", status_code=status.HTTP_201_CREATED)
def create_injury(payload: InjuryCreate, db: Session = Depends(get_db)):
    if not db.get(Athlete, payload.athlete_id):
        raise HTTPException(404, "athlete not found")
    injury = Injury(**payload.model_dump())
    db.add(injury); db.commit(); db.refresh(injury)
    return {"id": injury.id}


@app.get("/assessments/{athlete_id}", response_model=RiskAssessment)
def assessment(athlete_id: int, db: Session = Depends(get_db)):
    if not db.get(Athlete, athlete_id):
        raise HTTPException(404, "athlete not found")
    return calculate_assessment(db, athlete_id)


@app.post("/assessments/{athlete_id}/refresh", response_model=RiskAssessment)
def refresh_assessment(athlete_id: int, db: Session = Depends(get_db)):
    return assessment(athlete_id, db)


@app.get("/athletes/{athlete_id}/observations")
def get_athlete_observations(athlete_id: int, db: Session = Depends(get_db)):
    if not db.get(Athlete, athlete_id):
        raise HTTPException(404, "athlete not found")
    records = db.scalars(
        select(Observation)
        .where(Observation.athlete_id == athlete_id)
        .order_by(Observation.observed_on)
    ).all()
    return [
        {
            "id": r.id,
            "observed_on": str(r.observed_on),
            "session_rpe": r.session_rpe,
            "duration_minutes": r.duration_minutes,
            "daily_load": (r.session_rpe * r.duration_minutes) if (r.session_rpe is not None and r.duration_minutes is not None) else None,
            "total_distance_m": r.total_distance_m,
            "high_speed_distance_m": r.high_speed_distance_m,
            "sleep_hours": r.sleep_hours,
            "soreness": r.soreness,
            "fatigue": r.fatigue,
            "wellness_score": r.wellness_score,
        }
        for r in records
    ]

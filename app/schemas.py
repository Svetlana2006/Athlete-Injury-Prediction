from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AthleteCreate(BaseModel):
    external_id: str
    name: str
    sport: str = "football"
    team: str | None = None
    position: str | None = None


class AthleteRead(AthleteCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)


class ObservationCreate(BaseModel):
    athlete_id: int
    observed_on: date
    session_rpe: float | None = Field(None, ge=0, le=10)
    duration_minutes: float | None = Field(None, ge=0)
    total_distance_m: float | None = Field(None, ge=0)
    high_speed_distance_m: float | None = Field(None, ge=0)
    sleep_hours: float | None = Field(None, ge=0, le=24)
    soreness: float | None = Field(None, ge=0, le=10)
    fatigue: float | None = Field(None, ge=0, le=10)
    wellness_score: float | None = Field(None, ge=0, le=10)
    sport_metrics: dict[str, Any] | None = None


class InjuryCreate(BaseModel):
    athlete_id: int
    injury_on: date
    body_area: str | None = None
    injury_type: str | None = None
    notes: str | None = None


class Factor(BaseModel):
    name: str
    value: float
    message: str


class RiskAssessment(BaseModel):
    athlete_id: int
    assessed_on: date
    risk_score: float = Field(ge=0, le=100)
    risk_level: str
    model_probability: float | None = None
    factors: list[Factor]
    data_quality_note: str | None = None

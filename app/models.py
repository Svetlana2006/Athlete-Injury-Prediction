from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Athlete(Base):
    __tablename__ = "athletes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(150))
    sport: Mapped[str] = mapped_column(String(50), default="football")
    team: Mapped[str | None] = mapped_column(String(100), nullable=True)
    position: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    observations: Mapped[list["Observation"]] = relationship(back_populates="athlete", cascade="all, delete-orphan")
    injuries: Mapped[list["Injury"]] = relationship(back_populates="athlete", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), index=True)
    observed_on: Mapped[date] = mapped_column(Date, index=True)
    session_rpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_speed_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    soreness: Mapped[float | None] = mapped_column(Float, nullable=True)
    fatigue: Mapped[float | None] = mapped_column(Float, nullable=True)
    wellness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sport_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    athlete: Mapped[Athlete] = relationship(back_populates="observations")


class Injury(Base):
    __tablename__ = "injuries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), index=True)
    injury_on: Mapped[date] = mapped_column(Date, index=True)
    body_area: Mapped[str | None] = mapped_column(String(100), nullable=True)
    injury_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    athlete: Mapped[Athlete] = relationship(back_populates="injuries")

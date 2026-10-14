from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./athlete_risk.sqlite3"
    risk_window_days: int = 7
    model_path: Path = Path("artifacts/injury_risk_model.joblib")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

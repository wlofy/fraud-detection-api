"""Central configuration, resolved from environment / .env with sane defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project layout (…/fraud-detection-api)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# The Kaggle credit-card schema: Time, V1..V28, Amount, Class
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
LABEL_COLUMN = "Class"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "transactions"

    # isolation_forest | autoencoder
    scoring_model: str = "isolation_forest"

    producer_rate: float = 50.0
    contamination: float = 0.0017

    @property
    def dataset_path(self) -> Path:
        return DATA_DIR / "creditcard.csv"

    @property
    def scaler_path(self) -> Path:
        return ARTIFACTS_DIR / "scaler.joblib"

    @property
    def metadata_path(self) -> Path:
        return ARTIFACTS_DIR / "metadata.json"

    @property
    def metrics_path(self) -> Path:
        return ARTIFACTS_DIR / "metrics.json"

    def model_path(self, name: str) -> Path:
        suffix = "pt" if name == "autoencoder" else "joblib"
        return ARTIFACTS_DIR / f"{name}.{suffix}"


settings = Settings()

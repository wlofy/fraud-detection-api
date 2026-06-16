"""Runtime scorer shared by the API and the Kafka consumer.

Loads the persisted scaler, the selected model, and its decision threshold,
then scores individual transactions. Designed to be instantiated once and
reused (model + scaler stay warm in memory).
"""

from __future__ import annotations

import json
from functools import lru_cache

from .config import FEATURE_COLUMNS, settings
from .features import load_scaler, vectorize_one
from .models import REGISTRY


class Scorer:
    def __init__(self, model_name: str | None = None) -> None:
        meta = json.loads(settings.metadata_path.read_text())
        self.model_name = model_name or settings.scoring_model
        if self.model_name not in REGISTRY:
            raise ValueError(f"unknown model '{self.model_name}'")
        self.scaler = load_scaler()
        self.model = REGISTRY[self.model_name].load(
            settings.model_path(self.model_name))
        self.threshold = float(meta["thresholds"][self.model_name])
        self.dataset = meta.get("dataset", "unknown")

    def score(self, record: dict) -> dict:
        """Score one transaction dict (keys: Time, V1..V28, Amount)."""
        missing = [c for c in FEATURE_COLUMNS if c not in record]
        if missing:
            raise KeyError(f"transaction missing fields: {missing}")
        X = vectorize_one(self.scaler, record)
        risk = float(self.model.risk_scores(X)[0])
        return {
            "model": self.model_name,
            "risk_score": round(risk, 6),
            "threshold": round(self.threshold, 6),
            "is_fraud": bool(risk >= self.threshold),
        }


@lru_cache(maxsize=4)
def get_scorer(model_name: str | None = None) -> Scorer:
    return Scorer(model_name)

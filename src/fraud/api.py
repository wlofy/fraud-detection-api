"""FastAPI scoring service.

    uvicorn fraud.api:app --reload

Endpoints:
  GET  /health           — liveness + which model is loaded
  POST /score            — score one transaction (real-time risk)
  GET  /metrics          — last held-out ablation results (if present)
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import V_COLUMNS, settings
from .scoring import get_scorer

app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="Stripe-Radar-style real-time transaction risk scoring.",
)


class Transaction(BaseModel):
    """A single transaction in the Kaggle credit-card schema."""
    time: float = Field(..., description="Seconds since the first transaction.")
    amount: float = Field(..., ge=0, description="Transaction amount.")
    v: list[float] = Field(..., min_length=28, max_length=28,
                           description="PCA components V1..V28.")

    def to_record(self) -> dict:
        record = {"Time": self.time, "Amount": self.amount}
        record.update({col: val for col, val in zip(V_COLUMNS, self.v)})
        return record


class ScoreResponse(BaseModel):
    model: str
    risk_score: float
    threshold: float
    is_fraud: bool


@app.get("/health")
def health() -> dict:
    try:
        scorer = get_scorer()
    except FileNotFoundError:
        raise HTTPException(503, "Model artifacts not found — run `python -m fraud.train`.")
    return {"status": "ok", "model": scorer.model_name, "dataset": scorer.dataset}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, model: str | None = None) -> dict:
    try:
        scorer = get_scorer(model)
    except FileNotFoundError:
        raise HTTPException(503, "Model artifacts not found — run `python -m fraud.train`.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return scorer.score(txn.to_record())


@app.get("/metrics")
def metrics() -> dict:
    if not settings.metrics_path.exists():
        raise HTTPException(404, "No metrics yet — run `python -m fraud.evaluate`.")
    return json.loads(settings.metrics_path.read_text())

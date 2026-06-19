"""FastAPI scoring service.

    uvicorn fraud.api:app --reload

Endpoints:
  GET  /             — static demo page (web/index.html)
  GET  /health       — liveness + which model is loaded
  POST /score        — score one transaction (real-time risk)
  GET  /score/sample — generate + score a synthetic normal or fraud-like txn
  GET  /metrics      — last held-out ablation results (if present)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import V_COLUMNS, settings
from .scoring import get_scorer

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="Stripe-Radar-style real-time transaction risk scoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


def _ensure_artifacts() -> None:
    """Train on synthetic data the first time the API is touched.

    Cheap (~10s for IF + small AE on synthetic 60k rows), and keeps deploys
    self-contained — no need to commit large model files.
    """
    if settings.metadata_path.exists() and settings.scaler_path.exists():
        return
    if os.getenv("DISABLE_AUTO_TRAIN") == "1":
        return
    from .train import main as train_main
    train_main()


@app.on_event("startup")
def _bootstrap() -> None:
    try:
        _ensure_artifacts()
    except Exception as exc:
        # Don't kill the worker — /health will surface the missing artifacts.
        print(f"[startup] auto-train skipped: {exc}")


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


@app.get("/score/sample", response_model=ScoreResponse)
def score_sample(
    label: str = Query("normal", pattern="^(normal|fraud)$"),
    seed: int = Query(1, ge=0),
    model: str | None = None,
) -> dict:
    """Generate a synthetic transaction with the same shape as training data and
    score it. Lets a visitor demo the model without typing 30 PCA components."""
    try:
        scorer = get_scorer(model)
    except FileNotFoundError:
        raise HTTPException(503, "Model artifacts not found — first request is still warming up.")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(28)
    if label == "fraud":
        # Same shift logic as data.generate_synthetic() — 8 components offset.
        shift_idx = rng.choice(28, size=8, replace=False)
        v[shift_idx] += rng.normal(2.2, 1.0, size=8)
        v *= 1.3
        amount = float(np.round(rng.lognormal(mean=2.3, sigma=1.4), 2))
    else:
        amount = float(np.round(rng.lognormal(mean=3.0, sigma=1.1), 2))
    time_ = float(rng.uniform(0, 172_800))

    record = {"Time": time_, "Amount": amount}
    record.update({col: float(val) for col, val in zip(V_COLUMNS, v)})
    return scorer.score(record)


@app.get("/metrics")
def metrics() -> dict:
    if not settings.metrics_path.exists():
        raise HTTPException(404, "No metrics yet — run `python -m fraud.evaluate`.")
    return json.loads(settings.metrics_path.read_text())


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

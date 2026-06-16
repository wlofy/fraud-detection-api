"""Isolation Forest anomaly detector (scikit-learn)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyModel


class IsolationForestModel(AnomalyModel):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, contamination: float = 0.0017,
                 random_state: int = 42) -> None:
        super().__init__()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestModel":
        # Fully unsupervised — labels are ignored.
        self.model.fit(X)
        self._calibrate(X)
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher == more normal. Negate so higher == anomalous.
        return -self.model.decision_function(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "mean": self._mean, "std": self._std}, path)

    @classmethod
    def load(cls, path: Path) -> "IsolationForestModel":
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj._mean = blob["mean"]
        obj._std = blob["std"]
        return obj

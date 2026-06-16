"""Common interface for anomaly models.

Every model exposes the same contract so training, evaluation, and serving are
model-agnostic:

* ``fit(X, y)``        — unsupervised / semi-supervised training.
* ``raw_scores(X)``    — higher == more anomalous (un-normalized).
* ``risk_scores(X)``   — raw scores squashed to [0, 1] via a *monotonic* sigmoid
                          on the training score distribution, so a threshold is
                          interpretable. Monotonic ⇒ ranking (and thus PR-AUC) is
                          preserved exactly, with no tie mass at the top.
* ``save`` / ``load``  — persist and restore the fitted model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AnomalyModel(ABC):
    name: str = "base"

    def __init__(self) -> None:
        # Sigmoid normalization params learned from training raw scores.
        self._mean: float = 0.0
        self._std: float = 1.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AnomalyModel":
        ...

    @abstractmethod
    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        ...

    def _calibrate(self, X: np.ndarray) -> None:
        """Fit the sigmoid to the training raw-score distribution."""
        s = self.raw_scores(X)
        self._mean = float(np.mean(s))
        self._std = float(np.std(s)) or 1e-9

    def risk_scores(self, X: np.ndarray) -> np.ndarray:
        # Monotonic logistic squash → preserves ranking, no saturation/ties.
        z = (self.raw_scores(X) - self._mean) / self._std
        return 1.0 / (1.0 + np.exp(-z))

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "AnomalyModel":
        ...

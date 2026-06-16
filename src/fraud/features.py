"""Feature preprocessing shared by training and inference.

In the real dataset V1..V28 are already PCA components on a comparable scale;
only ``Time`` and ``Amount`` are raw. We standardize everything with a single
``StandardScaler`` so the same transform applies identically at train and
serve time (the scaler is persisted to ``artifacts/``).
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, settings


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    return scaler


def save_scaler(scaler: StandardScaler) -> None:
    settings.scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, settings.scaler_path)


def load_scaler() -> StandardScaler:
    return joblib.load(settings.scaler_path)


def transform(scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    return scaler.transform(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))


def vectorize_one(scaler: StandardScaler, record: dict) -> np.ndarray:
    """Turn a single transaction dict into a scaled (1, n_features) array."""
    row = np.array([[float(record[c]) for c in FEATURE_COLUMNS]], dtype=np.float64)
    return scaler.transform(row)

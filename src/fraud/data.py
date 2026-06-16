"""Data loading.

Uses the real Kaggle ``creditcard.csv`` when present in ``data/``; otherwise
generates a synthetic dataset that mirrors the same schema (Time, V1..V28,
Amount, Class) and class imbalance so the whole pipeline runs end-to-end
without credentials. The synthetic frauds are *partially* separable from the
normal class so reported metrics are realistic rather than perfect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, LABEL_COLUMN, V_COLUMNS, settings


def generate_synthetic(n_rows: int = 60_000, fraud_rate: float = 0.0017,
                       seed: int = 42) -> pd.DataFrame:
    """Synthetic data shaped like the Kaggle credit-card dataset.

    Normal transactions: V-features ~ N(0, 1). Frauds: a subset of V-features is
    shifted by a moderate margin with extra variance, plus heavier-tailed
    amounts. The overlap is deliberate so detectors don't hit a trivial 1.0.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_normal = n_rows - n_fraud

    # V1..V28 — PCA-like standardized components in the real dataset.
    normal_v = rng.standard_normal((n_normal, len(V_COLUMNS)))

    fraud_v = rng.standard_normal((n_fraud, len(V_COLUMNS)))
    # Shift ~8 of the components to create a detectable-but-overlapping signal.
    shifted = rng.choice(len(V_COLUMNS), size=8, replace=False)
    fraud_v[:, shifted] += rng.normal(2.2, 1.0, size=(n_fraud, len(shifted)))
    fraud_v *= 1.3  # slightly fatter spread

    v = np.vstack([normal_v, fraud_v])
    labels = np.concatenate([np.zeros(n_normal, int), np.ones(n_fraud, int)])

    # Time: monotonically increasing seconds over ~2 days, like the real set.
    time = np.sort(rng.uniform(0, 172_800, size=n_rows))

    # Amount: log-normal; frauds skew to a different (often smaller) profile.
    amount = np.empty(n_rows)
    amount[:n_normal] = rng.lognormal(mean=3.0, sigma=1.1, size=n_normal)
    amount[n_normal:] = rng.lognormal(mean=2.3, sigma=1.4, size=n_fraud)

    df = pd.DataFrame(v, columns=V_COLUMNS)
    df.insert(0, "Time", time - time.min())
    df["Amount"] = np.round(amount, 2)
    df[LABEL_COLUMN] = labels

    # Shuffle so frauds aren't all at the tail.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Return (dataframe, is_real). Falls back to synthetic if no CSV present."""
    path = settings.dataset_path
    if path.exists():
        df = pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        if verbose:
            print(f"Loaded REAL dataset: {path} ({len(df):,} rows, "
                  f"{int(df[LABEL_COLUMN].sum()):,} frauds)")
        return df, True

    df = generate_synthetic(fraud_rate=settings.contamination)
    if verbose:
        print(f"No {path.name} found — using SYNTHETIC dataset "
              f"({len(df):,} rows, {int(df[LABEL_COLUMN].sum()):,} frauds). "
              f"Drop the real Kaggle CSV at {path} to train on it.")
    return df, False


def load_split(test_size: float = 0.3, seed: int = 42
               ) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Stratified train/test split shared by training and evaluation."""
    df, is_real = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[LABEL_COLUMN], random_state=seed,
    )
    return (train_df.reset_index(drop=True),
            test_df.reset_index(drop=True), is_real)

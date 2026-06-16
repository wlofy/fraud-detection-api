"""Core ML-contract tests — no Kafka, no persisted artifacts required."""

from __future__ import annotations

import numpy as np

from fraud.config import FEATURE_COLUMNS, LABEL_COLUMN
from fraud.data import generate_synthetic
from fraud.features import fit_scaler, transform
from fraud.models import AutoencoderModel, IsolationForestModel


def _xy():
    df = generate_synthetic(n_rows=8000, fraud_rate=0.02, seed=1)
    scaler = fit_scaler(df)
    X = transform(scaler, df)
    y = df[LABEL_COLUMN].to_numpy()
    return X, y


def test_synthetic_schema():
    df = generate_synthetic(n_rows=2000, fraud_rate=0.02, seed=0)
    for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
        assert col in df.columns
    assert df[LABEL_COLUMN].sum() > 0
    assert set(df[LABEL_COLUMN].unique()) <= {0, 1}


def test_risk_scores_in_unit_range():
    X, y = _xy()
    model = IsolationForestModel(n_estimators=50).fit(X, y)
    risk = model.risk_scores(X)
    assert risk.min() >= 0.0 and risk.max() <= 1.0


def test_isolation_forest_separates_fraud():
    X, y = _xy()
    model = IsolationForestModel(n_estimators=100).fit(X, y)
    risk = model.risk_scores(X)
    assert risk[y == 1].mean() > risk[y == 0].mean()


def test_autoencoder_separates_fraud():
    X, y = _xy()
    model = AutoencoderModel(epochs=8).fit(X, y)
    risk = model.risk_scores(X)
    assert risk[y == 1].mean() > risk[y == 0].mean()


def test_save_load_roundtrip(tmp_path):
    X, y = _xy()
    model = IsolationForestModel(n_estimators=50).fit(X, y)
    path = tmp_path / "iso.joblib"
    model.save(path)
    reloaded = IsolationForestModel.load(path)
    assert np.allclose(model.risk_scores(X), reloaded.risk_scores(X))

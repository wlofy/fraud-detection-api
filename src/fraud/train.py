"""Train both anomaly detectors and persist artifacts.

    python -m fraud.train

Produces in ``artifacts/``:
  * scaler.joblib              — fitted StandardScaler
  * isolation_forest.joblib    — fitted Isolation Forest
  * autoencoder.pt             — trained autoencoder
  * metadata.json              — feature list, thresholds, dataset provenance
  * metrics.json               — held-out ablation results (via evaluate)
"""

from __future__ import annotations

import json
import time

import numpy as np

from .config import FEATURE_COLUMNS, settings
from .data import load_split
from .evaluate import evaluate_all, print_table
from .features import fit_scaler, save_scaler, transform
from .models import AutoencoderModel, IsolationForestModel


def main() -> None:
    t0 = time.time()
    train_df, _, is_real = load_split()
    y_train = train_df["Class"].to_numpy()

    scaler = fit_scaler(train_df)
    save_scaler(scaler)
    X_train = transform(scaler, train_df)
    print(f"Train rows: {len(X_train):,}  frauds: {int(y_train.sum())}  "
          f"features: {X_train.shape[1]}")

    builders = {
        "isolation_forest": IsolationForestModel(
            n_estimators=200, contamination=settings.contamination),
        "autoencoder": AutoencoderModel(epochs=20),
    }

    thresholds: dict[str, float] = {}
    for name, model in builders.items():
        print(f"\nTraining {name} …")
        ts = time.time()
        model.fit(X_train, y_train)
        # Decision threshold: the risk quantile matching expected contamination.
        train_risk = model.risk_scores(X_train)
        thr = float(np.quantile(train_risk, 1.0 - settings.contamination))
        thresholds[name] = thr
        model.save(settings.model_path(name))
        print(f"  done in {time.time() - ts:.1f}s  threshold={thr:.4f}  "
              f"-> {settings.model_path(name).name}")

    metadata = {
        "version": 1,
        "dataset": "real" if is_real else "synthetic",
        "feature_columns": FEATURE_COLUMNS,
        "n_features": int(X_train.shape[1]),
        "contamination": settings.contamination,
        "thresholds": thresholds,
        "models": list(builders.keys()),
    }
    settings.metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote {settings.metadata_path.name}")

    # Held-out evaluation + ablation table.
    print_table(evaluate_all())
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

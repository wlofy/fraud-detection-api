"""Evaluation + ablation.

Computes ranking metrics (PR-AUC, ROC-AUC, precision@k) that are appropriate
for extreme class imbalance, plus operating-point metrics at the persisted
decision threshold. Run standalone after training:

    python -m fraud.evaluate
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)

from .config import settings
from .data import load_split
from .features import load_scaler, transform
from .models import REGISTRY


def evaluate_model(model, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    risk = model.risk_scores(X)
    n_pos = int(y.sum())

    # Ranking metrics — threshold-free, robust to imbalance.
    pr_auc = float(average_precision_score(y, risk))
    roc_auc = float(roc_auc_score(y, risk))

    # precision@k with k = number of true frauds (== recall@k here).
    order = np.argsort(-risk)
    topk = order[:n_pos] if n_pos else order[:0]
    precision_at_k = float(y[topk].mean()) if n_pos else 0.0

    # Operating point at the deployed threshold.
    pred = (risk >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_k": precision_at_k,
        "threshold": float(threshold),
        "precision_at_threshold": float(p),
        "recall_at_threshold": float(r),
        "f1_at_threshold": float(f1),
        "flagged": int(pred.sum()),
    }


def evaluate_all() -> dict:
    _, test_df, is_real = load_split()
    scaler = load_scaler()
    X = transform(scaler, test_df)
    y = test_df["Class"].to_numpy()

    meta = json.loads(settings.metadata_path.read_text())
    thresholds = meta["thresholds"]

    results = {}
    for name, cls in REGISTRY.items():
        model = cls.load(settings.model_path(name))
        results[name] = evaluate_model(model, X, y, thresholds[name])

    report = {
        "dataset": "real" if is_real else "synthetic",
        "test_rows": int(len(y)),
        "test_frauds": int(y.sum()),
        "models": results,
    }
    settings.metrics_path.write_text(json.dumps(report, indent=2))
    return report


def print_table(report: dict) -> None:
    print(f"\nAblation — {report['dataset']} dataset "
          f"({report['test_rows']:,} test rows, {report['test_frauds']} frauds)\n")
    header = (f"{'model':<18}{'PR-AUC':>9}{'ROC-AUC':>9}{'P@k':>8}"
              f"{'Prec':>8}{'Recall':>8}{'F1':>8}{'flagged':>9}")
    print(header)
    print("-" * len(header))
    for name, m in report["models"].items():
        print(f"{name:<18}{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}"
              f"{m['precision_at_k']:>8.3f}{m['precision_at_threshold']:>8.3f}"
              f"{m['recall_at_threshold']:>8.3f}{m['f1_at_threshold']:>8.3f}"
              f"{m['flagged']:>9}")
    print()


if __name__ == "__main__":
    print_table(evaluate_all())

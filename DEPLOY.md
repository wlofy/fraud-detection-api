# Deploy — free hosting

Single Render web service: FastAPI serves the scoring endpoints **and** the
demo page at `/`. Kafka, matplotlib, and pytest are kept out of the deploy
image (still available for local dev via `requirements.txt`).

## Quick deploy (Render)

1. Render dashboard → **New +** → **Blueprint** → connect this repo. It picks
   up `render.yaml` and `Dockerfile`.
2. First build takes ~5 minutes (downloads CPU-only torch + sklearn).
3. First request triggers synthetic-data training (~30s). The demo page polls
   `/health` until the worker is warm.

## What gets deployed

| Path | What |
|---|---|
| `/` | Demo UI — buttons to score normal vs. fraud-like synthetic transactions |
| `/health` | Liveness — also reports which model + dataset is loaded |
| `/score` (POST) | Score a real txn (Time, V1..V28, Amount) |
| `/score/sample` (GET) | Generate + score a synthetic txn — `?label=normal\|fraud&seed=N&model=isolation_forest\|autoencoder` |
| `/metrics` | Held-out ablation: PR-AUC, ROC-AUC, precision@k, F1 |
| `/docs` | Swagger UI |

## Tradeoffs accepted for the free tier

- **Synthetic data only.** No Kaggle CSV is shipped. The schema and class
  imbalance match — metrics are realistic but not headline-grabbing.
- **Free instance sleeps after 15 min idle** → ~30s cold start.
- **Kafka pipeline excluded from the image** to keep build time and image size
  down. `docker-compose.yml` still runs the full stack locally.
- **Torch is CPU-only** via `download.pytorch.org/whl/cpu` — drops image size
  by ~600MB.

## Local dev unchanged

```bash
pip install -r requirements.txt
python -m fraud.train
uvicorn fraud.api:app --reload
# open http://localhost:8000
```

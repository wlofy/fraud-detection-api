# Fraud Detection API

A Stripe-Radar-style service that scores card transactions for fraud risk **in
real time**. Two unsupervised anomaly detectors — an **Isolation Forest** and a
**PyTorch autoencoder** — are trained on the
[Kaggle credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
benchmarked against each other (PR-AUC / precision@k), served over a **FastAPI**
endpoint, and fed a live stream replayed through **Kafka**.

```
┌──────────────┐   JSON txns    ┌─────────┐   consume    ┌──────────────┐
│  producer.py │ ─────────────► │  Kafka  │ ───────────► │  consumer.py │
│ (replay test │   topic:       │ (KRaft) │              │  Scorer →    │
│   split)     │  transactions  └─────────┘              │  risk + flag │
└──────────────┘                                         └──────────────┘
                         FastAPI  /score  ◄── same Scorer (request/response)
```

> **No Kaggle credentials?** Everything still runs. If `data/creditcard.csv` is
> absent, the pipeline trains on a **synthetic dataset with the identical schema**
> (`Time, V1..V28, Amount, Class`) and matching class imbalance. Drop the real
> CSV in `data/` to switch over — no code changes.

## Why these choices

- **Isolation Forest** — fast, no scaling assumptions, the standard baseline for
  tabular anomaly detection. Fully unsupervised.
- **Autoencoder** — trained to reconstruct *legitimate* transactions only; fraud
  reconstructs poorly, so reconstruction MSE is the anomaly score
  (semi-supervised). Shows whether a learned manifold beats tree isolation.
- **PR-AUC & precision@k over accuracy** — frauds are ~0.17% of traffic, so
  accuracy is meaningless. Ranking metrics under extreme imbalance are what
  matter, and they're what the ablation reports.

## Results

Held-out ablation on the **synthetic fallback** (60k rows, 0.17% fraud; swap in
the real Kaggle CSV for headline numbers). Ranking metrics, since accuracy is
meaningless at this imbalance:

| model              | PR-AUC | ROC-AUC | precision@k | precision\* | recall\* | F1\* |
|--------------------|:------:|:-------:|:-----------:|:-----------:|:--------:|:----:|
| **isolation_forest** | **0.934** | 0.9999 | **0.871** | 0.794 | 0.871 | 0.831 |
| autoencoder        | 0.488  | 0.9990  | 0.548       | 0.514       | 0.581    | 0.545 |

\* at the deployed decision threshold (risk quantile matching the contamination
rate). Reproduce with `python -m fraud.train`.

**Live stream** (6k transactions replayed through Kafka, scored by the consumer):
`precision ≈ 0.73, recall = 1.00` — every planted fraud caught, a handful of
false positives, with a rolling readout printed as messages arrive.

> On this synthetic data the Isolation Forest wins decisively. The autoencoder is
> deliberately small (20 epochs, 8-unit bottleneck); it tends to close the gap on
> the real dataset where the V-features carry richer manifold structure.

## Quick start

```powershell
# 1. Install
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. (optional) real dataset — needs Kaggle API token
python scripts/download_data.py

# 3. Train both models + print the ablation table
python -m fraud.train

# 4. Serve the scoring API
uvicorn fraud.api:app --reload      # http://localhost:8000/docs
```

### Real-time stream (Kafka)

```powershell
# Start a single-node Kafka broker (KRaft, no Zookeeper)
docker compose up -d

# Terminal A — score the stream as it arrives
python -m fraud.consumer                 # add --model autoencoder to compare

# Terminal B — replay the held-out transactions
python -m fraud.producer --rate 200
```

The consumer prints every flagged transaction plus a rolling precision/recall
readout (the producer ships the true label alongside each message for live
scoring feedback).

### Scoring one transaction over HTTP

```powershell
$body = @{
  time   = 40000
  amount = 149.62
  v      = @(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/score -Method Post -Body $body -ContentType application/json
```

```json
{ "model": "isolation_forest", "risk_score": 0.026421, "threshold": 0.97612, "is_fraud": false }
```

Switch model per request: `POST /score?model=autoencoder`.

## Endpoints

| Method | Path       | Purpose                                       |
|--------|------------|-----------------------------------------------|
| GET    | `/health`  | Liveness + which model/dataset is loaded      |
| POST   | `/score`   | Score one transaction (`?model=` to override) |
| GET    | `/metrics` | Last held-out ablation results                |

## Configuration

Copy `.env.example` → `.env`. Key knobs: `KAFKA_BOOTSTRAP_SERVERS`,
`KAFKA_TOPIC`, `SCORING_MODEL` (`isolation_forest` | `autoencoder`),
`PRODUCER_RATE`, `CONTAMINATION` (expected fraud fraction → decision threshold).

## Project layout

```
src/fraud/
  config.py        settings + paths + the Kaggle schema
  data.py          load real CSV or synthetic fallback; stratified split
  features.py      StandardScaler fit/transform (shared train + serve)
  models/          base interface, isolation_forest.py, autoencoder.py
  train.py         fit both, calibrate thresholds, save artifacts
  evaluate.py      PR-AUC / ROC-AUC / precision@k ablation
  scoring.py       warm Scorer used by API + consumer
  api.py           FastAPI app
  producer.py      replay transactions → Kafka
  consumer.py      consume → score → flag, with live precision/recall
scripts/download_data.py
tests/test_scoring.py
docker-compose.yml
```

## Tests

```powershell
pytest -q
```

Covers the schema, unit-range risk scores, that both models rank frauds above
normals, and a save/load round-trip — all without needing Kafka or artifacts.

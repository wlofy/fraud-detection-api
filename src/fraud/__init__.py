"""Real-time fraud-detection scoring service.

A Stripe-Radar-style pipeline: train unsupervised anomaly detectors on the
Kaggle credit-card dataset (or a synthetic fallback), serve risk scores over a
FastAPI endpoint, and replay transactions through Kafka for a real-time stream.
"""

__version__ = "0.1.0"

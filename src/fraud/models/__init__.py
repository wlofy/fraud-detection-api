from .base import AnomalyModel
from .isolation_forest import IsolationForestModel
from .autoencoder import AutoencoderModel

REGISTRY = {
    "isolation_forest": IsolationForestModel,
    "autoencoder": AutoencoderModel,
}

__all__ = ["AnomalyModel", "IsolationForestModel", "AutoencoderModel", "REGISTRY"]

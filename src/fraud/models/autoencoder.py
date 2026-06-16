"""Autoencoder anomaly detector (PyTorch).

Semi-supervised: the network is trained to reconstruct *normal* transactions
only. Frauds, being out of distribution, reconstruct poorly, so the
reconstruction MSE is the anomaly score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import AnomalyModel


class _AE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16, bottleneck: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderModel(AnomalyModel):
    name = "autoencoder"

    def __init__(self, n_features: int | None = None, hidden: int = 16,
                 bottleneck: int = 8, epochs: int = 20, batch_size: int = 512,
                 lr: float = 1e-3, seed: int = 42) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.net: _AE | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoencoderModel":
        torch.manual_seed(self.seed)
        # Train on normal rows only when labels are available.
        X_train = X[y == 0] if y is not None else X
        self.n_features = X.shape[1]
        self.net = _AE(self.n_features, self.hidden, self.bottleneck)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        data = torch.tensor(X_train, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=self.batch_size,
            shuffle=True,
        )

        self.net.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                opt.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                opt.step()

        self._calibrate(X)
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None, "model not fitted/loaded"
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            recon = self.net(x)
            mse = torch.mean((recon - x) ** 2, dim=1)
        return mse.numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "n_features": self.n_features,
            "hidden": self.hidden,
            "bottleneck": self.bottleneck,
            "mean": self._mean,
            "std": self._std,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_features=blob["n_features"], hidden=blob["hidden"],
                  bottleneck=blob["bottleneck"])
        obj.net = _AE(blob["n_features"], blob["hidden"], blob["bottleneck"])
        obj.net.load_state_dict(blob["state_dict"])
        obj._mean = blob["mean"]
        obj._std = blob["std"]
        return obj

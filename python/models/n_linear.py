from __future__ import annotations

import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class _LinearModel(pl.LightningModule):
    """Minimal linear regression model using PyTorch Lightning."""

    def __init__(self, input_dim: int, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return self.linear(x).squeeze(-1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    """Train a simple linear model using PyTorch Lightning."""
    if params is None:
        params = {}
    lr = float(params.get("lr", 1e-3))
    epochs = int(params.get("epochs", 10))
    batch_size = int(params.get("batch_size", 32))

    model = _LinearModel(X.shape[1], lr)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_model_summary=False)
    trainer.fit(model, loader)
    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Generate predictions using the trained Lightning model."""
    net: _LinearModel = model["model"]
    net.eval()
    with torch.no_grad():
        preds = net(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    return preds

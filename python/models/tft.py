"""Temporal Fusion Transformer implemented via PyTorch Lightning."""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class _TFT(pl.LightningModule):
    def __init__(self, input_dim: int, hidden: int, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = torch.nn.Linear(input_dim, hidden)
        encoder_layer = torch.nn.TransformerEncoderLayer(hidden, nhead=2)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        x = self.encoder(x)
        x = self.transformer(x)
        out = self.fc(x.mean(dim=1))
        return out.squeeze(-1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss

    def configure_optimizers(self):  # pragma: no cover - simple
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> dict:
    if params is None:
        params = {}

    lr = float(params.get("lr", 1e-3))
    hidden = int(params.get("hidden", 32))
    epochs = int(params.get("epochs", 5))
    batch_size = int(params.get("batch_size", 32))

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _TFT(X.shape[1], hidden, lr)
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_model_summary=False)
    trainer.fit(model, loader)
    return {"model": model}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    net: _TFT = model["model"]
    net.eval()
    with torch.no_grad():
        preds = net(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).cpu().numpy()
    return preds.squeeze()



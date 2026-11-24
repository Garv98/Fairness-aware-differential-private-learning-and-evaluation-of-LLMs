"""Model definitions. ``MLP`` with default hidden=(64, 32) is identical to the
original ``src/models.py::CreditRiskMLP`` so earlier results stay comparable."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    out = [torch.sigmoid(model(Xt[i:i + batch])) for i in range(0, len(Xt), batch)]
    return torch.cat(out).numpy()


@torch.no_grad()
def per_sample_loss(model: nn.Module, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32))
    return nn.functional.binary_cross_entropy_with_logits(
        logits, torch.as_tensor(y, dtype=torch.float32), reduction="none").numpy()

"""Non-private reference trainers: ERM, class-weighted ERM, exact Group DRO
(Sagawa et al., 2020) over intersectional cells."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NonPrivateConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    method: str = "erm"            # erm | class_weighted | groupdro
    dro_eta: float = 0.01
    verbose: bool = False


def train_nonprivate(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                     cell_ids: np.ndarray, n_cells: int, cfg: NonPrivateConfig, callback=None):
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32),
                       torch.as_tensor(cell_ids, dtype=torch.long))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    pos_weight = None
    if cfg.method == "class_weighted":
        counts = np.bincount(y.astype(int), minlength=2)
        pos_weight = torch.tensor(counts[0] / counts[1], dtype=torch.float32)
    q = torch.ones(n_cells) / n_cells

    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        for xb, yb, cb in loader:
            opt.zero_grad()
            logits = model(xb)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none", pos_weight=pos_weight)
            if cfg.method == "groupdro":
                cell_loss = torch.zeros(n_cells).index_add_(0, cb, losses)
                cnt = torch.zeros(n_cells).index_add_(0, cb, torch.ones_like(losses))
                present = cnt > 0
                mean_loss = torch.where(present, cell_loss / cnt.clamp_min(1), torch.zeros_like(cell_loss))
                with torch.no_grad():
                    q = q * torch.exp(cfg.dro_eta * mean_loss.detach() * present)
                    q = q / q.sum()
                loss = (q * mean_loss).sum()
            else:
                loss = losses.mean()
            loss.backward()
            opt.step()
            tot += float(loss)
        if cfg.verbose:
            print(f"epoch {ep + 1}: loss {tot / len(loader):.4f}")
        if callback is not None:
            callback(ep + 1, model, {"epoch": ep + 1, "loss": tot / len(loader)})
    return model

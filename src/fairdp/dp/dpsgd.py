"""Unified DP-SGD trainer covering the baselines and IG-DPSGD.

One loop, switchable components (all under a single fixed (eps, delta), see
:mod:`fairdp.dp.accounting`):

  clip_mode
    "flat"          standard DP-SGD (Abadi et al., 2016)
    "dpsgdf"        DPSGD-F-style group-dependent clipping bounds (after
                    Xu et al., 2021): C_c = C0 * (1 + f_c / max f), f_c =
                    private clipped fraction of cell c; noise scales with max C.
    "global_adapt"  DPSGD-Global-Adapt-style scaling (after Esipova et al.,
                    2023): g -> (C/Z) g if ||g|| <= Z else C g/||g||, with Z
                    tracked privately at quantile ``gamma`` of gradient norms.
  dro               private intersectional Group-DRO reweighting
                    (exponentiated-gradient on noisy per-cell losses).
  smoothing         hierarchical shrinkage of noisy per-cell estimates toward
                    their marginal parents (marginals derived from cell stats,
                    i.e. post-processing).

Per-sample gradients are computed with ``torch.func`` (vmap + grad).
Poisson subsampling is used so the accountant's amplification applies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from torch.func import functional_call, grad_and_value, vmap

from .accounting import NoisePlan, plan_noise, default_delta
from ..groups import CellIndex


@dataclass
class DPConfig:
    target_epsilon: float = 3.0
    delta: float | None = None             # default 1/n^1.1
    epochs: int = 20
    expected_batch_size: int = 256
    lr: float = 1e-3
    optimizer: str = "adam"
    max_grad_norm: float = 1.0
    accountant: str = "rdp"
    seed: int = 0
    # components
    clip_mode: str = "flat"                # flat | dpsgdf | global_adapt
    dro: bool = False
    balance_labels: bool = False           # baseline: private label re-balancing (no groups)
    smoothing: bool = False
    aux_frac: float = 0.0                  # share of privacy-loss for stats
    # hyper-parameters of components
    loss_bound: float = 4.0                # per-sample loss clipped to [0, B]
    dro_eta: float = 0.05
    dro_warmup_frac: float = 0.1
    weight_cap: float = 3.0                # kappa: r_c in [1/kappa, kappa]
    dro_label_cells: bool = True           # DRO cells = intersection x label (Sagawa et al.)
    smoothing_strength: float = 4.0        # m (pseudo-count, per-batch units)
    ema: float = 0.95
    clip_quantile: float = 0.9             # gamma (global_adapt)
    clip_lr: float = 0.2                   # eta_Z (geometric update)
    verbose: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def needs_stats(self) -> bool:
        return self.dro or self.balance_labels or self.clip_mode in ("dpsgdf", "global_adapt")

    @property
    def uses_label_cells(self) -> bool:
        return self.balance_labels or (self.dro and self.dro_label_cells)

    def stats_blocks(self) -> int:
        k = 0
        if self.needs_stats:
            k += 1                          # counts
        if self.dro:
            k += 1                          # clipped losses
        if self.clip_mode in ("dpsgdf", "global_adapt"):
            k += 1                          # indicator ||g|| <= threshold
        return k


@dataclass
class TrainLog:
    noise: NoisePlan
    config: dict
    history: list = field(default_factory=list)   # per-epoch diagnostics
    final_cell_weights: list | None = None
    final_clip: dict | None = None


def _bce(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, y)


class _NoisyCellStats:
    """EMA of noisy per-batch cell statistics + hierarchical smoothing."""

    def __init__(self, cells: CellIndex, ema: float, strength: float, smoothing: bool):
        self.cells, self.ema, self.m, self.smoothing = cells, ema, strength, smoothing
        self.v: dict[str, np.ndarray] = {}

    def update(self, name: str, noisy: np.ndarray):
        if name not in self.v:
            self.v[name] = noisy.copy()
        else:
            self.v[name] = self.ema * self.v[name] + (1 - self.ema) * noisy

    def ratio(self, num: str, den: str = "count", prior: float | None = None) -> np.ndarray:
        """Estimate per-cell mean num/den (e.g. mean loss, fraction below)."""
        S = self.v[num]
        N = np.maximum(self.v[den], 0.0)
        if not self.smoothing:
            return np.clip(S / np.maximum(N, 1e-3), 0.0, 1.0)
        tot_S, tot_N = S.sum(), max(N.sum(), 1e-3)
        glob = float(np.clip(tot_S / tot_N, 0, 1)) if prior is None else prior
        parents = []
        for a, M in self.cells.marginal_membership.items():
            ms, mn = M @ S, np.maximum(M @ N, 0.0)
            # marginal estimate shrunk toward global
            marg = (ms + self.m * glob) / (mn + self.m)
            parents.append(M.T @ marg)          # map marginal level -> cells
        parent = np.clip(np.mean(parents, axis=0), 0, 1)
        return np.clip((S + self.m * parent) / (N + self.m), 0.0, 1.0)

    def freq(self) -> np.ndarray:
        N = np.maximum(self.v["count"], 0.0)
        if self.smoothing:
            N = N + 0.1                          # avoid zero mass
        p = N / max(N.sum(), 1e-6)
        return np.maximum(p, 1e-4)


def train_dp(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
             cell_ids: np.ndarray, cells: CellIndex, cfg: DPConfig,
             callback=None) -> tuple[torch.nn.Module, TrainLog]:
    """``callback(epoch, model, record)`` is called after every epoch with the
    model holding the current parameters (post-processing; no privacy cost)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n = len(y)
    q = cfg.expected_batch_size / n
    steps_per_epoch = int(round(1 / q))
    T = cfg.epochs * steps_per_epoch
    delta = cfg.delta or default_delta(n)
    k = cfg.stats_blocks()
    noise = plan_noise(cfg.target_epsilon, delta, q, T, aux_frac=cfg.aux_frac if k else 0.0,
                       stats_blocks=k, accountant=cfg.accountant)
    log = TrainLog(noise=noise, config=asdict(cfg))

    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    cid = np.asarray(cell_ids)
    if (cid < 0).any():
        raise ValueError("training rows with unknown cell id")
    C = cells.n_cells

    params = {k_: v.detach().clone().requires_grad_(False) for k_, v in model.named_parameters()}
    names = list(params)

    def loss_fn(p, x, t):
        out = functional_call(model, p, (x.unsqueeze(0),)).squeeze()
        return _bce(out, t)

    per_sample = vmap(grad_and_value(loss_fn), in_dims=(None, 0, 0))

    # optimizer on plain tensors
    opt_params = [torch.nn.Parameter(params[k_].clone()) for k_ in names]
    opt = (torch.optim.Adam(opt_params, lr=cfg.lr) if cfg.optimizer == "adam"
           else torch.optim.SGD(opt_params, lr=cfg.lr, momentum=0.9))

    stats = _NoisyCellStats(cells, cfg.ema, cfg.smoothing_strength, cfg.smoothing)
    log_r = np.zeros(C)                     # DRO log importance ratio q_c / p_c (starts at ERM)
    cell_w = np.ones(C)                      # per-sample weights in (0,1]
    w_scale = 1.0
    C0 = cfg.max_grad_norm
    clip_c = np.full(C, C0)                  # per-cell clipping (dpsgdf)
    Z = C0                                   # global_adapt threshold
    warmup = int(cfg.dro_warmup_frac * T)

    for step in range(T):
        mask = rng.random(n) < q
        idx = np.nonzero(mask)[0]
        cur = {k_: p.detach() for k_, p in zip(names, opt_params)}
        if len(idx):
            xb, yb, cb = Xt[idx], yt[idx], cid[idx]
            grads, losses = per_sample(cur, xb, yb)
            flat = torch.cat([grads[k_].reshape(len(idx), -1) for k_ in names], dim=1)
            norms = flat.norm(dim=1).clamp_min(1e-12)
            norms_np = norms.numpy()

            # ---- clipping / scaling --------------------------------------
            if cfg.clip_mode == "flat":
                factor = (C0 / norms).clamp(max=1.0)
                sens = C0
                below = None
            elif cfg.clip_mode == "dpsgdf":
                cc = torch.as_tensor(clip_c[cb], dtype=torch.float32)
                factor = (cc / norms).clamp(max=1.0)
                sens = float(clip_c.max())
                below = (norms_np <= C0).astype(float)      # clipped-fraction stat at C0
            elif cfg.clip_mode == "global_adapt":
                factor = torch.where(norms <= Z, torch.full_like(norms, C0 / Z), C0 / norms)
                sens = C0
                below = (norms_np <= Z).astype(float)
            else:
                raise ValueError(cfg.clip_mode)

            w = torch.as_tensor(cell_w[cb], dtype=torch.float32)
            summed = (flat * (factor * w).unsqueeze(1)).sum(0)
        else:
            summed = torch.zeros(sum(p.numel() for p in opt_params))
            sens = C0 if cfg.clip_mode != "dpsgdf" else float(clip_c.max())
            cb, losses, below = np.array([], dtype=int), torch.zeros(0), None

        noisy = summed + torch.randn_like(summed) * noise.sigma_grad * sens
        g = noisy / (q * n) * w_scale

        # ---- optimizer step ---------------------------------------------
        offset = 0
        for p in opt_params:
            num = p.numel()
            p.grad = g[offset:offset + num].view_as(p).clone()
            offset += num
        opt.step()

        # ---- auxiliary private statistics (same batch, accounted jointly)
        if k:
            ss = noise.sigma_stats
            cnt = np.bincount(cb, minlength=C).astype(float)
            stats.update("count", cnt + rng.normal(0, ss, C))
            if cfg.dro:
                lb = np.clip(losses.detach().numpy() / cfg.loss_bound, 0, 1) if len(cb) else np.zeros(0)
                stats.update("loss", np.bincount(cb, weights=lb, minlength=C) + rng.normal(0, ss, C))
            if below is not None or cfg.clip_mode in ("dpsgdf", "global_adapt"):
                bl = below if below is not None else np.zeros(0)
                stats.update("below", np.bincount(cb, weights=bl, minlength=C) + rng.normal(0, ss, C))

            # ---- post-processing of noisy statistics --------------------
            if cfg.clip_mode == "global_adapt":
                frac_below = float(np.clip(stats.v["below"].sum() / max(stats.v["count"].sum(), 1e-3), 0, 1))
                Z = float(np.clip(Z * math.exp(-cfg.clip_lr * (frac_below - cfg.clip_quantile)), 1e-3, 1e3))
            if cfg.clip_mode == "dpsgdf":
                f_clip = 1.0 - stats.ratio("below")
                clip_c = C0 * (1.0 + f_clip / max(f_clip.max(), 1e-6))
            if cfg.balance_labels:
                if "_y" not in cells.marginal_membership:
                    raise ValueError("balance_labels requires label-augmented cells")
                My = cells.marginal_membership["_y"]           # [2, C]
                p_y = np.maximum(My @ stats.freq(), 1e-3)
                inv = My.T @ (1.0 / p_y)                         # per-cell 1/p(y)
                cell_w = inv / inv.max()
                w_scale = 1.0 / float(np.sum(stats.freq() * cell_w))
            if cfg.dro and step >= warmup:
                # Exponentiated-gradient Group DRO, parametrised by the
                # importance ratio r_c = q_c / p_c and started at ERM (r = 1).
                # Updating log r relative to the current weighted mean loss
                # and capping |log r| <= log(kappa) keeps the weighted
                # objective close to ERM; normalising by max r keeps every
                # per-sample weight <= 1 (sensitivity unchanged) and never
                # downweights the whole batch (which would amplify noise).
                L = stats.ratio("loss")
                p_hat = stats.freq()
                r = np.exp(log_r)
                q_cur = p_hat * r / np.sum(p_hat * r)
                log_r = log_r + cfg.dro_eta * (L - float(np.sum(q_cur * L)))
                log_r = np.clip(log_r, -math.log(cfg.weight_cap), math.log(cfg.weight_cap))
                r = np.exp(log_r)
                cell_w = r / r.max()                            # in (0, 1]
                w_scale = 1.0 / float(np.sum(p_hat * cell_w))   # keep effective lr

        if (step + 1) % steps_per_epoch == 0:
            ep = (step + 1) // steps_per_epoch
            rec = {"epoch": ep, "Z": Z, "w_scale": w_scale}
            if cfg.dro and "loss" in stats.v:
                rec["cell_loss_max"] = float(stats.ratio("loss").max())
            log.history.append(rec)
            if cfg.verbose:
                print(rec)
            if callback is not None:
                with torch.no_grad():
                    for (_, p_model), p in zip(model.named_parameters(), opt_params):
                        p_model.copy_(p.detach())
                callback(ep, model, rec)

    with torch.no_grad():
        for (k_, p_model), p in zip(model.named_parameters(), opt_params):
            p_model.copy_(p.detach())
    log.final_cell_weights = cell_w.tolist()
    log.final_clip = {"Z": Z, "clip_c": clip_c.tolist()}
    return model, log

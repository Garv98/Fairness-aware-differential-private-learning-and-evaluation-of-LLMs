"""Privacy accounting for DP-SGD with auxiliary group-statistics releases.

At every step, on the same Poisson-subsampled batch, the trainer releases

  * the noisy clipped gradient sum      sum_i clip(g_i) + N(0, (sigma_g * C)^2 I)
  * (optional) a noisy statistics vector sum_i s_i      + N(0, sigma_s^2 I)

where each per-sample statistics vector s_i is made of ``k`` one-hot blocks
with entries in [0, 1] (so ||s_i||_2 <= sqrt(k)). After whitening each block
by its noise scale, the pair is ONE Gaussian mechanism on the subsample with
unit noise and L2 sensitivity

      Delta^2 = 1/sigma_g^2 + k/sigma_s^2  =:  1/sigma_eff^2 .

So the joint per-step mechanism is exactly a subsampled Gaussian mechanism
with noise multiplier ``sigma_eff`` and the standard accountant applies with
no loose composition. A *budget split* rho in [0, 1) allocates the
"privacy-loss mass" 1/sigma_eff^2 between gradients (1-rho) and statistics
(rho):  sigma_g = sigma_eff / sqrt(1-rho),  sigma_s = sigma_eff * sqrt(k/rho).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from opacus.accountants import RDPAccountant, PRVAccountant
from opacus.accountants.utils import get_noise_multiplier


def make_accountant(kind: str = "rdp"):
    return {"rdp": RDPAccountant, "prv": PRVAccountant}[kind]()


def epsilon_for(sigma: float, sample_rate: float, steps: int, delta: float,
                accountant: str = "rdp") -> float:
    acc = make_accountant(accountant)
    acc.history = [(sigma, sample_rate, steps)]
    return float(acc.get_epsilon(delta=delta))


def calibrate_sigma(target_epsilon: float, delta: float, sample_rate: float,
                    steps: int, accountant: str = "rdp") -> float:
    """Smallest noise multiplier achieving ``target_epsilon`` at ``delta``."""
    return float(get_noise_multiplier(target_epsilon=target_epsilon,
                                      target_delta=delta,
                                      sample_rate=sample_rate, steps=steps,
                                      accountant=accountant,
                                      epsilon_tolerance=0.005))


@dataclass(frozen=True)
class NoisePlan:
    sigma_eff: float      # noise multiplier of the joint mechanism (accounted)
    sigma_grad: float     # gradient noise multiplier (x clipping sensitivity)
    sigma_stats: float    # absolute std of noise added to the stats vector
    aux_frac: float
    stats_blocks: int
    sample_rate: float
    steps: int
    delta: float
    epsilon: float        # accounted epsilon of the full training run


def split_budget(sigma_eff: float, aux_frac: float, stats_blocks: int) -> tuple[float, float]:
    if stats_blocks == 0 or aux_frac == 0.0:
        return sigma_eff, math.inf
    if not 0.0 < aux_frac < 1.0:
        raise ValueError("aux_frac must be in (0, 1)")
    sigma_g = sigma_eff / math.sqrt(1.0 - aux_frac)
    sigma_s = sigma_eff * math.sqrt(stats_blocks / aux_frac)
    return sigma_g, sigma_s


def plan_noise(target_epsilon: float, delta: float, sample_rate: float,
               steps: int, aux_frac: float = 0.0, stats_blocks: int = 0,
               accountant: str = "rdp") -> NoisePlan:
    if stats_blocks > 0 and aux_frac == 0.0:
        raise ValueError("statistics requested but aux_frac == 0")
    sigma_eff = calibrate_sigma(target_epsilon, delta, sample_rate, steps, accountant)
    sg, ss = split_budget(sigma_eff, aux_frac, stats_blocks)
    eps = epsilon_for(sigma_eff, sample_rate, steps, delta, accountant)
    return NoisePlan(sigma_eff, sg, ss, aux_frac, stats_blocks, sample_rate,
                     steps, delta, eps)


def default_delta(n: int) -> float:
    """delta = 1 / n^1.1 (strictly smaller than 1/n)."""
    return 1.0 / (n ** 1.1)

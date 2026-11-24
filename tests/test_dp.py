import math

import numpy as np
import pytest
import torch

opacus = pytest.importorskip("opacus")

from fairdp.dp.accounting import (calibrate_sigma, epsilon_for, plan_noise, split_budget)  # noqa: E402
from fairdp.dp.dpsgd import DPConfig, train_dp  # noqa: E402
from fairdp.data.preprocess import prepare  # noqa: E402
from fairdp.groups import CellIndex  # noqa: E402
from fairdp.models import MLP, predict_proba  # noqa: E402


def test_calibration_hits_target():
    q, T, delta = 256 / 25000, 2000, 1e-5
    for target in (1.0, 3.0, 8.0):
        s = calibrate_sigma(target, delta, q, T)
        eps = epsilon_for(s, q, T, delta)
        assert eps <= target + 1e-6 and eps > target * 0.97


def test_matches_original_opacus_setting():
    """Original run: sigma=1.0, batch 64 Poisson on 25,932 rows, 20 epochs,
    delta=1e-5 -> reported eps = 1.3304 (results/final_dp_*.csv)."""
    n = 25932
    q = 64 / n
    T = 20 * (n // 64)          # opacus loader length with drop_last semantics
    eps = epsilon_for(1.0, q, T, 1e-5)
    assert abs(eps - 1.3304) < 0.02


def test_budget_split_preserves_joint_sensitivity():
    s_eff, rho, k = 1.3, 0.1, 3
    sg, ss = split_budget(s_eff, rho, k)
    joint = 1 / sg ** 2 + k / ss ** 2
    assert math.isclose(joint, 1 / s_eff ** 2, rel_tol=1e-12)
    plan = plan_noise(3.0, 1e-5, 0.01, 1000, aux_frac=rho, stats_blocks=k)
    assert plan.sigma_grad > plan.sigma_eff and plan.epsilon <= 3.0 + 1e-6


def test_per_sample_grads_match_opacus():
    from opacus import GradSampleModule
    from torch.func import functional_call, grad, vmap
    torch.manual_seed(0)
    model = MLP(7)
    X, y = torch.randn(16, 7), torch.randint(0, 2, (16,)).float()
    gsm = GradSampleModule(MLP(7), loss_reduction="sum")
    gsm._module.load_state_dict(model.state_dict())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(gsm(X), y, reduction="sum")
    loss.backward()
    params = {k: v.detach() for k, v in model.named_parameters()}

    def f(p, x, t):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            functional_call(model, p, (x.unsqueeze(0),)).squeeze(), t)

    ours = vmap(grad(f), in_dims=(None, 0, 0))(params, X, y)
    for name, p in gsm._module.named_parameters():
        torch.testing.assert_close(ours[name], p.grad_sample, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("ov", [
    dict(clip_mode="flat"),
    dict(clip_mode="dpsgdf", aux_frac=0.05),
    dict(clip_mode="global_adapt", aux_frac=0.05),
    dict(clip_mode="global_adapt", dro=True, smoothing=True, aux_frac=0.05),
])
def test_trainer_runs_and_learns(synthetic_credit, ov):
    d = prepare(synthetic_credit, seed=0)
    cells = CellIndex(d.train.groups, d.group_attrs)
    cfg = DPConfig(target_epsilon=8.0, epochs=3, expected_batch_size=128, lr=0.005, seed=0, **ov)
    model, log = train_dp(MLP(d.train.X.shape[1]), d.train.X, d.train.y,
                          cells.encode(d.train.groups), cells, cfg)
    assert log.noise.epsilon <= 8.0 + 1e-6
    p = predict_proba(model, d.test.X)
    acc = ((p >= 0.5) == d.test.y).mean()
    assert acc >= max(d.test.y.mean(), 1 - d.test.y.mean()) - 0.05


def test_dro_weights_never_exceed_one_so_sensitivity_holds(synthetic_credit):
    """Privacy argument: per-sample contribution = w_i * clip(g_i) with w_i <= 1,
    so the gradient sensitivity stays C. Check the trainer keeps w in (0, 1]."""
    d = prepare(synthetic_credit, seed=0)
    gy = d.train.groups.assign(_y=d.train.y.astype(str))
    cells = CellIndex(gy, d.group_attrs + ["_y"])
    for ov in (dict(dro=True, smoothing=True), dict(balance_labels=True)):
        cfg = DPConfig(target_epsilon=3.0, epochs=2, expected_batch_size=128, seed=1, aux_frac=0.05, **ov)
        _, log = train_dp(MLP(d.train.X.shape[1]), d.train.X, d.train.y, cells.encode(gy), cells, cfg)
        w = np.asarray(log.final_cell_weights)
        assert (w > 0).all() and (w <= 1 + 1e-12).all()
        # joint mechanism: whitened sensitivity^2 = 1/sg^2 + k/ss^2 = 1/sigma_eff^2
        n = log.noise
        assert math.isclose(1 / n.sigma_grad ** 2 + n.stats_blocks / n.sigma_stats ** 2, 1 / n.sigma_eff ** 2, rel_tol=1e-9)
        assert n.epsilon <= cfg.target_epsilon + 1e-6

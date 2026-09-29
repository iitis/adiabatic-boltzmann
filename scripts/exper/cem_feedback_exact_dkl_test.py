"""
cem_feedback_exact_dkl_test.py -- root-causes the DKL regression seen in the
real D-Wave protocol test (cem_feedback_dwave_protocol_test.py): new_trust
correctly hit beta_eff=1 on both real devices, but had WORSE tail-averaged
DKL than the old (buggy) rule, because it needed a much larger beta_x
correction (1 -> 4.5-9.6 vs old's 1 -> 2.6-3.3) and only had 80 iterations /
a 40-iteration tail window to settle.

Two competing explanations:
  (a) genuine real-hardware effect: large beta_x means weak couplings,
      dominated by device control noise (ICE) -- would NOT show up here,
      since this test has no hardware noise at all.
  (b) pure convergence-time artifact: new_trust's larger required correction
      just has a longer transient, and once it settles its EXACT (noise-free)
      steady-state DKL should be <= old's, same as at N=8/N=16 GPU tests.

This script computes DKL EXACTLY via full 2^N enumeration at every
iteration (no finite-sample noise in the METRIC -- only the CEM estimate
beta_hat driving the beta_x updates is sample-based, exactly as in real
experiments), using the same exact-Gibbs-at-prescribed-beta_hw harness and
the same trained late-stage N=16 h=1.0 network as
cem_feedback_gpu_protocol_test.py, but run 4x longer (400 iterations) so any
convergence-time confound has time to resolve.

p_beta(v) ~ exp(-beta*a.v) * prod_j 2cosh(beta*Theta_j(v)) is the exact
visible marginal at inverse temperature beta (same formula used for the
ground-truth beta_eff check in cem_validation_sweep.py); DKL is against
pi_theta = p_beta(v) at beta=1 (the model's own exact Born distribution).

Usage:
    python scripts/exper/cem_feedback_exact_dkl_test.py
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from sampler import ClassicalSampler
from encoder import (
    SRLinearSystem, conjugate_gradient, estimate_beta_eff_cem,
    is_cem_fit_degenerate, is_cem_step_untrusted,
)
from kl_utils import all_configs_jax

N, M = 16, 16
H_FIELD = 1.0
LR = 0.08
REG = 0.05
N_TRAIN_SAMPLES = 500
N_TRAIN_ITER = 150

N_SAMPLES = 200
N_ITER = 400
CEM_EMA_ALPHA = 0.3
BETA_MIN, BETA_MAX = 0.05, 20.0
BETA_HW_VALUES = [1.0, 3.0, 8.0]
N_SEEDS = 4

_CONFIGS_CACHE = {}


def exact_marginal_at_beta(rbm, N, beta):
    """Exact p_beta(v) over all 2^N configs -- see module docstring."""
    if N not in _CONFIGS_CACHE:
        _CONFIGS_CACHE[N] = all_configs_jax(N)
    configs = _CONFIGS_CACHE[N]
    a_v = configs @ rbm.a
    theta = configs @ rbm.W + rbm.b[None, :]
    log_unnorm = -beta * a_v + jnp.sum(jnp.log(2.0 * jnp.cosh(beta * theta)), axis=1)
    return jax.nn.softmax(log_unnorm)


def exact_dkl_at_beta(rbm, N, beta_eff):
    """DKL(p_beta_eff(v) || pi_theta(v)) computed exactly, no sampling noise."""
    p = exact_marginal_at_beta(rbm, N, beta_eff)
    pi = exact_marginal_at_beta(rbm, N, 1.0)
    mask = p > 0
    return float(jnp.sum(jnp.where(mask, p * (jnp.log(jnp.where(mask, p, 1.0)) - jnp.log(pi)), 0.0)))


def train_late_stage_rbm(seed):
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(seed))
    ising = TransverseFieldIsing1D(N, H_FIELD)
    sampler = ClassicalSampler(method="gibbs", n_warmup=200, n_sweeps=5)
    sampler._key = jax.random.PRNGKey(seed + 500)
    for _ in range(N_TRAIN_ITER):
        V = sampler.sample(rbm, N_TRAIN_SAMPLES, config={}, return_hidden=False, return_jax=True)
        E = ising.local_energy_batch(V, rbm)
        Theta = V @ rbm.W + rbm.b[None, :]
        TanH = jnp.tanh(Theta)
        sr = SRLinearSystem(V, TanH, E, REG)
        x, _ = conjugate_gradient(sr.matvec, sr.force, tol=1e-8, maxiter=200)
        xa, xb, xW = sr.unpack(x)
        update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
        rbm.set_weights(rbm.get_weights() - LR * update)
    return rbm


def scaled_rbm(rbm, scale):
    r = FullyConnectedRBM(rbm.n_visible, rbm.n_hidden, jax.random.PRNGKey(0))
    r.a = rbm.a * scale
    r.b = rbm.b * scale
    r.W = rbm.W * scale
    return r


def run_rule(rule, beta_hw, base_rbm, seed):
    sampler = ClassicalSampler(method="gibbs", n_warmup=100, n_sweeps=1)
    sampler._key = jax.random.PRNGKey(2000 * seed + int(beta_hw * 7))

    beta_x = 1.0
    beta_x_traj, exact_dkl_traj = [], []
    for _ in range(N_ITER):
        scale = beta_hw / beta_x
        rbm_scaled = scaled_rbm(base_rbm, scale)
        V, H = sampler.sample(rbm_scaled, N_SAMPLES, config={}, return_hidden=True, return_jax=True)
        beta_hat = estimate_beta_eff_cem(V, H, base_rbm)

        if rule == "old":
            beta_x = (1 - CEM_EMA_ALPHA) * beta_x + CEM_EMA_ALPHA * beta_hat
        elif rule == "new_trust":
            if not is_cem_fit_degenerate(beta_hat) and not is_cem_step_untrusted(beta_hat, beta_x):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        else:
            raise ValueError(rule)
        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        beta_x_traj.append(beta_x)

        # exact beta_eff realised THIS iteration relative to the frozen base network
        beta_eff_now = beta_hw / beta_x
        exact_dkl_traj.append(exact_dkl_at_beta(base_rbm, N, beta_eff_now))

    return np.array(beta_x_traj), np.array(exact_dkl_traj)


if __name__ == "__main__":
    print("--- training late-stage N=16 h=1.0 networks (reused across rules/beta_hw) ---")
    base_rbms = [train_late_stage_rbm(seed) for seed in range(N_SEEDS)]

    for rule in ["old", "new_trust"]:
        print(f"\n=== rule = {rule} ===")
        for beta_hw in BETA_HW_VALUES:
            early_dkls, late_dkls, final_beta_xs = [], [], []
            for seed in range(N_SEEDS):
                beta_x_traj, dkl_traj = run_rule(rule, beta_hw, base_rbms[seed], seed)
                early_dkls.append(dkl_traj[:N_ITER // 4].mean())      # first 25% = transient
                late_dkls.append(dkl_traj[-N_ITER // 4:].mean())      # last 25% = steady state
                final_beta_xs.append(beta_x_traj[-N_ITER // 4:].mean())
            early_dkls, late_dkls, final_beta_xs = map(np.array, (early_dkls, late_dkls, final_beta_xs))
            print(f"  beta_hw={beta_hw:4.1f}  early(transient)_DKL={early_dkls.mean():.5f}+-{early_dkls.std():.5f}  "
                  f"late(steady-state)_DKL={late_dkls.mean():.5f}+-{late_dkls.std():.5f}  "
                  f"final_beta_x={final_beta_xs.mean():.3f}+-{final_beta_xs.std():.3f}")

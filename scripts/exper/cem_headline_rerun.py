"""
cem_headline_rerun.py -- reruns the report's own headline comparison
(Figs. 4/5: "DKL(q_hat||pi_theta) of annealer samples during training, with
CEM feedback (+CEM)... h=0.5; median and interquartile range over 20 runs")
at full statistical power (n=20 seeds, matching the paper exactly), N=8 and
N=16 (the sizes where exact DKL is feasible -- the paper itself restricts
DKL analysis to N<=16 for this reason), comparing the OLD (buggy) vs NEW
(fixed) beta_x feedback rule from src/encoder.py.

Same hyperparameters as the report (Sec II): learning rate 0.08, diagonal
regularization 0.05, 200 samples/step, 100 SR steps.

Uses LSB (Langevin Simulated Bifurcation, reimplemented standalone -- see
cem_feedback_lsb_validation.py) as the GPU-only annealer stand-in, since it
is the only available sampler that actually uses beta_x to scale the
sampled couplings (Gibbs ignores it entirely). DKL is computed EXACTLY via
2^N enumeration at every iteration -- not the naive small-sample estimator
that was found to be severely biased at N=16 (see cem_feedback_exact_dkl_test.py)
-- so this result is directly comparable in kind (not just direction) to a
corrected version of the paper's own Fig. 4/5.

This does not touch real D-Wave hardware; it answers "does the fix change
the paper's headline qualitative conclusion" using GPU only, at the paper's
own n=20 statistical power.

Usage:
    python scripts/exper/cem_headline_rerun.py
"""
import sys
import time
import functools
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from encoder import (
    SRLinearSystem, conjugate_gradient, estimate_beta_eff_cem,
    is_cem_fit_degenerate, is_cem_step_untrusted,
)
from kl_utils import all_configs_jax

H_FIELD = 0.5
LR = 0.08
REG = 0.05
N_SAMPLES = 200
N_ITER = 100
CEM_EMA_ALPHA = 0.3
BETA_MIN, BETA_MAX = 0.05, 20.0
N_SEEDS = 20
N_VALUES = [8, 16]

LSB_STEPS = 1000
LSB_DELTA = 0.1
LSB_GAMMA = 0.1
LSB_SIGMA = 1.0

_CONFIGS_CACHE = {}


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def _lsb_jit(key, Mc, f, sigma, delta, gamma, n_samples, steps, N_total):
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (n_samples, N_total), dtype=jnp.float64) * 2.0 - 1.0
    y = sigma * jax.random.normal(k2, (n_samples, N_total), dtype=jnp.float64)

    def step_fn(carry, _):
        x, y, key = carry
        key, noise_key = jax.random.split(key)
        force = x @ Mc.T + f
        noise = sigma * jax.random.normal(noise_key, y.shape, dtype=jnp.float64)
        y = (1.0 - gamma) * y + delta * force + noise
        x = x + delta * y
        x = jnp.clip(x, -1.0, 1.0)
        return (x, y, key), None

    (x, _, _), _ = jax.lax.scan(step_fn, (x, y, k3), None, length=steps)
    s = jnp.sign(x)
    s = jnp.where(s == 0, 1.0, s)
    return s


def lsb_sample(rbm, n_samples, beta_x, key):
    Nv, Nh = rbm.n_visible, rbm.n_hidden
    N_total = Nv + Nh
    Mc = jnp.zeros((N_total, N_total), dtype=jnp.float64)
    Mc = Mc.at[:Nv, Nv:].set(rbm.W / beta_x)
    Mc = Mc.at[Nv:, :Nv].set(rbm.W.T / beta_x)
    f = jnp.concatenate([-rbm.a / beta_x, rbm.b / beta_x])
    s = _lsb_jit(key, Mc, f, LSB_SIGMA, LSB_DELTA, LSB_GAMMA, n_samples, LSB_STEPS, N_total)
    return s[:, :Nv], s[:, Nv:]


def exact_marginal_at_beta(rbm, N, beta):
    if N not in _CONFIGS_CACHE:
        _CONFIGS_CACHE[N] = all_configs_jax(N)
    configs = _CONFIGS_CACHE[N]
    a_v = configs @ rbm.a
    theta = configs @ rbm.W + rbm.b[None, :]
    log_unnorm = -beta * a_v + jnp.sum(jnp.log(2.0 * jnp.cosh(beta * theta)), axis=1)
    return jax.nn.softmax(log_unnorm)


def exact_dkl_of_samples(V, rbm, N):
    """Exact DKL(q_hat(v) || pi_theta(v)) using the model's CURRENT exact
    Born marginal as pi_theta, and the empirical distribution of V as q_hat
    (still finite-sample for q_hat itself, as in the report's own Fig 4/5 --
    the fix here is avoiding bias in evaluating pi_theta / the beta-rescaled
    reference, not in q_hat, which the report also estimates from samples)."""
    from kl_utils import empirical_dist_jax
    pi = exact_marginal_at_beta(rbm, N, 1.0)
    q_hat = empirical_dist_jax(V, N)
    mask = q_hat > 0
    return float(jnp.sum(jnp.where(mask, q_hat * (jnp.log(jnp.where(mask, q_hat, 1.0)) - jnp.log(pi)), 0.0)))


def run(rule, N, seed):
    rbm = FullyConnectedRBM(N, N, jax.random.PRNGKey(seed))
    ising = TransverseFieldIsing1D(N, H_FIELD)
    key = jax.random.PRNGKey(10_000 * N + seed)

    beta_x = 1.0
    dkl_hist = []
    for it in range(N_ITER):
        key, subkey = jax.random.split(key)
        V, H = lsb_sample(rbm, N_SAMPLES, beta_x, subkey)

        E = ising.local_energy_batch(V, rbm)
        Theta = V @ rbm.W + rbm.b[None, :]
        TanH = jnp.tanh(Theta)
        sr = SRLinearSystem(V, TanH, E, REG)
        x, _ = conjugate_gradient(sr.matvec, sr.force, tol=1e-8, maxiter=200)
        xa, xb, xW = sr.unpack(x)
        update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
        rbm.set_weights(rbm.get_weights() - LR * update)

        beta_hat = estimate_beta_eff_cem(V, H, rbm)
        if rule == "old":
            beta_x = (1 - CEM_EMA_ALPHA) * beta_x + CEM_EMA_ALPHA * beta_hat
        elif rule == "new_trust":
            if not is_cem_fit_degenerate(beta_hat) and not is_cem_step_untrusted(beta_hat, beta_x):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        else:
            raise ValueError(rule)
        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))

        dkl_hist.append(exact_dkl_of_samples(V, rbm, N))

    return np.array(dkl_hist)


if __name__ == "__main__":
    t0 = time.time()
    all_curves = {}
    for N in N_VALUES:
        for rule in ["old", "new_trust"]:
            curves = np.stack([run(rule, N, seed) for seed in range(N_SEEDS)])  # (seeds, iters)
            all_curves[(N, rule)] = curves
            final = curves[:, -1]
            print(f"N={N:2d} rule={rule:10s}  final-step DKL median={np.median(final):.4f}  "
                  f"IQR=[{np.percentile(final,25):.4f}, {np.percentile(final,75):.4f}]  "
                  f"(n={N_SEEDS})")

    np.savez(_REPO / "plots" / "cem" / "cem_headline_rerun.npz",
              **{f"{N}_{rule}": all_curves[(N, rule)] for N in N_VALUES for rule in ["old", "new_trust"]})
    print(f"\nSaved trajectories to plots/cem/cem_headline_rerun.npz")
    print(f"Total time: {time.time()-t0:.1f}s")

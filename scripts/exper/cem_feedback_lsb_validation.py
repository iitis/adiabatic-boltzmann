"""
cem_feedback_lsb_validation.py -- validates the beta_x feedback-rule fix
(src/encoder.py) against a genuinely different sampler: Langevin Simulated
Bifurcation (LSB, Kubo & Goto 2025), which -- unlike the classical Gibbs
sampler used in cem_feedback_rule_test.py -- actually uses beta_x to scale
the couplings it samples from (Gibbs ignores beta_x entirely; see the NOTE
in cem_validation_sweep.py). LSB was removed from src/sampler.py at commit
0b8ee9f8b; it is reimplemented here standalone (unchanged kernel) purely for
this validation, not reintroduced into the shipped sampler.

D-Wave hardware was also considered (the user offered up to 2 minutes of
device time) but the configured token in ~/.config/dwave/dwave.conf fails
authentication (SolverAuthenticationError: Invalid token or access denied),
so this run substitutes LSB as the "different physical sampler" check.

Runs a REAL SR training loop (same SRLinearSystem/conjugate_gradient/
estimate_beta_eff_cem as encoder.Trainer) on TFIM N=8, h=0.5 -- the exact
instance size and field used in the report's own Sec III.C confirmation --
under the OLD (buggy, linear EMA) and NEW (log-EMA + reject) beta_x rules,
and compares the actual quantity the paper cares about: DKL(q_hat(v) || pi_theta(v))
against the model's own exact Born distribution (feasible by enumeration at
N=8), exactly as in Figs. 4/5 of the report.

Usage:
    python scripts/exper/cem_feedback_lsb_validation.py
"""
import sys
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
from kl_utils import all_configs_jax, exact_psi_sq, empirical_dist_jax

N, M = 8, 8
H_FIELD = 0.5
LR = 0.08
REG = 0.05
N_SAMPLES = 200
N_ITER = 100
CEM_EMA_ALPHA = 0.3
BETA_MIN, BETA_MAX = 0.05, 20.0
N_SEEDS = 8

# LSB defaults, unchanged from the removed implementation
LSB_STEPS = 1000
LSB_DELTA = 0.1
LSB_GAMMA = 0.1
LSB_SIGMA = 1.0  # = 1/sqrt(lsb_sigma_inv2=1.0)


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


def d_kl(q, p):
    mask = q > 0
    return float(jnp.sum(jnp.where(mask, q * (jnp.log(jnp.where(mask, q, 1.0)) - jnp.log(p)), 0.0)))


def train(rule, seed):
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(seed))
    ising = TransverseFieldIsing1D(N, H_FIELD)
    key = jax.random.PRNGKey(10_000 + seed)

    beta_x = 1.0
    dkl_hist, beta_hist = [], []
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
        elif rule == "new":
            # bounds-reject only (no trust-region guard) -- for comparison
            if not is_cem_fit_degenerate(beta_hat):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        elif rule == "new_trust":
            # exactly the shipped src/encoder.py rule
            if not is_cem_fit_degenerate(beta_hat) and not is_cem_step_untrusted(beta_hat, beta_x):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        else:
            raise ValueError(rule)
        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        beta_hist.append(beta_x)

        p_exact = exact_psi_sq(rbm, N)
        q_hat = empirical_dist_jax(V, N)
        dkl_hist.append(d_kl(q_hat, p_exact))

    return np.array(dkl_hist), np.array(beta_hist)


if __name__ == "__main__":
    for rule in ["old", "new", "new_trust"]:
        finals_dkl, finals_beta = [], []
        for seed in range(N_SEEDS):
            dkl_hist, beta_hist = train(rule, seed)
            finals_dkl.append(dkl_hist[-15:].mean())
            finals_beta.append(beta_hist[-15:].mean())
            print(f"  [{rule}] seed={seed}  final DKL={dkl_hist[-1]:.4f}  "
                  f"final beta_x={beta_hist[-1]:.3f}")
        finals_dkl = np.array(finals_dkl)
        finals_beta = np.array(finals_beta)
        print(f"=== rule={rule}: DKL mean={finals_dkl.mean():.4f}+-{finals_dkl.std():.4f}  "
              f"median={np.median(finals_dkl):.4f}  beta_x={finals_beta.mean():.3f}+-{finals_beta.std():.3f} ===\n")

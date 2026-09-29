"""
cem_feedback_rule_test.py -- compares candidate fixes for the beta_x feedback
rule bug (report Sec III.C): the current rule beta_x <- 0.7*beta_x + 0.3*beta_hat
is an arithmetic mean of beta_x and beta_hat, but beta_hat estimates the RATIO
beta_eff = beta_hw/beta_x, so any weighted mean of beta_x and beta_hat has fixed
point beta_x* = beta_hat* regardless of the mix weight, which combined with
beta_hat* = beta_hw/beta_x* forces beta_x* = sqrt(beta_hw) -- wrong unless
beta_hw = 1.

Candidate rules (all with the same damping strength alpha=0.3, same as the
current default cem_ema_alpha):

  current  (buggy):  beta_x <- (1-a)*beta_x + a*beta_hat
  naive    (paper's "multiplicative update", alpha=1, no damping):
                      beta_x <- beta_x * beta_hat
  log_ema  (proposed fix): beta_x <- beta_x * beta_hat**a
                      i.e. exponential smoothing done in log(beta_x) space,
                      since beta_hat is a ratio not an absolute temperature.
                      Reduces to "naive" at a=1.

Setup mirrors the paper's own confirmation experiment (Sec III.C): exact
Gibbs sampling of the RBM with parameters scaled by beta_hw/beta_x (emulating
a fixed hardware inverse temperature beta_hw and a trainable software scale
beta_x), beta_hat computed via the real encoder.estimate_beta_eff_cem on the
UNSCALED parameters -- byte-for-byte the same call the real Trainer makes.

Usage:
    python scripts/exper/cem_feedback_rule_test.py
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
from encoder import estimate_beta_eff_cem
from sampler import ClassicalSampler

N, M = 8, 8
N_SAMPLES = 200          # matches paper's "200 pairs" per CEM fit
N_ITER = 80              # matches typical SR run length used in Sec III.C-ish
ALPHA = 0.3              # same damping weight as the current cem_ema_alpha default
BETA_MIN, BETA_MAX = 0.05, 20.0
BETA_HW_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0]
N_SEEDS = 9              # paper used "3 runs per value" / "9 runs" for the naive rule check


def make_rbm(seed):
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(seed))
    rbm.W = rbm.W * 2.0   # nontrivial coupling strength, not near-zero
    return rbm


def scaled_rbm(rbm, scale):
    r = FullyConnectedRBM(rbm.n_visible, rbm.n_hidden, jax.random.PRNGKey(0))
    r.a = rbm.a * scale
    r.b = rbm.b * scale
    r.W = rbm.W * scale
    return r


def run_rule(rule, beta_hw, seed):
    rbm = make_rbm(seed)
    sampler = ClassicalSampler(method="gibbs", n_warmup=100, n_sweeps=1)
    sampler._key = jax.random.PRNGKey(1000 * seed + int(beta_hw * 7))

    beta_x = 1.0
    traj = []
    for _ in range(N_ITER):
        scale = beta_hw / beta_x
        rbm_scaled = scaled_rbm(rbm, scale)
        V, H = sampler.sample(rbm_scaled, N_SAMPLES, config={}, return_hidden=True, return_jax=True)
        beta_hat = estimate_beta_eff_cem(V, H, rbm)  # unscaled rbm, exactly as Trainer does

        if rule == "current":
            beta_x = (1 - ALPHA) * beta_x + ALPHA * beta_hat
        elif rule == "naive":
            beta_x = beta_x * beta_hat
        elif rule == "log_ema":
            beta_x = beta_x * (beta_hat ** ALPHA)
        else:
            raise ValueError(rule)

        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        traj.append(beta_x)
    return np.array(traj)


def summarize(rule):
    print(f"\n=== rule = {rule} ===")
    for beta_hw in BETA_HW_VALUES:
        finals = []
        blew_up = 0
        for seed in range(N_SEEDS):
            traj = run_rule(rule, beta_hw, seed)
            tail = traj[-10:]  # last 10 iterations
            finals.append(tail.mean())
            # instability: relative swing in the tail
            if tail.std() / max(tail.mean(), 1e-9) > 0.25:
                blew_up += 1
        finals = np.array(finals)
        beta_eff = beta_hw / finals
        print(
            f"  beta_hw={beta_hw:4.1f}  target beta_x={beta_hw:5.2f}  sqrt(beta_hw)={np.sqrt(beta_hw):5.2f}  "
            f"beta_x_final={finals.mean():6.3f}+-{finals.std():5.3f}  "
            f"beta_eff={beta_eff.mean():5.3f}+-{beta_eff.std():5.3f}  "
            f"unstable_runs={blew_up}/{N_SEEDS}"
        )


if __name__ == "__main__":
    for rule in ["current", "naive", "log_ema"]:
        summarize(rule)

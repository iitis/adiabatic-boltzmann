"""
cem_feedback_rule_tune.py -- quick alpha sweep for the log_ema fix, plus a
"reject degenerate fits" safeguard (skip the update when beta_hat is pinned
at the CEM search bounds [0.01, 50], a known sign of an unreliable fit --
see the outlier analysis in the report, Fig. 2 caption).

Reuses the same setup as cem_feedback_rule_test.py; smaller grid for speed.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from model import FullyConnectedRBM
from encoder import estimate_beta_eff_cem
from sampler import ClassicalSampler

N, M = 8, 8
N_SAMPLES = 200
N_ITER = 120
BETA_MIN, BETA_MAX = 0.05, 20.0
CEM_BOUNDS = (0.01, 50.0)
BETA_HW_VALUES = [1.0, 3.0, 8.0]
N_SEEDS = 12


def make_rbm(seed):
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(seed))
    rbm.W = rbm.W * 2.0
    return rbm


def scaled_rbm(rbm, scale):
    r = FullyConnectedRBM(rbm.n_visible, rbm.n_hidden, jax.random.PRNGKey(0))
    r.a = rbm.a * scale
    r.b = rbm.b * scale
    r.W = rbm.W * scale
    return r


def run_rule(rule, alpha, beta_hw, seed):
    rbm = make_rbm(seed)
    sampler = ClassicalSampler(method="gibbs", n_warmup=100, n_sweeps=1)
    sampler._key = jax.random.PRNGKey(1000 * seed + int(beta_hw * 7))

    beta_x = 1.0
    traj = []
    for _ in range(N_ITER):
        scale = beta_hw / beta_x
        rbm_scaled = scaled_rbm(rbm, scale)
        V, H = sampler.sample(rbm_scaled, N_SAMPLES, config={}, return_hidden=True, return_jax=True)
        beta_hat = estimate_beta_eff_cem(V, H, rbm)

        degenerate = (beta_hat < CEM_BOUNDS[0] * 1.5) or (beta_hat > CEM_BOUNDS[1] * 0.98)

        if rule == "log_ema":
            if not degenerate:
                beta_x = beta_x * (beta_hat ** alpha)
        elif rule == "log_ema_no_reject":
            beta_x = beta_x * (beta_hat ** alpha)
        else:
            raise ValueError(rule)

        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        traj.append(beta_x)
    return np.array(traj)


def summarize(rule, alpha):
    print(f"\n=== rule={rule}  alpha={alpha} ===")
    for beta_hw in BETA_HW_VALUES:
        finals = []
        for seed in range(N_SEEDS):
            traj = run_rule(rule, alpha, beta_hw, seed)
            finals.append(traj[-15:].mean())
        finals = np.array(finals)
        beta_eff = beta_hw / finals
        print(
            f"  beta_hw={beta_hw:4.1f}  beta_x_final={finals.mean():6.3f}+-{finals.std():5.3f}  "
            f"beta_eff={beta_eff.mean():5.3f}+-{beta_eff.std():5.3f}"
        )


if __name__ == "__main__":
    print("--- with rejection of degenerate fits ---")
    for alpha in [0.3, 0.4, 0.5]:
        summarize("log_ema", alpha)

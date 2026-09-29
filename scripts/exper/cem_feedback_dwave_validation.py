"""
cem_feedback_dwave_validation.py -- validates the beta_x feedback-rule fix
(src/encoder.py) on REAL D-Wave hardware (Advantage_system6 / pegasus),
using the chain-free DWaveTopologyRBM (no embedding chains, so no chain-
strength tuning and minimal QPU overhead per call).

Strict device-time budget: this script tracks cumulative qpu_access_time
(the actual billed device time, not wall-clock) via sampler.last_sampling_time_s
and stops submitting new jobs once DEVICE_TIME_BUDGET_S is reached. A single
100-read call on this chain-free embedding measured ~26ms of device time, so
the planned N_SEEDS x 2 rules x N_ITER calls is expected to cost a few
seconds total -- far under the budget -- but the check is enforced
unconditionally regardless of that estimate.

Same real SR training loop and same DKL(q_hat(v) || pi_theta(v)) metric as
cem_feedback_lsb_validation.py (report's own Sec IV methodology), same
instance size (N=8, h=0.5) and same report hyperparameters (lr=0.08, reg=0.05).

Usage:
    python scripts/exper/cem_feedback_dwave_validation.py
"""
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from model import DWaveTopologyRBM
from ising import TransverseFieldIsing1D
from sampler import DimodSampler
from encoder import (
    SRLinearSystem, conjugate_gradient, estimate_beta_eff_cem,
    is_cem_fit_degenerate, is_cem_step_untrusted,
)
from kl_utils import exact_psi_sq, empirical_dist_jax

N, M = 8, 8
H_FIELD = 0.5
LR = 0.08
REG = 0.05
N_SAMPLES = 150
N_ITER = 20
CEM_EMA_ALPHA = 0.3
BETA_MIN, BETA_MAX = 0.05, 20.0
N_SEEDS = 3

DEVICE_TIME_BUDGET_S = 100.0  # hard cap, under the user's 120s allowance
_cumulative_device_time_s = 0.0


def d_kl(q, p):
    mask = q > 0
    return float(jnp.sum(jnp.where(mask, q * (jnp.log(jnp.where(mask, q, 1.0)) - jnp.log(p)), 0.0)))


def budget_check():
    if _cumulative_device_time_s >= DEVICE_TIME_BUDGET_S:
        raise RuntimeError(
            f"Device-time budget exhausted: {_cumulative_device_time_s:.2f}s "
            f">= {DEVICE_TIME_BUDGET_S:.2f}s. Stopping before another QPU call."
        )


def train(rule, seed, sampler):
    global _cumulative_device_time_s
    rbm = DWaveTopologyRBM(N, M, jax.random.PRNGKey(seed), solver="pegasus", seed=42, live=True)
    ising = TransverseFieldIsing1D(N, H_FIELD)

    beta_x = 1.0
    dkl_hist, beta_hist, device_time_hist = [], [], []
    for it in range(N_ITER):
        budget_check()
        V_raw, H_raw = sampler.sample(
            rbm, N_SAMPLES,
            config={"beta_x": beta_x, "annealing_time": 20, "num_reads": N_SAMPLES},
            return_hidden=True,
        )
        _cumulative_device_time_s += sampler.last_sampling_time_s
        device_time_hist.append(_cumulative_device_time_s)

        V = jnp.asarray(V_raw, dtype=jnp.float64)
        H = jnp.asarray(H_raw, dtype=jnp.float64)

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
            if not is_cem_fit_degenerate(beta_hat) and not is_cem_step_untrusted(beta_hat, beta_x):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        else:
            raise ValueError(rule)
        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        beta_hist.append(beta_x)

        p_exact = exact_psi_sq(rbm, N)
        q_hat = empirical_dist_jax(V, N)
        dkl_hist.append(d_kl(q_hat, p_exact))

        print(f"    [{rule} seed={seed} it={it:2d}] beta_hat={beta_hat:7.4f} "
              f"beta_x={beta_x:7.4f} DKL={dkl_hist[-1]:.4f} "
              f"cum_device_time={_cumulative_device_time_s:.3f}s")

    return np.array(dkl_hist), np.array(beta_hist)


if __name__ == "__main__":
    sampler = DimodSampler(method="pegasus")
    t0 = time.time()
    for rule in ["old", "new"]:
        finals_dkl, finals_beta = [], []
        for seed in range(N_SEEDS):
            try:
                dkl_hist, beta_hist = train(rule, seed, sampler)
            except RuntimeError as e:
                print(f"  BUDGET STOP: {e}")
                break
            finals_dkl.append(dkl_hist[-5:].mean())
            finals_beta.append(beta_hist[-5:].mean())
        finals_dkl = np.array(finals_dkl)
        finals_beta = np.array(finals_beta)
        print(f"=== rule={rule}: DKL mean={finals_dkl.mean():.4f}+-{finals_dkl.std():.4f}  "
              f"median={np.median(finals_dkl):.4f}  beta_x={finals_beta.mean():.3f}+-{finals_beta.std():.3f} ===\n")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    print(f"Total QPU device time used: {_cumulative_device_time_s:.3f}s "
          f"(budget was {DEVICE_TIME_BUDGET_S:.0f}s, user cap 120s)")

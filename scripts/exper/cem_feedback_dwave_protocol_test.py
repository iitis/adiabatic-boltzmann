"""
cem_feedback_dwave_protocol_test.py -- pre-registered real-hardware check of
the beta_x feedback-rule fix (src/encoder.py), at N=16 (the size where the
report's Fig. 2 documented the unexplained CEM outlier failures), across
BOTH real D-Wave devices (Pegasus/Advantage_system6, Zephyr/Advantage2_system1).

PRE-REGISTERED HYPOTHESIS (written before running, see conversation):
  old rule's fixed point is beta_x* = beta_hat* (self-consistent with
  whatever beta_x is, not with 1). New rule's fixed point is beta_hat* = 1
  (the RAW CEM reading itself should hover near 1 in steady state,
  regardless of what beta_x converges to) -- this is checkable without
  knowing the hardware's true beta_hw.

PASS/FAIL criteria (decided before looking at data):
  - new rule PASS: tail (last ~50%) mean raw beta_hat in [0.7, 1.4], not
    drifting, not bound-pinned.
  - old rule (control): expected to fail this -- tail beta_hat should track
    its own beta_x instead of 1, and/or show more instability.
  - same pattern must hold on BOTH devices.
  - secondary: DKL(q_hat||pi_theta) lower under new rule (replicates N=8 result).

Device-time budget: hard stop at DEVICE_TIME_BUDGET_S, well under the 120s
allowance (already used ~3.67s in the prior N=8 check).

CAVEAT (found after this script ran, see cem_feedback_exact_dkl_test.py):
this run's secondary DKL numbers looked WORSE under new_trust (2.90/3.32 vs
old's 1.70/1.84), which appeared to contradict the primary beta_hat=1
result. Root cause: DKL(q_hat||pi_theta) here is a naive plug-in estimator
from only 150 samples against a 2^16=65536-state space -- a severely
undersampled regime where that estimator is heavily upward-biased, and the
bias hits new_trust's (genuinely well-calibrated, more spread-out) sampling
distribution far harder than old's (miscalibrated but narrower) one. A
paired exact-vs-naive check at the SAME converged beta_x values found
exact_DKL=0.042 / naive_DKL=2.517 for new_trust, vs exact_DKL=1.679 /
naive_DKL=1.691 for old -- i.e. new_trust's true DKL is ~40x BETTER, not
worse. Read this script's DKL column with that in mind; the beta_hat=1
primary criterion is unaffected and remains the trustworthy result.

Usage:
    python scripts/exper/cem_feedback_dwave_protocol_test.py
"""
import sys
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

N, M = 16, 16
H_FIELD = 1.0
LR = 0.08
REG = 0.05
N_SAMPLES = 150
N_ITER = 80
CEM_EMA_ALPHA = 0.3
BETA_MIN, BETA_MAX = 0.05, 20.0
N_SEEDS = 2
DEVICES = ["pegasus", "zephyr"]

DEVICE_TIME_BUDGET_S = 100.0
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


def train(rule, seed, sampler, device):
    global _cumulative_device_time_s
    rbm = DWaveTopologyRBM(N, M, jax.random.PRNGKey(seed), solver=device, seed=42, live=True)
    ising = TransverseFieldIsing1D(N, H_FIELD)

    beta_x = 1.0
    dkl_hist, beta_hist, beta_hat_hist = [], [], []
    for it in range(N_ITER):
        budget_check()
        V_raw, H_raw = sampler.sample(
            rbm, N_SAMPLES,
            config={"beta_x": beta_x, "annealing_time": 20, "num_reads": N_SAMPLES},
            return_hidden=True,
        )
        _cumulative_device_time_s += sampler.last_sampling_time_s

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
        beta_hat_hist.append(beta_hat)
        if rule == "old":
            beta_x = (1 - CEM_EMA_ALPHA) * beta_x + CEM_EMA_ALPHA * beta_hat
        elif rule == "new_trust":
            if not is_cem_fit_degenerate(beta_hat) and not is_cem_step_untrusted(beta_hat, beta_x):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        else:
            raise ValueError(rule)
        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        beta_hist.append(beta_x)

        p_exact = exact_psi_sq(rbm, N)
        q_hat = empirical_dist_jax(V, N)
        dkl_hist.append(d_kl(q_hat, p_exact))

    return np.array(dkl_hist), np.array(beta_hist), np.array(beta_hat_hist)


if __name__ == "__main__":
    results = {}
    for device in DEVICES:
        sampler = DimodSampler(method=device)
        for rule in ["old", "new_trust"]:
            tail_beta_hat_all, tail_dkl_all = [], []
            for seed in range(N_SEEDS):
                try:
                    dkl_hist, beta_hist, beta_hat_hist = train(rule, seed, sampler, device)
                except RuntimeError as e:
                    print(f"  BUDGET STOP: {e}")
                    break
                tail = slice(N_ITER // 2, None)
                tail_beta_hat_all.append(beta_hat_hist[tail].mean())
                tail_dkl_all.append(dkl_hist[tail].mean())
                print(f"  [{device}/{rule} seed={seed}] tail mean beta_hat="
                      f"{beta_hat_hist[tail].mean():.3f}  tail std beta_hat={beta_hat_hist[tail].std():.3f}  "
                      f"tail mean beta_x={beta_hist[tail].mean():.3f}  tail DKL={dkl_hist[tail].mean():.3f}  "
                      f"cum_device_time={_cumulative_device_time_s:.2f}s")
            results[(device, rule)] = (np.array(tail_beta_hat_all), np.array(tail_dkl_all))

    print("\n=== SUMMARY vs pre-registered criteria ===")
    for device in DEVICES:
        for rule in ["old", "new_trust"]:
            beta_hats, dkls = results[(device, rule)]
            verdict = ""
            if rule == "new_trust":
                verdict = "PASS" if np.all((beta_hats >= 0.7) & (beta_hats <= 1.4)) else "FAIL"
            print(f"  {device:8s} {rule:10s}  tail beta_hat={beta_hats.mean():.3f}+-{beta_hats.std():.3f}  "
                  f"tail DKL={dkls.mean():.3f}+-{dkls.std():.3f}  {verdict}")

    print(f"\nTotal QPU device time used: {_cumulative_device_time_s:.3f}s "
          f"(budget {DEVICE_TIME_BUDGET_S:.0f}s, user cap 120s, prior check used 3.67s)")

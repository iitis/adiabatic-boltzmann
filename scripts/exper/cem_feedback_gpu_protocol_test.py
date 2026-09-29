"""
cem_feedback_gpu_protocol_test.py -- pre-registered GPU check of the beta_x
feedback-rule fix (src/encoder.py), on a genuinely TRAINED "late" N=16, h=1.0
network (not just randomly-scaled weights), with EXACTLY KNOWN beta_hw (via
prescribed Gibbs scaling) -- see the conversation for the written protocol
and pass/fail bands, decided before this script was run.

Step 1: train one N=16, h=1.0 TFIM-RBM via plain SR+Gibbs (no CEM, beta=1)
for N_TRAIN_ITER iterations -- a genuine late-stage checkpoint, matching the
regime (N=16, h=1, mid/late network) where the report's Fig. 2 documented
unexplained CEM outlier failures.

Step 2: freeze those parameters. Run the exact prescribed-beta_hw feedback
rule test (Gibbs sampling scaled by beta_hw/beta_x, beta_hat computed via the
real encoder.estimate_beta_eff_cem on the frozen UNSCALED network) for
beta_hw in {1, 3, 8}, N_SEEDS seeds each, old vs new_trust rules.

PRE-REGISTERED PASS/FAIL (written before running):
  - new_trust PASS: for every beta_hw, tail-mean beta_eff = beta_hw/beta_x in
    [0.8, 1.25], tail relative std < 0.15.
  - old rule: expected to satisfy this only trivially at beta_hw=1, fail at
    beta_hw=3 (predicted beta_eff ~= sqrt(3) ~= 1.73) and beta_hw=8
    (predicted ~= sqrt(8) ~= 2.83).
  - single-step beta_x jumps > 3x should be ~0 under new_trust, nonzero
    under old.

Usage:
    python scripts/exper/cem_feedback_gpu_protocol_test.py
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

N, M = 16, 16
H_FIELD = 1.0
LR = 0.08
REG = 0.05
N_TRAIN_SAMPLES = 500
N_TRAIN_ITER = 150

N_SAMPLES = 200
N_ITER = 100
CEM_EMA_ALPHA = 0.3
BETA_MIN, BETA_MAX = 0.05, 20.0
BETA_HW_VALUES = [1.0, 3.0, 8.0]
N_SEEDS = 6


def train_late_stage_rbm(seed):
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(seed))
    ising = TransverseFieldIsing1D(N, H_FIELD)
    sampler = ClassicalSampler(method="gibbs", n_warmup=200, n_sweeps=5)
    sampler._key = jax.random.PRNGKey(seed + 500)

    for it in range(N_TRAIN_ITER):
        V = sampler.sample(rbm, N_TRAIN_SAMPLES, config={}, return_hidden=False, return_jax=True)
        E = ising.local_energy_batch(V, rbm)
        Theta = V @ rbm.W + rbm.b[None, :]
        TanH = jnp.tanh(Theta)
        sr = SRLinearSystem(V, TanH, E, REG)
        x, _ = conjugate_gradient(sr.matvec, sr.force, tol=1e-8, maxiter=200)
        xa, xb, xW = sr.unpack(x)
        update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
        rbm.set_weights(rbm.get_weights() - LR * update)
    print(f"  [train seed={seed}] final |W|_rms="
          f"{float(jnp.sqrt(jnp.mean(rbm.W**2))):.3f}  final E={float(jnp.mean(E)):.4f}")
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
    beta_x_traj = []
    max_jump = 0.0
    for _ in range(N_ITER):
        scale = beta_hw / beta_x
        rbm_scaled = scaled_rbm(base_rbm, scale)
        V, H = sampler.sample(rbm_scaled, N_SAMPLES, config={}, return_hidden=True, return_jax=True)
        beta_hat = estimate_beta_eff_cem(V, H, base_rbm)

        prev_beta_x = beta_x
        if rule == "old":
            beta_x = (1 - CEM_EMA_ALPHA) * beta_x + CEM_EMA_ALPHA * beta_hat
        elif rule == "new_trust":
            if not is_cem_fit_degenerate(beta_hat) and not is_cem_step_untrusted(beta_hat, beta_x):
                beta_x = beta_x * (beta_hat ** CEM_EMA_ALPHA)
        else:
            raise ValueError(rule)
        beta_x = float(np.clip(beta_x, BETA_MIN, BETA_MAX))
        max_jump = max(max_jump, abs(beta_x / prev_beta_x - 1.0) if prev_beta_x > 0 else 0.0)
        beta_x_traj.append(beta_x)
    return np.array(beta_x_traj), max_jump


if __name__ == "__main__":
    print("--- Step 1: training late-stage N=16 h=1.0 networks ---")
    base_rbms = [train_late_stage_rbm(seed) for seed in range(N_SEEDS)]

    print("\n--- Step 2: prescribed-beta_hw feedback-rule test on frozen late-stage networks ---")
    summary = {}
    for rule in ["old", "new_trust"]:
        print(f"\n=== rule = {rule} ===")
        for beta_hw in BETA_HW_VALUES:
            finals = []
            big_jumps = 0
            for seed in range(N_SEEDS):
                traj, max_jump = run_rule(rule, beta_hw, base_rbms[seed], seed)
                tail = traj[-30:]
                finals.append(tail.mean())
                if max_jump > 2.0:  # >3x jump means ratio-1 > 2.0
                    big_jumps += 1
            finals = np.array(finals)
            beta_eff = beta_hw / finals
            rel_std = (beta_hw / finals).std() / (beta_hw / finals).mean()
            verdict = ""
            if rule == "new_trust":
                verdict = "PASS" if (0.8 <= beta_eff.mean() <= 1.25 and rel_std < 0.15) else "FAIL"
            print(f"  beta_hw={beta_hw:4.1f}  beta_eff={beta_eff.mean():.3f}+-{beta_eff.std():.3f}  "
                  f"rel_std={rel_std:.3f}  big_jumps(>3x)={big_jumps}/{N_SEEDS}  {verdict}")
            summary[(rule, beta_hw)] = beta_eff.mean()

    print("\n=== SUMMARY ===")
    for beta_hw in BETA_HW_VALUES:
        print(f"  beta_hw={beta_hw}: old beta_eff={summary[('old', beta_hw)]:.3f} "
              f"(predicted sqrt={np.sqrt(beta_hw):.3f})  "
              f"new_trust beta_eff={summary[('new_trust', beta_hw)]:.3f} (predicted ~1.0)")

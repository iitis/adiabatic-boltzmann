"""
cem_feedback_scale_check.py -- stress-tests the beta_x feedback-rule fix at
the SAME (N, h) the report's own Fig. 2 used to document the unexplained CEM
outlier failures ("8 of 324 fits off by >3x, 5 reach beta=50, all at
beta_true>=2 on the mid and late networks"): N=16, h=1.0, trained long enough
to reach a "late" (larger-weight, more saturated) network, not just the
lightly-trained N=8 short runs used in cem_feedback_lsb_validation.py.

Reuses the exact same LSB harness/functions as cem_feedback_lsb_validation.py
(same real SR loop, same estimate_beta_eff_cem / is_cem_fit_degenerate /
is_cem_step_untrusted from encoder.py, same DKL(q_hat||pi_theta) metric) --
just at the scale where the original failures were actually observed.

CAVEAT (found after this script ran, see cem_feedback_exact_dkl_test.py):
its DKL(q_hat||pi_theta) is a naive plug-in estimator from only 200 samples
against a 2^16=65536-state space -- severely undersampled, and heavily
upward-biased in a way that hits a genuinely well-calibrated (more
spread-out) distribution harder than a miscalibrated (narrower) one. The
DKL numbers here (old=4.24, new_trust=1.39) still showed the right
direction, but should not be read as quantitatively reliable at this N;
cem_feedback_exact_dkl_test.py's exact-enumeration DKL is the trustworthy
version of this comparison.

Usage:
    python scripts/exper/cem_feedback_scale_check.py
"""
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "exper"))

import numpy as np
import cem_feedback_lsb_validation as m

m.N, m.M = 16, 16
m.H_FIELD = 1.0
m.N_ITER = 150
m.N_SEEDS = 5

if __name__ == "__main__":
    t0 = time.time()
    for rule in ["old", "new_trust"]:
        finals_dkl, finals_beta = [], []
        for seed in range(m.N_SEEDS):
            dkl_hist, beta_hist = m.train(rule, seed)
            finals_dkl.append(dkl_hist[-15:].mean())
            finals_beta.append(beta_hist[-15:].mean())
            n_pinned = int(np.sum(dkl_hist[-15:] > 5 * np.median(dkl_hist[-15:])))
            print(f"  [{rule}] seed={seed}  final DKL={dkl_hist[-1]:.4f}  "
                  f"tail_max={dkl_hist[-15:].max():.4f}  final beta_x={beta_hist[-1]:.3f}")
        finals_dkl = np.array(finals_dkl)
        finals_beta = np.array(finals_beta)
        print(f"=== rule={rule}: DKL mean={finals_dkl.mean():.4f}+-{finals_dkl.std():.4f}  "
              f"median={np.median(finals_dkl):.4f}  beta_x={finals_beta.mean():.3f}+-{finals_beta.std():.3f} "
              f"(N=16,h=1.0,{m.N_ITER}iters) ===\n")
    print(f"Total time: {time.time()-t0:.1f}s")

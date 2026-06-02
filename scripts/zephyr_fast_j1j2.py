"""
zephyr_fast_j1j2.py — zephyr (standard annealing) benchmark for heisenberg_j1j2_1d.

Runs N=8 and N=12 across selected J2 values with 2 seeds each, using fixed
sensible hyperparameters (no search).

Results are saved in the normal results/ tree via save_results().
"""

import itertools
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from hparam_optuna import run_trial, HAMILTONIAN_REGISTRY

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

N_VALUES   = [8, 12]
J2_VALUES  = [0.1, 0.2, 0.3, 0.5, 0.8]
SEEDS      = [0, 1]

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

N_ITERATIONS      = 150
N_SAMPLES         = 1000
N_HIDDEN_ALPHA    = 2       # n_hidden = alpha × N
LEARNING_RATE     = 0.05
REGULARIZATION    = 1e-4
CG_TOL            = 1e-8
CG_MAXITER        = 200
FAST_ANNEAL_TIME  = 7.0     # nanoseconds

HAMILTONIAN     = "heisenberg_j1j2_1d"
SAMPLING_METHOD = "zephyr"
OUTPUT_DIR      = _REPO / "results"

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    entry = HAMILTONIAN_REGISTRY[HAMILTONIAN]
    combos = list(itertools.product(N_VALUES, J2_VALUES, SEEDS))
    total = len(combos)

    print(f"zephyr benchmark — {HAMILTONIAN}")
    print(f"  N={N_VALUES}  J2={J2_VALUES}  seeds={SEEDS}")
    print(f"  {total} runs × {N_ITERATIONS} iters × ~0.3 s ≈ {total * N_ITERATIONS * 0.3 / 60:.1f} min\n")

    for idx, (N, J2, seed) in enumerate(combos, 1):
        n_hidden = int(N_HIDDEN_ALPHA * N)
        phys = {**entry["defaults"], "J2": J2}

        print(f"[{idx}/{total}] N={N}  J2={J2}  seed={seed}  n_hidden={n_hidden}")
        try:
            result = run_trial(
                N=N,
                hamiltonian=HAMILTONIAN,
                phys_params=phys,
                n_hidden=n_hidden,
                ansatz_type="rbm",
                lr=LEARNING_RATE,
                reg=REGULARIZATION,
                n_samples=N_SAMPLES,
                sampling_method=SAMPLING_METHOD,
                n_iterations=N_ITERATIONS,
                cg_tol=CG_TOL,
                cg_maxiter=CG_MAXITER,
                use_cem=False,
                cem_interval=5,
                cem_ema_alpha=0.3,
                T_initial=5.0,
                T_final=1.0,
                n_warmup=0,
                lsb_steps=1000,
                lsb_sigma=1.0,
                lsb_delta=1.0,
                fast_anneal_time_ns=FAST_ANNEAL_TIME,
                seed=seed,
                output_dir=OUTPUT_DIR,
            )
            print(f"  rel_error={result['rel_error']:.4f}  t={result['wall_time_s']:.1f}s")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print("\nDone.")

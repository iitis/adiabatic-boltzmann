"""
D-Wave fixed-param benchmark for heisenberg_j1j2_1d.

Runs N and J2 sweeps with a fixed set of seeds using sensible hyperparameters
(no search). Supports pegasus_fast and zephyr backends via --sampler.

Usage (from project root):
    python scripts/dwave_j1j2_bench.py                           # pegasus_fast defaults
    python scripts/dwave_j1j2_bench.py --sampler zephyr
    python scripts/dwave_j1j2_bench.py --sampler zephyr --N 8 12 --j2 0.1 0.5
    python scripts/dwave_j1j2_bench.py --dry-run
"""

import argparse
import itertools
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from hparam_optuna import HAMILTONIAN_REGISTRY, run_trial

HAMILTONIAN      = "heisenberg_j1j2_1d"
N_ITERATIONS     = 150
N_SAMPLES        = 1000
N_HIDDEN_ALPHA   = 2
LEARNING_RATE    = 0.05
REGULARIZATION   = 1e-4
CG_TOL           = 1e-8
CG_MAXITER       = 200
FAST_ANNEAL_TIME = 7.0  # nanoseconds

_DEFAULTS = {
    "pegasus_fast": dict(N=[12],     j2=[0.1, 0.45, 0.55, 0.7]),
    "zephyr":       dict(N=[8, 12],  j2=[0.1, 0.2,  0.3,  0.5,  0.8]),
}


def _parse_args():
    p = argparse.ArgumentParser(
        description="D-Wave fixed-param benchmark for heisenberg_j1j2_1d",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sampler", choices=["pegasus_fast", "zephyr"], default="pegasus_fast",
                   help="D-Wave backend to use")
    p.add_argument("--N",         type=int,   nargs="+", default=None, metavar="N",
                   help="System sizes (default: sampler-specific)")
    p.add_argument("--j2",        type=float, nargs="+", default=None, metavar="J2",
                   help="J2/J1 values (default: sampler-specific)")
    p.add_argument("--seeds",     type=int,   nargs="+", default=[0, 1])
    p.add_argument("--iterations", type=int,  default=N_ITERATIONS)
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    N_values  = args.N  or _DEFAULTS[args.sampler]["N"]
    J2_values = args.j2 or _DEFAULTS[args.sampler]["j2"]

    entry  = HAMILTONIAN_REGISTRY[HAMILTONIAN]
    combos = list(itertools.product(N_values, J2_values, args.seeds))
    total  = len(combos)

    print(f"{args.sampler} benchmark — {HAMILTONIAN}")
    print(f"  N={N_values}  J2={J2_values}  seeds={args.seeds}")
    print(f"  {total} runs × {args.iterations} iters\n")

    for idx, (N, J2, seed) in enumerate(combos, 1):
        n_hidden = int(N_HIDDEN_ALPHA * N)
        phys = {**entry["defaults"], "J2": J2}
        print(f"[{idx}/{total}] N={N}  J2={J2}  seed={seed}  n_hidden={n_hidden}")
        if args.dry_run:
            continue
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
                sampling_method=args.sampler,
                n_iterations=args.iterations,
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
                output_dir=_REPO / "results",
            )
            print(f"  rel_error={result['rel_error']:.4f}  t={result['wall_time_s']:.1f}s")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()

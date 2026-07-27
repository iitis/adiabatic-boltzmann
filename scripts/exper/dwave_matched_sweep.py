#!/usr/bin/env python3
"""
dwave_matched_sweep.py — run D-Wave Pegasus/Zephyr QPU at the same
(lr, reg, n_samples, iterations) cell as scripts/exper/mcmc_matched_sweep.py
(FPGA/VeloxQ/Metropolis/Gibbs/LSB's shared point: lr=0.08, reg=0.05, ns=200,
iter=100), so Figure 10 can add D-Wave to that matched comparison.

QPU access time is a metered, shared, non-reset budget (src/time.json).
Measured live cost is ~0.04-0.065s/call, flat across N=8-128 for both
methods. Two independent caps are enforced before every run, checking
time.json live (never silently skipped):
  DWAVE_BUDGET_MS   absolute cumulative ceiling across all sessions ever.
  SESSION_BUDGET_MS how much NEW QPU time *this invocation* may spend,
                    measured from its own baseline at startup.
Either one tripping aborts cleanly -- same pattern as
scripts/j1j2/j1j2_bench.py's _qpu_budget_exceeded().

Each run's per-iteration sampling_time_s is also checked for outliers
(see _flag_timing_outliers): archived Pegasus/Zephyr data shows occasional
single-iteration spikes 10-100x the run's own median (likely D-Wave-side
background calibration; not reproduced in live testing against a fresh
sampler/embedding, but not something client code can rule out either).
Outliers are flagged, not dropped or altered, so a corrupted TTE median
can be caught before being trusted.

Runs in-process (Trainer/DimodSampler directly, same construction as
scripts/main.py's --sampler dimod path) — never shells out to main.py.

Usage:
    python scripts/exper/dwave_matched_sweep.py
    python scripts/exper/dwave_matched_sweep.py --methods pegasus --seeds 2 --smoke-test
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import jax
jax.config.update("jax_enable_x64", True)
from argparse import Namespace

from helpers import save_results, read_qpu_time_ms
from model import FullyConnectedRBM
from sampler import DimodSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D

DEFAULT_SIZES = [8, 12, 16, 24, 32, 64, 128]
DEFAULT_METHODS = ["pegasus", "zephyr"]
DWAVE_BUDGET_MS = 60 * 60 * 1000  # raised from 30 to 60 min for this sweep (absolute, cumulative ceiling)
SESSION_BUDGET_MS = 25 * 60 * 1000  # this invocation may spend at most 25 min of NEW QPU time
DWAVE_TIME_FILE = Path("time.json")
OUTLIER_FACTOR = 8  # flag an iteration if it's this many times its run's own median


def _require_qpu_time_ms() -> float:
    if not DWAVE_TIME_FILE.exists():
        raise FileNotFoundError(
            f"{DWAVE_TIME_FILE} not found — D-Wave budget tracking file missing. "
            "Create it with {\"time_ms\": 0} or run a D-Wave experiment first."
        )
    return read_qpu_time_ms(DWAVE_TIME_FILE)


def _qpu_budget_exceeded(session_baseline_ms: float) -> bool:
    used = _require_qpu_time_ms()
    if used >= DWAVE_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] {used / 60_000:.2f} min used >= "
            f"{DWAVE_BUDGET_MS / 60_000:.0f} min absolute limit. Aborting."
        )
        return True
    session_spent = used - session_baseline_ms
    if session_spent >= SESSION_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] {session_spent / 60_000:.2f} min spent this session >= "
            f"{SESSION_BUDGET_MS / 60_000:.0f} min session cap. Aborting."
        )
        return True
    return False


def _flag_timing_outliers(history, label):
    st = [v for v in history.get("sampling_time_s", []) if v is not None]
    if len(st) < 2:
        return
    med = sorted(st)[len(st) // 2]
    if med <= 0:
        return
    for i, v in enumerate(st):
        if v >= OUTLIER_FACTOR * med:
            print(f"  [TIMING WARNING] {label} iter {i}: sampling_time_s={v:.3f}s "
                  f"is >= {OUTLIER_FACTOR}x this run's median ({med:.3f}s) — "
                  f"not dropped, but check before trusting this run's TTE.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                   choices=["pegasus", "zephyr"])
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds, 0..seeds-1")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=str(_ROOT / "results"))
    p.add_argument("--smoke-test", action="store_true",
                   help="1 size, 1 seed, 3 iterations — verify the plumbing only")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip (N, method, seed) combos whose result file already exists")
    return p.parse_args()


def run_one(size, method, seed, args):
    ns_args = Namespace(
        model="1d", size=size, h=args.h, rbm="full", n_hidden=size,
        sampler="dimod", sampling_method=method,
        iterations=args.iterations, learning_rate=args.lr,
        regularization=args.reg, n_samples=args.n_samples,
        output_dir=args.output_dir, seed=seed, visualize=False, cem=False,
    )

    out_file = (
        Path(args.output_dir) / "tfim_1d" / str(size) / "dimod" / method /
        f"result_1d_h{args.h}_rbmfull_nh{size}_lr{args.lr}_reg{args.reg}"
        f"_ns{args.n_samples}_seed{seed}_iter{args.iterations}_cem0_sigma1.0.json.gz"
    )
    if args.skip_existing and out_file.exists():
        print(f"  skip (exists): {out_file}")
        return None

    ising = TransverseFieldIsing1D(size, args.h)
    rbm = FullyConnectedRBM(size, size, jax.random.PRNGKey(seed))
    sampler = DimodSampler(method=method)

    trainer_config = {
        "learning_rate": args.lr,
        "n_iterations": args.iterations,
        "n_samples": args.n_samples,
        "regularization": args.reg,
        "seed": seed,
    }
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=ns_args)
    history = trainer.train()
    save_results(ns_args, history, ising, rbm, energy_j=trainer.total_energy_j)
    return history


def main():
    args = parse_args()
    if args.smoke_test:
        args.sizes = args.sizes[:1]
        args.seeds = 1
        args.iterations = 3

    session_baseline_ms = _require_qpu_time_ms()
    print(f"[QPU BUDGET] session baseline: {session_baseline_ms / 60_000:.2f} min already used "
          f"(absolute cap {DWAVE_BUDGET_MS / 60_000:.0f} min, session cap "
          f"+{SESSION_BUDGET_MS / 60_000:.0f} min for this run)")

    for method in args.methods:
        for size in args.sizes:
            for seed in range(args.seeds):
                if _qpu_budget_exceeded(session_baseline_ms):
                    return
                print(f"=== {method} N={size} seed={seed} "
                      f"(QPU used: {_require_qpu_time_ms() / 60_000:.2f} min) ===")
                try:
                    history = run_one(size, method, seed, args)
                    if history is not None:
                        _flag_timing_outliers(history, f"{method} N={size} seed={seed}")
                except RuntimeError as e:
                    if "busclique failed to find a biclique embedding" in str(e):
                        print(f"  SKIPPED size (no embedding exists at this N): {e}")
                        break  # deterministic for this (method, size); every seed would fail identically
                    print(f"  SKIPPED seed (sampling call failed after retries): {e}")
                    continue  # transient (network/timing-field) failure; other seeds may still succeed


if __name__ == "__main__":
    main()

"""
FBM vs RBM hyperparameter search on the J1-J2 1D frustrated Ising chain.

Compares FullBoltzmannMachine (RBM + visible-visible couplings J) against the
plain FullyConnectedRBM baseline across J2 values, learning rates, and hidden
unit counts.  Results land in the same results/j1j2_1d/ tree as the ViT
benchmark so they can be loaded by the same analysis notebooks.

Usage (from project root):
    python scripts/fbm_j1j2_benchmark.py                    # classical FBM + RBM
    python scripts/fbm_j1j2_benchmark.py --mode dwave        # D-Wave only
    python scripts/fbm_j1j2_benchmark.py --mode both         # classical + D-Wave
    python scripts/fbm_j1j2_benchmark.py --ansatz fbm        # FBM only (classical)
    python scripts/fbm_j1j2_benchmark.py --ansatz rbm        # RBM baseline only
    python scripts/fbm_j1j2_benchmark.py --dry-run
    python scripts/fbm_j1j2_benchmark.py --sizes 8 --lrs 0.05 0.1
    python scripts/fbm_j1j2_benchmark.py --j2 0.0 0.5 1.0
    python scripts/fbm_j1j2_benchmark.py --iterations 500
    python scripts/fbm_j1j2_benchmark.py --mode dwave --dwave-methods zephyr
"""

import argparse
import sys
import time
from pathlib import Path
from argparse import Namespace

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)

from model import FullyConnectedRBM, DWaveTopologyRBM, FullBoltzmannMachine
from sampler import ClassicalSampler, DimodSampler
from encoder import Trainer
from ising import J1J2Ising1D
from helpers import save_results, read_qpu_time_ms

# ── Parameter space ───────────────────────────────────────────────────────────

J2_VALUES  = [0.0, 0.1, 0.3, 0.45, 0.55, 0.7, 1.0]
SEEDS      = [1, 42]
J1         = 1.0
H          = 0.5

LR_VALUES  = [0.01, 0.05, 0.1]

REGULARIZATION  = 1e-3
N_SAMPLES       = 1000
SAMPLING_METHOD = "metropolis"

# ── D-Wave parameters ─────────────────────────────────────────────────────────

DWAVE_SIZES            = [8, 16]
DWAVE_LR_VALUES        = [0.1, 0.01]
DWAVE_SAMPLING_METHODS = ["pegasus", "zephyr"]
DWAVE_RBM_TYPES        = ["full", "fullbm"]   # fullbm maps J_vv directly to QUBO edges
DWAVE_REGULARIZATION   = 1e-5

DWAVE_BUDGET_MS  = 75 * 60 * 1000
DWAVE_TIME_FILE  = Path("time.json")


# ── QPU budget helpers ────────────────────────────────────────────────────────


def _require_qpu_time_ms() -> float:
    if not DWAVE_TIME_FILE.exists():
        raise FileNotFoundError(
            f"{DWAVE_TIME_FILE} not found — D-Wave budget tracking file is missing. "
            "Create it with {\"time_ms\": 0} or run a D-Wave experiment first."
        )
    return read_qpu_time_ms(DWAVE_TIME_FILE)


def qpu_budget_exceeded() -> bool:
    used = _require_qpu_time_ms()
    if used >= DWAVE_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] Accumulated QPU time {used / 60_000:.2f} min "
            f">= limit {DWAVE_BUDGET_MS / 60_000:.0f} min. "
            "Aborting remaining D-Wave experiments."
        )
        return True
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_args(
    N: int,
    J2: float,
    seed: int,
    lr: float,
    rbm_type: str,           # "full" | "fullbm"
    iterations: int,
    sampler: str = "custom",
    sampling_method: str = SAMPLING_METHOD,
    regularization: float = REGULARIZATION,
) -> Namespace:
    return Namespace(
        model="j1j2_1d",
        size=N,
        J1=J1,
        J2=J2,
        h=H,
        J=J1,
        delta=1.0,
        alpha=2.0,
        ansatz="rbm",
        rbm=rbm_type,
        n_hidden=N,
        d_model=32,
        n_layers=2,
        n_heads=4,
        patch_size=2,
        sampler=sampler,
        sampling_method=sampling_method,
        n_samples=N_SAMPLES,
        iterations=iterations,
        learning_rate=lr,
        regularization=regularization,
        seed=seed,
        cem=False,
        cem_interval=5,
        sigma=1.0,
        visualize=False,
        output_dir=str(_REPO / "results"),
    )


def _result_path(args: Namespace) -> Path:
    from helpers import _model_subdir, _model_params_str, _ansatz_str
    output_dir = (
        Path(args.output_dir)
        / _model_subdir(args.model)
        / str(args.size)
        / args.sampler
        / args.sampling_method
    )
    fname = (
        f"result"
        f"_{args.model}"
        f"{_model_params_str(args)}"
        f"{_ansatz_str(args)}"
        f"_lr{args.learning_rate}"
        f"_reg{args.regularization}"
        f"_ns{args.n_samples}"
        f"_seed{args.seed}"
        f"_iter{args.iterations}"
        f"_cem{int(args.cem)}"
        f"_sigma{float(args.sigma)}"
        f".json"
    )
    return output_dir / fname


def _build_wave_fn(rbm_type: str, N: int, key):
    if rbm_type == "fullbm":
        return FullBoltzmannMachine(N, N, key)
    elif rbm_type == "full":
        return FullyConnectedRBM(N, N, key)
    else:
        return DWaveTopologyRBM(N, N, key, solver=rbm_type)


# ── Classical run ─────────────────────────────────────────────────────────────


def run_one(
    N: int,
    J2: float,
    seed: int,
    lr: float,
    rbm_type: str,
    iterations: int,
    dry_run: bool = False,
) -> dict | None:
    args = _make_args(N, J2, seed, lr, rbm_type, iterations)
    out  = _result_path(args)

    label = (
        f"N={N:2d}  J2={J2:.2f}  seed={seed:3d}"
        f"  lr={lr}  rbm={rbm_type}"
    )

    if out.exists():
        print(f"  [skip]  {label}  → {out.name}")
        return None

    if dry_run:
        print(f"  [would run]  {label}")
        return None

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  iters={iterations}  ns={N_SAMPLES}  reg={REGULARIZATION}")
    print(f"{'='*70}")

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

    wave_fn = _build_wave_fn(rbm_type, N, model_key)
    print(f"  {wave_fn}")

    ising = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    try:
        exact = ising.exact_ground_energy()
        print(f"  Exact ground energy: {exact:.6f}")
    except NotImplementedError:
        exact = None
        print("  Exact energy: not available")

    sampler = ClassicalSampler(method=SAMPLING_METHOD)
    sampler._key = sampler_key

    config = {
        "learning_rate": lr,
        "n_iterations": iterations,
        "n_samples": N_SAMPLES,
        "regularization": REGULARIZATION,
        "seed": seed,
    }

    t0      = time.perf_counter()
    trainer = Trainer(wave_fn, ising, sampler, config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    error   = abs(final_E - exact) if exact is not None else None

    print(f"\n  Final energy : {final_E:.6f}")
    if exact is not None:
        print(f"  Exact energy : {exact:.6f}")
        print(f"  Error        : {error:.6f}  ({100 * error / abs(exact):.2f}%)")
    print(f"  Wall time    : {elapsed:.1f}s")

    save_results(args, history, ising, rbm=wave_fn)
    return {"final_energy": final_E, "exact_energy": exact, "error": error}


# ── D-Wave run ────────────────────────────────────────────────────────────────


def run_dwave_one(
    N: int,
    J2: float,
    seed: int,
    lr: float,
    sampling_method: str,
    rbm_type: str,
    iterations: int,
    dry_run: bool = False,
) -> dict | None:
    args = _make_args(
        N, J2, seed, lr, rbm_type, iterations,
        sampler="dimod",
        sampling_method=sampling_method,
        regularization=DWAVE_REGULARIZATION,
    )
    out = _result_path(args)

    label = (
        f"N={N:2d}  J2={J2:.2f}  seed={seed:3d}"
        f"  lr={lr}  method={sampling_method}  rbm={rbm_type}"
    )

    if out.exists():
        print(f"  [skip]  {label}  → {out.name}")
        return None

    if dry_run:
        print(f"  [would run]  {label}")
        return None

    print(f"\n{'='*70}")
    print(f"  [D-Wave] {label}")
    print(f"  iters={iterations}  ns={N_SAMPLES}  reg={DWAVE_REGULARIZATION}")
    print(f"{'='*70}")

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    wave_fn = _build_wave_fn(rbm_type, N, model_key)
    print(f"  {wave_fn}")

    ising = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    try:
        exact = ising.exact_ground_energy()
        print(f"  Exact ground energy: {exact:.6f}")
    except NotImplementedError:
        exact = None
        print("  Exact energy: not available")

    sampler = DimodSampler(method=sampling_method)

    config = {
        "learning_rate": lr,
        "n_iterations": iterations,
        "n_samples": N_SAMPLES,
        "regularization": DWAVE_REGULARIZATION,
        "seed": seed,
    }

    t0      = time.perf_counter()
    trainer = Trainer(wave_fn, ising, sampler, config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    error   = abs(final_E - exact) if exact is not None else None

    print(f"\n  Final energy : {final_E:.6f}")
    if exact is not None:
        print(f"  Exact energy : {exact:.6f}")
        print(f"  Error        : {error:.6f}  ({100 * error / abs(exact):.2f}%)")
    print(f"  Wall time    : {elapsed:.1f}s")

    save_results(args, history, ising, rbm=wave_fn)
    return {"final_energy": final_E, "exact_energy": exact, "error": error}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="FBM vs RBM hyperparameter search on J1-J2 1D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["classical", "dwave", "both"], default="classical",
        help="Which experiments to run (default: classical)",
    )
    # Classical options
    parser.add_argument(
        "--ansatz", choices=["fbm", "rbm", "both"], default="both",
        help="Which ansatz to benchmark in classical mode (default: both)",
    )
    parser.add_argument("--sizes",      type=int,   nargs="+", default=[8, 16])
    parser.add_argument("--seeds",      type=int,   nargs="+", default=SEEDS)
    parser.add_argument("--j2",         type=float, nargs="+", default=J2_VALUES)
    parser.add_argument("--lrs",        type=float, nargs="+", default=LR_VALUES)
    parser.add_argument("--iterations", type=int,              default=300,
                        help="Training iterations for classical runs (default: 300)")
    # D-Wave options
    parser.add_argument("--dwave-sizes",      type=int,   nargs="+", default=DWAVE_SIZES)
    parser.add_argument("--dwave-lrs",        type=float, nargs="+", default=DWAVE_LR_VALUES)
    parser.add_argument("--dwave-methods",    type=str,   nargs="+", default=DWAVE_SAMPLING_METHODS,
                        choices=["pegasus", "zephyr"])
    parser.add_argument("--dwave-rbm",        type=str,   nargs="+", default=DWAVE_RBM_TYPES,
                        choices=["full", "fullbm", "pegasus", "zephyr"])
    parser.add_argument("--dwave-iterations", type=int,              default=300,
                        help="Training iterations for D-Wave runs (default: 300)")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    sizes = sorted(cli.sizes)
    seeds = sorted(cli.seeds)
    j2s   = sorted(cli.j2)
    lrs   = sorted(cli.lrs)

    run_classical = cli.mode in ("classical", "both")
    run_dwave     = cli.mode in ("dwave", "both")

    # ── Classical sweep ───────────────────────────────────────────────────────

    if run_classical:
        rbm_types = []
        if cli.ansatz in ("rbm", "both"):
            rbm_types.append("full")
        if cli.ansatz in ("fbm", "both"):
            rbm_types.append("fullbm")

        total = len(sizes) * len(rbm_types) * len(lrs) * len(j2s) * len(seeds)
        print(f"FBM vs RBM J1-J2 classical benchmark  (n_hidden = N)")
        print(f"  Ansatz     : {rbm_types}")
        print(f"  Sizes      : {sizes}")
        print(f"  J2         : {j2s}")
        print(f"  Seeds      : {seeds}")
        print(f"  LRs        : {lrs}")
        print(f"  Iterations : {cli.iterations}")
        print(f"  Total      : {total} runs\n")

        done = skipped = failed = 0

        for N in sizes:
            for rbm_type in rbm_types:
                for lr in lrs:
                    for J2 in j2s:
                        for seed in seeds:
                            try:
                                result = run_one(
                                    N, J2, seed, lr, rbm_type,
                                    iterations=cli.iterations,
                                    dry_run=cli.dry_run,
                                )
                                if result is None:
                                    skipped += 1
                                else:
                                    done += 1
                            except Exception as e:
                                print(
                                    f"\n  [ERROR] N={N} J2={J2} seed={seed} "
                                    f"lr={lr} rbm={rbm_type}: {e}"
                                )
                                failed += 1

        print(f"\n{'='*70}")
        print(f"Classical done: {done}  Skipped: {skipped}  Failed: {failed}")

    # ── D-Wave sweep ──────────────────────────────────────────────────────────

    if run_dwave:
        dwave_sizes   = sorted(cli.dwave_sizes)
        dwave_lrs     = sorted(cli.dwave_lrs)
        dwave_methods = cli.dwave_methods
        dwave_rbm     = cli.dwave_rbm
        dwave_j2s     = j2s
        dwave_seeds   = seeds

        try:
            used_ms = _require_qpu_time_ms()
        except Exception as e:
            print(f"[QPU BUDGET ERROR] Cannot read {DWAVE_TIME_FILE}: {e} — aborting D-Wave sweep.")
            return

        print(f"\n{'='*70}")
        print(f"FBM/RBM J1-J2 D-Wave benchmark  (n_hidden = N)")
        print(f"  Sizes      : {dwave_sizes}")
        print(f"  J2         : {dwave_j2s}")
        print(f"  Seeds      : {dwave_seeds}")
        print(f"  LRs        : {dwave_lrs}")
        print(f"  Methods    : {dwave_methods}")
        print(f"  RBM types  : {dwave_rbm}")
        print(f"  Iterations : {cli.dwave_iterations}")
        print(
            f"  QPU budget : {DWAVE_BUDGET_MS / 60_000:.0f} min total  |  "
            f"used: {used_ms / 60_000:.2f} min  |  "
            f"remaining: {max(0.0, DWAVE_BUDGET_MS / 60_000 - used_ms / 60_000):.2f} min"
        )

        dwave_done = dwave_skipped = dwave_failed = 0

        for N in dwave_sizes:
            for method in dwave_methods:
                for rbm_type in dwave_rbm:
                    # topology RBMs must match their sampler
                    if rbm_type not in ("full", "fullbm") and rbm_type != method:
                        continue
                    for lr in dwave_lrs:
                        for J2 in dwave_j2s:
                            for seed in dwave_seeds:
                                try:
                                    if qpu_budget_exceeded():
                                        print("  Aborting remaining D-Wave experiments.")
                                        print(
                                            f"\nD-Wave done: {dwave_done}  "
                                            f"Skipped: {dwave_skipped}  "
                                            f"Failed: {dwave_failed}"
                                        )
                                        return
                                except Exception as e:
                                    print(
                                        f"[QPU BUDGET ERROR] Cannot read {DWAVE_TIME_FILE}: "
                                        f"{e} — aborting."
                                    )
                                    return

                                try:
                                    result = run_dwave_one(
                                        N, J2, seed, lr, method, rbm_type,
                                        iterations=cli.dwave_iterations,
                                        dry_run=cli.dry_run,
                                    )
                                    if result is None:
                                        dwave_skipped += 1
                                    else:
                                        dwave_done += 1
                                except Exception as e:
                                    print(
                                        f"\n  [D-Wave ERROR] N={N} J2={J2} seed={seed} "
                                        f"lr={lr} method={method} rbm={rbm_type}: {e}"
                                    )
                                    dwave_failed += 1

        print(f"\n{'='*70}")
        print(
            f"D-Wave done: {dwave_done}  Skipped: {dwave_skipped}  Failed: {dwave_failed}"
        )
        print(f"Total QPU time used: {_require_qpu_time_ms() / 60_000:.2f} min")


if __name__ == "__main__":
    main()

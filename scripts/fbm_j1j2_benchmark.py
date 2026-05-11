"""
FBM vs RBM hyperparameter search on the J1-J2 1D frustrated Ising chain.

Compares FullBoltzmannMachine (RBM + visible-visible couplings J) against the
plain FullyConnectedRBM baseline across J2 values, learning rates, and hidden
unit counts.  Results land in the same results/j1j2_1d/ tree as the ViT
benchmark so they can be loaded by the same analysis notebooks.

Usage (from project root):
    python scripts/fbm_j1j2_benchmark.py                   # FBM + RBM baseline
    python scripts/fbm_j1j2_benchmark.py --ansatz fbm       # FBM only
    python scripts/fbm_j1j2_benchmark.py --ansatz rbm       # RBM baseline only
    python scripts/fbm_j1j2_benchmark.py --dry-run
    python scripts/fbm_j1j2_benchmark.py --sizes 8 --lrs 0.05 0.1
    python scripts/fbm_j1j2_benchmark.py --j2 0.0 0.5 1.0
    python scripts/fbm_j1j2_benchmark.py --nh-ratios 1 2
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

from model import FullyConnectedRBM, FullBoltzmannMachine
from sampler import ClassicalSampler
from encoder import Trainer
from ising import J1J2Ising1D
from helpers import save_results

# ── Parameter space ───────────────────────────────────────────────────────────

J2_VALUES  = [0.0, 0.1, 0.3, 0.45, 0.55, 0.7, 1.0]
SEEDS      = [1, 42]
J1         = 1.0
H          = 0.5

LR_VALUES  = [0.01, 0.05, 0.1]

# n_hidden = NH_RATIO * N.  ratio=1 matches visible count; ratio=2 gives more
# expressive power at the cost of extra J parameters: N*(N-1)/2 grows as N².
NH_RATIOS  = [1, 2]

_ITERATIONS = {8: 300, 16: 300}

REGULARIZATION = 1e-3
N_SAMPLES      = 1000
SAMPLING_METHOD = "metropolis"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_args(
    N: int,
    J2: float,
    seed: int,
    lr: float,
    n_hidden: int,
    rbm_type: str,          # "full" | "fullbm"
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
        n_hidden=n_hidden,
        # ViT fields left at defaults so _ansatz_str / _result_path work
        d_model=32,
        n_layers=2,
        n_heads=4,
        patch_size=2,
        sampler="custom",
        sampling_method=SAMPLING_METHOD,
        n_samples=N_SAMPLES,
        iterations=_ITERATIONS[N],
        learning_rate=lr,
        regularization=REGULARIZATION,
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


# ── Single run ────────────────────────────────────────────────────────────────


def run_one(
    N: int,
    J2: float,
    seed: int,
    lr: float,
    n_hidden: int,
    rbm_type: str,
    dry_run: bool = False,
) -> dict | None:
    args = _make_args(N, J2, seed, lr, n_hidden, rbm_type)
    out  = _result_path(args)

    label = (
        f"N={N:2d}  J2={J2:.2f}  seed={seed:3d}"
        f"  lr={lr}  nh={n_hidden}  rbm={rbm_type}"
    )

    if out.exists():
        print(f"  [skip]  {label}  → {out.name}")
        return None

    if dry_run:
        print(f"  [would run]  {label}")
        return None

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  iters={args.iterations}  ns={N_SAMPLES}  reg={REGULARIZATION}")
    print(f"{'='*70}")

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

    if rbm_type == "fullbm":
        wave_fn = FullBoltzmannMachine(N, n_hidden, model_key)
    else:
        wave_fn = FullyConnectedRBM(N, n_hidden, model_key)
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
        "n_iterations": args.iterations,
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="FBM vs RBM hyperparameter search on J1-J2 1D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ansatz", choices=["fbm", "rbm", "both"], default="both",
        help="Which ansatz to benchmark (default: both)",
    )
    parser.add_argument("--sizes",     type=int,   nargs="+", default=[8, 16])
    parser.add_argument("--seeds",     type=int,   nargs="+", default=SEEDS)
    parser.add_argument("--j2",        type=float, nargs="+", default=J2_VALUES)
    parser.add_argument("--lrs",       type=float, nargs="+", default=LR_VALUES)
    parser.add_argument("--nh-ratios", type=int,   nargs="+", default=NH_RATIOS,
                        help="n_hidden = ratio * N (default: 1 2)")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    sizes     = sorted(cli.sizes)
    seeds     = sorted(cli.seeds)
    j2s       = sorted(cli.j2)
    lrs       = sorted(cli.lrs)
    nh_ratios = sorted(cli.nh_ratios)

    rbm_types = []
    if cli.ansatz in ("rbm", "both"):
        rbm_types.append("full")
    if cli.ansatz in ("fbm", "both"):
        rbm_types.append("fullbm")

    unsupported = [N for N in sizes if N not in _ITERATIONS]
    if unsupported:
        print(f"[warn] No iteration budget for N={unsupported}; skipping those sizes.")
        sizes = [N for N in sizes if N in _ITERATIONS]

    total = len(sizes) * len(rbm_types) * len(nh_ratios) * len(lrs) * len(j2s) * len(seeds)
    print(f"FBM vs RBM J1-J2 benchmark")
    print(f"  Ansatz    : {rbm_types}")
    print(f"  Sizes     : {sizes}")
    print(f"  J2        : {j2s}")
    print(f"  Seeds     : {seeds}")
    print(f"  LRs       : {lrs}")
    print(f"  NH ratios : {nh_ratios}")
    print(f"  Total     : {total} runs\n")

    done = skipped = failed = 0

    for N in sizes:
        for rbm_type in rbm_types:
            for nh_ratio in nh_ratios:
                n_hidden = N * nh_ratio
                for lr in lrs:
                    for J2 in j2s:
                        for seed in seeds:
                            try:
                                result = run_one(
                                    N, J2, seed, lr, n_hidden, rbm_type,
                                    dry_run=cli.dry_run,
                                )
                                if result is None:
                                    skipped += 1
                                else:
                                    done += 1
                            except Exception as e:
                                print(
                                    f"\n  [ERROR] N={N} J2={J2} seed={seed} "
                                    f"lr={lr} nh={n_hidden} rbm={rbm_type}: {e}"
                                )
                                failed += 1

    print(f"\n{'='*70}")
    print(f"Done: {done}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()

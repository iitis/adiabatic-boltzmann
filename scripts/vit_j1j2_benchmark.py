"""
ViT benchmark on the J1-J2 1D frustrated Ising chain.

Runs the ViT wave function ansatz on the same (N, J2, seed) combinations
that exist in results/j1j2_1d/ from the RBM baseline, so the two can be
compared directly.

Usage (from project root):
    python scripts/vit_j1j2_benchmark.py
    python scripts/vit_j1j2_benchmark.py --dry-run
    python scripts/vit_j1j2_benchmark.py --sizes 8 16
    python scripts/vit_j1j2_benchmark.py --sizes 32 --seeds 1 42

The script skips any (N, J2, seed) for which a ViT result file already exists.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from argparse import Namespace

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from model_vit import ViTWaveFunction
from sampler import GenericClassicalSampler
from encoder_generic import TrainerGeneric
from ising import J1J2Ising1D
from helpers import save_results

# ── Parameter space (mirrors the RBM baseline) ───────────────────────────────

J2_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS     = [1, 7, 42, 123]
J1        = 1.0
H         = 0.5

# Iterations to match RBM baseline
_ITERATIONS = {8: 300, 16: 300, 32: 500}

# ViT architecture scaled to system size.
# patch_size=2 → n_patches = N//2 tokens fed into attention.
# d_model and n_layers grow with N to keep expressivity comparable to n_hidden=N.
_VIT_CONFIG = {
    8:  dict(d_model=16, n_heads=2, n_layers=2, patch_size=2),
    16: dict(d_model=32, n_heads=4, n_layers=2, patch_size=2),
    32: dict(d_model=32, n_heads=4, n_layers=3, patch_size=4),
}

# Training hyperparameters
LR           = 0.05    # ViT needs a smaller step than RBM (0.1)
REGULARIZATION = 0.001
N_SAMPLES    = 1000
N_WARMUP     = 20      # MH warmup steps (in units of N); fewer than RBM because
                        # each ViT forward pass is ~10× more expensive than psi_ratio


# ── Result path helpers ───────────────────────────────────────────────────────

def _make_args(N: int, J2: float, seed: int, vit_cfg: dict) -> Namespace:
    """Build the argparse.Namespace that helpers.save_results() expects."""
    return Namespace(
        model="j1j2_1d",
        size=N,
        J1=J1,
        J2=J2,
        h=H,
        J=J1,
        delta=1.0,
        alpha=2.0,
        ansatz="vit",
        rbm="full",          # unused for ViT but needed by some helpers
        n_hidden=None,
        d_model=vit_cfg["d_model"],
        n_layers=vit_cfg["n_layers"],
        n_heads=vit_cfg["n_heads"],
        patch_size=vit_cfg["patch_size"],
        sampler="custom",
        sampling_method="metropolis",
        n_samples=N_SAMPLES,
        iterations=_ITERATIONS[N],
        learning_rate=LR,
        regularization=REGULARIZATION,
        seed=seed,
        cem=False,
        cem_interval=5,
        sigma=1.0,
        visualize=False,
        output_dir=str(_REPO / "results"),
    )


def _result_path(args: Namespace) -> Path:
    """Reconstruct the output file path without actually running training."""
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


# ── Single training run ───────────────────────────────────────────────────────

def run_one(N: int, J2: float, seed: int, dry_run: bool = False) -> dict | None:
    vit_cfg = _VIT_CONFIG[N]
    args    = _make_args(N, J2, seed, vit_cfg)
    out     = _result_path(args)

    label = f"N={N:2d}  J2={J2:.2f}  seed={seed:3d}"

    if out.exists():
        print(f"  [skip]  {label}  → {out.name}")
        return None

    if dry_run:
        print(f"  [would run]  {label}")
        return None

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  ViT: d_model={vit_cfg['d_model']}  n_layers={vit_cfg['n_layers']}"
          f"  n_heads={vit_cfg['n_heads']}  patch_size={vit_cfg['patch_size']}")
    print(f"  iters={args.iterations}  ns={N_SAMPLES}  lr={LR}")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(seed)
    key, vit_key, sampler_key = jax.random.split(key, 3)

    vit = ViTWaveFunction(
        n_visible=N,
        n_layers=vit_cfg["n_layers"],
        d_model=vit_cfg["d_model"],
        n_heads=vit_cfg["n_heads"],
        patch_size=vit_cfg["patch_size"],
        key=vit_key,
        geometry="1d",
    )
    print(f"  {vit}")

    ising = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    try:
        exact = ising.exact_ground_energy()
        print(f"  Exact ground energy: {exact:.6f}")
    except NotImplementedError:
        exact = None
        print("  Exact energy: not available")

    sampler = GenericClassicalSampler(n_warmup=N_WARMUP, n_sweeps=1)
    sampler._key = sampler_key

    config = {
        "learning_rate": LR,
        "n_iterations": args.iterations,
        "n_samples": N_SAMPLES,
        "regularization": REGULARIZATION,
        "seed": seed,
    }

    t0 = time.perf_counter()
    trainer = TrainerGeneric(vit, ising, sampler, config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    error = abs(final_E - exact) if exact is not None else None

    print(f"\n  Final energy : {final_E:.6f}")
    if exact is not None:
        print(f"  Exact energy : {exact:.6f}")
        print(f"  Error        : {error:.6f}")
    print(f"  Wall time    : {elapsed:.1f}s")

    save_results(args, history, ising, rbm=None)
    return {"final_energy": final_E, "exact_energy": exact, "error": error}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ViT J1-J2 benchmark")
    parser.add_argument("--sizes",  type=int,   nargs="+", default=[8, 16, 32],
                        help="System sizes to run (default: 8 16 32)")
    parser.add_argument("--seeds",  type=int,   nargs="+", default=SEEDS,
                        help="Random seeds (default: 1 7 42 123)")
    parser.add_argument("--j2",     type=float, nargs="+", default=J2_VALUES,
                        help="J2 values to sweep (default: all 13)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    cli = parser.parse_args()

    sizes = sorted(cli.sizes)
    seeds = sorted(cli.seeds)
    j2s   = sorted(cli.j2)

    total = len(sizes) * len(j2s) * len(seeds)
    print(f"ViT J1-J2 benchmark")
    print(f"  Sizes : {sizes}")
    print(f"  J2    : {j2s}")
    print(f"  Seeds : {seeds}")
    print(f"  Total : {total} runs\n")

    done = 0
    skipped = 0
    failed  = 0

    for N in sizes:
        if N not in _VIT_CONFIG:
            print(f"[warn] No ViT config defined for N={N}, skipping.")
            continue
        for J2 in j2s:
            for seed in seeds:
                try:
                    result = run_one(N, J2, seed, dry_run=cli.dry_run)
                    if result is None:
                        skipped += 1
                    else:
                        done += 1
                except Exception as e:
                    print(f"\n  [ERROR] N={N} J2={J2} seed={seed}: {e}")
                    failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {done}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()

"""
ViT + D-Wave RBM hyperparameter search on the J1-J2 1D frustrated Ising chain.

Sweeps learning rate and patch size to find the best ViT configuration, and
optionally runs D-Wave (pegasus/zephyr) RBM experiments on the same model for
direct comparison.  Results land in the same results/j1j2_1d/ tree, keyed by
the full hyperparameter set so nothing is overwritten.

Usage (from project root):
    python scripts/vit_j1j2_benchmark.py                   # ViT only
    python scripts/vit_j1j2_benchmark.py --mode dwave       # D-Wave only
    python scripts/vit_j1j2_benchmark.py --mode both        # ViT + D-Wave
    python scripts/vit_j1j2_benchmark.py --dry-run
    python scripts/vit_j1j2_benchmark.py --sizes 8 --lrs 0.05 0.1
    python scripts/vit_j1j2_benchmark.py --patch-sizes 1
    python scripts/vit_j1j2_benchmark.py --mode dwave --dwave-methods zephyr
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

from model_vit import ViTWaveFunction
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import GenericClassicalSampler, DimodSampler, DWaveProposalSampler
from encoder_generic import TrainerGeneric
from encoder import Trainer
from ising import J1J2Ising1D
from helpers import save_results, read_qpu_time_ms

# ── Parameter space ───────────────────────────────────────────────────────────

J2_VALUES   = [0.0, 0.1, 0.3, 0.45, 0.55, 0.7, 1.0]
SEEDS       = [1, 42]
J1          = 1.0
H           = 0.5

# Hyperparameter axes being searched
LR_VALUES   = [0.01, 0.05, 0.01, 0.05, 0.1]

# patch_size=1: each spin its own token — attention directly learns NN and NNN
#   correlations, fully general for the J1-J2 chain.
# patch_size=2: NN pairs as tokens — J1 bond lives inside a patch, J2 connects
#   adjacent patches, a natural coarsening for this Hamiltonian.
# Both are physically motivated; patch_size>2 conflates spins too aggressively.
PATCH_SIZES = [1, 2]

_ITERATIONS = {8: 300, 16: 300, 32: 300}

# d_model/n_heads/n_layers fixed per N; only patch_size varies across the sweep
# so that LR and patch_size effects can be disentangled.
_VIT_BASE = {
    8:  dict(d_model=16, n_heads=2, n_layers=2),
    16: dict(d_model=32, n_heads=4, n_layers=2),
    32: dict(d_model=64, n_heads=4, n_layers=3),
}

REGULARIZATION = 0.001
N_SAMPLES      = 1000
N_WARMUP       = 20

# ── D-Wave parameters ─────────────────────────────────────────────────────────

DWAVE_SIZES          = [8, 16]            # chain lengths supported by QPU embedding
DWAVE_LR_VALUES      = [0.1, 0.01]
DWAVE_SAMPLING_METHODS = ["pegasus", "zephyr"]
DWAVE_RBM_TYPES      = ["full"]           # "full", "pegasus", "zephyr"
DWAVE_N_ITERATIONS   = 300
DWAVE_REGULARIZATION = 1e-5

# QPU budget — cumulative across all sessions, never reset
DWAVE_BUDGET_MS  = 75 * 60 * 1000        # 75 minutes in milliseconds
DWAVE_TIME_FILE  = Path("time.json")      # path relative to cwd (same as DimodSampler)


# ── QPU budget helpers ────────────────────────────────────────────────────────


def _require_qpu_time_ms() -> float:
    """Return accumulated QPU access time in ms.

    Raises FileNotFoundError if time.json does not exist — a missing file is not
    a valid starting state for D-Wave runs; it means budget tracking is broken.
    Raises on any other read/parse failure (no silent fallback).
    """
    if not DWAVE_TIME_FILE.exists():
        raise FileNotFoundError(
            f"{DWAVE_TIME_FILE} not found — D-Wave budget tracking file is missing. "
            "Create it with {{\"time_ms\": 0}} or run a D-Wave experiment first."
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


# ── ViT helpers ───────────────────────────────────────────────────────────────

def _make_args(
    N: int, J2: float, seed: int, lr: float, patch_size: int,
    sampler: str = "custom", sampling_method: str = "metropolis",
) -> Namespace:
    base = _VIT_BASE[N]
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
        rbm="full",
        n_hidden=None,
        d_model=base["d_model"],
        n_layers=base["n_layers"],
        n_heads=base["n_heads"],
        patch_size=patch_size,
        sampler=sampler,
        sampling_method=sampling_method,
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

_dwave_result_path = _result_path


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(
    N: int, J2: float, seed: int, lr: float, patch_size: int,
    dry_run: bool = False,
    vit_sampler: str = "metropolis",
    dwave_method: str = "pegasus",
    aux_rbm_hidden: int | None = None,
    rbm_lr: float = 0.01,
) -> dict | None:
    base = _VIT_BASE[N]

    sampler_key_str = "metropolis" if vit_sampler == "metropolis" else dwave_method
    sampler_name    = "custom"     if vit_sampler == "metropolis" else "dimod"
    args = _make_args(N, J2, seed, lr, patch_size, sampler_name, sampler_key_str)
    out  = _result_path(args)

    label = (
        f"N={N:2d}  J2={J2:.2f}  seed={seed:3d}"
        f"  lr={lr}  ph={patch_size}  sampler={vit_sampler}"
    )

    if out.exists():
        print(f"  [skip]  {label}  → {out.name}")
        return None

    if dry_run:
        print(f"  [would run]  {label}")
        return None

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  ViT: d_model={base['d_model']}  n_layers={base['n_layers']}"
          f"  n_heads={base['n_heads']}  patch_size={patch_size}")
    print(f"  iters={args.iterations}  ns={N_SAMPLES}  reg={REGULARIZATION}")
    print(f"{'='*70}")

    key = jax.random.PRNGKey(seed)
    key, vit_key, sampler_key = jax.random.split(key, 3)

    vit = ViTWaveFunction(
        n_visible=N,
        n_layers=base["n_layers"],
        d_model=base["d_model"],
        n_heads=base["n_heads"],
        patch_size=patch_size,
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

    if vit_sampler == "metropolis":
        sampler = GenericClassicalSampler(n_warmup=N_WARMUP, n_sweeps=1)
        sampler._key = sampler_key
    elif vit_sampler == "dwave-mh":
        n_aux = aux_rbm_hidden if aux_rbm_hidden is not None else N
        sampler = DWaveProposalSampler(
            n_visible=N,
            n_hidden=n_aux,
            key=sampler_key,
            dwave_method=dwave_method,
            rbm_lr=rbm_lr,
        )
        print(f"  Aux RBM: n_hidden={n_aux}  dwave_method={dwave_method}  rbm_lr={rbm_lr}")
    else:
        raise ValueError(f"Unknown vit_sampler: {vit_sampler!r}")

    config = {
        "learning_rate": lr,
        "n_iterations": args.iterations,
        "n_samples": N_SAMPLES,
        "regularization": REGULARIZATION,
        "seed": seed,
    }

    t0      = time.perf_counter()
    trainer = TrainerGeneric(vit, ising, sampler, config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    error   = abs(final_E - exact) if exact is not None else None

    print(f"\n  Final energy : {final_E:.6f}")
    if exact is not None:
        print(f"  Exact energy : {exact:.6f}")
        print(f"  Error        : {error:.6f}")
    print(f"  Wall time    : {elapsed:.1f}s")

    save_results(args, history, ising, rbm=None)
    return {"final_energy": final_E, "exact_energy": exact, "error": error}


# ── D-Wave run ────────────────────────────────────────────────────────────────


def _make_dwave_args(
    N: int, J2: float, seed: int, lr: float,
    sampling_method: str, rbm_type: str,
) -> Namespace:
    n_hidden = N
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
        sampler="dimod",
        sampling_method=sampling_method,
        n_samples=N_SAMPLES,
        iterations=DWAVE_N_ITERATIONS,
        learning_rate=lr,
        regularization=DWAVE_REGULARIZATION,
        seed=seed,
        cem=False,
        cem_interval=5,
        sigma=1.0,
        visualize=False,
        output_dir=str(_REPO / "results"),
    )


def run_dwave_one(
    N: int, J2: float, seed: int, lr: float,
    sampling_method: str, rbm_type: str, dry_run: bool = False,
) -> dict | None:
    args = _make_dwave_args(N, J2, seed, lr, sampling_method, rbm_type)
    out  = _dwave_result_path(args)

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
    print(f"  iters={DWAVE_N_ITERATIONS}  ns={N_SAMPLES}  reg={DWAVE_REGULARIZATION}")
    print(f"{'='*70}")

    key = jax.random.PRNGKey(seed)
    key, rbm_key = jax.random.split(key)

    n_hidden = N
    if rbm_type == "full":
        rbm = FullyConnectedRBM(N, n_hidden, rbm_key)
    else:
        rbm = DWaveTopologyRBM(N, n_hidden, rbm_key, solver=rbm_type)

    ising = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    try:
        exact = ising.exact_ground_energy()
        print(f"  Exact ground energy: {exact:.6f}")
    except NotImplementedError:
        exact = None
        print("  Exact energy: not available")

    sampler = DimodSampler(method=sampling_method)

    trainer_config = dict(
        learning_rate=lr,
        n_iterations=DWAVE_N_ITERATIONS,
        n_samples=N_SAMPLES,
        regularization=DWAVE_REGULARIZATION,
        seed=seed,
    )

    t0      = time.perf_counter()
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    error   = abs(final_E - exact) if exact is not None else None

    print(f"\n  Final energy : {final_E:.6f}")
    if exact is not None:
        print(f"  Exact energy : {exact:.6f}")
        print(f"  Error        : {error:.6f}")
    print(f"  Wall time    : {elapsed:.1f}s")

    save_results(args, history, ising, rbm=rbm)
    return {"final_energy": final_E, "exact_energy": exact, "error": error}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ViT + D-Wave RBM hyperparameter search on J1-J2 1D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["vit", "dwave", "both"], default="vit",
        help="Which experiments to run (default: vit)",
    )
    # ViT options
    parser.add_argument("--sizes",       type=int,   nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds",       type=int,   nargs="+", default=SEEDS)
    parser.add_argument("--j2",          type=float, nargs="+", default=J2_VALUES)
    parser.add_argument("--lrs",         type=float, nargs="+", default=LR_VALUES)
    parser.add_argument("--patch-sizes", type=int,   nargs="+", default=PATCH_SIZES)
    parser.add_argument(
        "--vit-sampler", choices=["metropolis", "dwave-mh"], default="metropolis",
        help="Sampler for ViT runs: classical MH or D-Wave MH proposal (default: metropolis)",
    )
    parser.add_argument(
        "--vit-dwave-method", choices=["pegasus", "zephyr"], default="pegasus",
        help="D-Wave QPU to use when --vit-sampler=dwave-mh (default: pegasus)",
    )
    parser.add_argument(
        "--aux-rbm-hidden", type=int, default=None,
        help="Hidden units in auxiliary RBM for dwave-mh (default: N, one per visible)",
    )
    parser.add_argument(
        "--rbm-lr", type=float, default=0.01,
        help="Learning rate for online CD update of auxiliary RBM (default: 0.01)",
    )
    # D-Wave options
    parser.add_argument("--dwave-sizes",   type=int,   nargs="+", default=DWAVE_SIZES)
    parser.add_argument("--dwave-lrs",     type=float, nargs="+", default=DWAVE_LR_VALUES)
    parser.add_argument("--dwave-methods", type=str,   nargs="+", default=DWAVE_SAMPLING_METHODS,
                        choices=["pegasus", "zephyr"])
    parser.add_argument("--dwave-rbm",     type=str,   nargs="+", default=DWAVE_RBM_TYPES,
                        choices=["full", "pegasus", "zephyr"])
    parser.add_argument("--dry-run",     action="store_true")
    cli = parser.parse_args()

    sizes       = sorted(cli.sizes)
    seeds       = sorted(cli.seeds)
    j2s         = sorted(cli.j2)
    lrs         = sorted(cli.lrs)
    patch_sizes = sorted(cli.patch_sizes)

    run_vit   = cli.mode in ("vit", "both")
    run_dwave = cli.mode in ("dwave", "both")

    # ── ViT sweep ────────────────────────────────────────────────────────────

    if run_vit:
        vit_sampler     = cli.vit_sampler
        vit_dwave_method = cli.vit_dwave_method
        aux_rbm_hidden  = cli.aux_rbm_hidden
        rbm_lr          = cli.rbm_lr

        if vit_sampler == "dwave-mh":
            try:
                used_ms = _require_qpu_time_ms()
            except Exception as e:
                print(f"[QPU BUDGET ERROR] {e} — aborting.")
                return
            print(
                f"  QPU budget : {DWAVE_BUDGET_MS / 60_000:.0f} min total  |  "
                f"used: {used_ms / 60_000:.2f} min  |  "
                f"remaining: {max(0.0, DWAVE_BUDGET_MS / 60_000 - used_ms / 60_000):.2f} min"
            )

        total = len(sizes) * len(j2s) * len(seeds) * len(lrs) * len(patch_sizes)
        print(f"ViT J1-J2 hyperparameter search  (sampler={vit_sampler})")
        print(f"  Sizes       : {sizes}")
        print(f"  J2          : {j2s}")
        print(f"  Seeds       : {seeds}")
        print(f"  LRs         : {lrs}")
        print(f"  Patch sizes : {patch_sizes}")
        print(f"  Total       : {total} runs\n")

        done = skipped = failed = 0

        for N in sizes:
            if N not in _VIT_BASE:
                print(f"[warn] No ViT config for N={N}, skipping.")
                continue
            for patch_size in patch_sizes:
                for lr in lrs:
                    for J2 in j2s:
                        for seed in seeds:
                            if vit_sampler == "dwave-mh":
                                try:
                                    if qpu_budget_exceeded():
                                        print("  Aborting remaining ViT+D-Wave experiments.")
                                        print(f"\nViT done: {done}  Skipped: {skipped}  Failed: {failed}")
                                        return
                                except Exception as e:
                                    print(f"[QPU BUDGET ERROR] {e} — aborting.")
                                    return

                            try:
                                result = run_one(
                                    N, J2, seed, lr, patch_size,
                                    dry_run=cli.dry_run,
                                    vit_sampler=vit_sampler,
                                    dwave_method=vit_dwave_method,
                                    aux_rbm_hidden=aux_rbm_hidden,
                                    rbm_lr=rbm_lr,
                                )
                                if result is None:
                                    skipped += 1
                                else:
                                    done += 1
                            except Exception as e:
                                print(f"\n  [ERROR] N={N} J2={J2} seed={seed} lr={lr} ph={patch_size}: {e}")
                                failed += 1

        print(f"\n{'='*70}")
        print(f"ViT done: {done}  Skipped: {skipped}  Failed: {failed}")

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
        print(f"D-Wave RBM J1-J2 sweep")
        print(f"  Sizes   : {dwave_sizes}")
        print(f"  J2      : {dwave_j2s}")
        print(f"  Seeds   : {dwave_seeds}")
        print(f"  LRs     : {dwave_lrs}")
        print(f"  Methods : {dwave_methods}")
        print(f"  RBM     : {dwave_rbm}")
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
                    if rbm_type not in ("full",) and rbm_type != method:
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

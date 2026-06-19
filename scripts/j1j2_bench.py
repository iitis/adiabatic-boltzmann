"""
J1-J2 1D benchmark — RBM, FBM, and ViT ansätze, classical and D-Wave backends.

Combines the former fbm_j1j2_benchmark.py and vit_j1j2_benchmark.py into one
entry point.  Select ansätze with --ansatz and backend with --mode.

Usage (from project root):
    python scripts/j1j2_bench.py                              # RBM + FBM, classical
    python scripts/j1j2_bench.py --ansatz vit                 # ViT, classical
    python scripts/j1j2_bench.py --ansatz rbm fbm vit         # all three
    python scripts/j1j2_bench.py --mode dwave                 # D-Wave RBM/FBM only
    python scripts/j1j2_bench.py --mode both                  # classical + D-Wave
    python scripts/j1j2_bench.py --ansatz rbm --mode dwave --dwave-methods zephyr
    python scripts/j1j2_bench.py --sizes 8 --lrs 0.05 0.1 --j2 0.0 0.5 1.0
    python scripts/j1j2_bench.py --dry-run
"""

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)

from encoder import Trainer
from encoder_generic import TrainerGeneric
from helpers import _ansatz_str, _model_params_str, _model_subdir, read_qpu_time_ms, save_results
from ising import J1J2Ising1D
from model import DWaveTopologyRBM, FullBoltzmannMachine, FullyConnectedRBM
from model_vit import ViTWaveFunction
from sampler import ClassicalSampler, DimodSampler, GenericClassicalSampler

# ── Shared defaults ───────────────────────────────────────────────────────────

J2_VALUES  = [0.0, 0.1, 0.3, 0.45, 0.55, 0.7, 1.0]
SEEDS      = [1, 42]
J1         = 1.0
H          = 0.5

LR_VALUES  = [0.01, 0.05, 0.1]

REGULARIZATION  = 1e-3
N_SAMPLES       = 1000

# ── ViT architecture per chain length ────────────────────────────────────────

_VIT_BASE = {
    8:  dict(d_model=16, n_heads=2, n_layers=2),
    16: dict(d_model=32, n_heads=4, n_layers=2),
    32: dict(d_model=64, n_heads=4, n_layers=3),
}
PATCH_SIZES = [1, 2]
N_WARMUP    = 20

# ── D-Wave defaults ───────────────────────────────────────────────────────────

DWAVE_SIZES            = [8, 16]
DWAVE_LR_VALUES        = [0.1, 0.01]
DWAVE_SAMPLING_METHODS = ["pegasus", "zephyr"]
DWAVE_RBM_TYPES        = ["full", "fullbm"]
DWAVE_REGULARIZATION   = 1e-5
DWAVE_N_ITERATIONS     = 300

DWAVE_BUDGET_MS = 75 * 60 * 1000
DWAVE_TIME_FILE = Path("time.json")


# ── QPU budget ────────────────────────────────────────────────────────────────


def _require_qpu_time_ms() -> float:
    if not DWAVE_TIME_FILE.exists():
        raise FileNotFoundError(
            f"{DWAVE_TIME_FILE} not found — D-Wave budget tracking file missing. "
            "Create it with {\"time_ms\": 0} or run a D-Wave experiment first."
        )
    return read_qpu_time_ms(DWAVE_TIME_FILE)


def _qpu_budget_exceeded() -> bool:
    used = _require_qpu_time_ms()
    if used >= DWAVE_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] {used / 60_000:.2f} min used >= "
            f"{DWAVE_BUDGET_MS / 60_000:.0f} min limit. Aborting D-Wave experiments."
        )
        return True
    return False


# ── Args namespace builders ───────────────────────────────────────────────────


def _rbm_args(N, J2, seed, lr, rbm_type, iterations, sampler="custom",
              sampling_method="metropolis", reg=REGULARIZATION) -> Namespace:
    return Namespace(
        model="j1j2_1d", size=N, J1=J1, J2=J2, h=H, J=J1, delta=1.0,
        alpha=2.0, ansatz="rbm", rbm=rbm_type, n_hidden=N,
        d_model=32, n_layers=2, n_heads=4, patch_size=2,
        sampler=sampler, sampling_method=sampling_method,
        n_samples=N_SAMPLES, iterations=iterations, learning_rate=lr,
        regularization=reg, seed=seed, cem=True, cem_interval=5,
        sigma=1.0, visualize=False, output_dir=str(_REPO / "results"),
    )


def _vit_args(N, J2, seed, lr, patch_size, iterations, sampler="custom",
              sampling_method="metropolis") -> Namespace:
    base = _VIT_BASE[N]
    return Namespace(
        model="j1j2_1d", size=N, J1=J1, J2=J2, h=H, J=J1, delta=1.0,
        alpha=2.0, ansatz="vit", rbm="full", n_hidden=None,
        d_model=base["d_model"], n_layers=base["n_layers"], n_heads=base["n_heads"],
        patch_size=patch_size, sampler=sampler, sampling_method=sampling_method,
        n_samples=N_SAMPLES, iterations=iterations, learning_rate=lr,
        regularization=REGULARIZATION, seed=seed, cem=False, cem_interval=5,
        sigma=1.0, visualize=False, output_dir=str(_REPO / "results"),
    )


def _result_path(args: Namespace) -> Path:
    output_dir = (
        Path(args.output_dir)
        / _model_subdir(args.model)
        / str(args.size)
        / args.sampler
        / args.sampling_method
    )
    fname = (
        f"result_{args.model}"
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


# ── Single-run helpers ────────────────────────────────────────────────────────


def _run_rbm(N, J2, seed, lr, rbm_type, iterations, dry_run=False) -> dict | None:
    args = _rbm_args(N, J2, seed, lr, rbm_type, iterations)
    out  = _result_path(args)
    label = f"N={N:2d}  J2={J2:.2f}  seed={seed}  lr={lr}  rbm={rbm_type}"
    if out.exists():
        print(f"  [skip]  {label}")
        return None
    if dry_run:
        print(f"  [would run]  {label}")
        return None

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

    wave_fn = FullBoltzmannMachine(N, N, model_key) if rbm_type == "fullbm" \
              else FullyConnectedRBM(N, N, model_key)
    ising   = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    sampler = ClassicalSampler(method="metropolis")
    sampler._key = sampler_key

    config = dict(learning_rate=lr, n_iterations=iterations, n_samples=N_SAMPLES,
                  regularization=REGULARIZATION, seed=seed, use_cem=True, cem_interval=5)
    t0 = time.perf_counter()
    history = Trainer(wave_fn, ising, sampler, config, args=args).train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    try:
        exact = ising.exact_ground_energy()
        print(f"  {label}  err={abs(final_E - exact) / abs(exact):.4f}  t={elapsed:.1f}s")
    except NotImplementedError:
        print(f"  {label}  E={final_E:.6f}  t={elapsed:.1f}s")
    save_results(args, history, ising, rbm=wave_fn)
    return {"final_energy": final_E}


def _run_vit(N, J2, seed, lr, patch_size, iterations, dry_run=False) -> dict | None:
    if N not in _VIT_BASE:
        print(f"  [skip] No ViT config for N={N}")
        return None
    args = _vit_args(N, J2, seed, lr, patch_size, iterations)
    out  = _result_path(args)
    label = f"N={N:2d}  J2={J2:.2f}  seed={seed}  lr={lr}  ph={patch_size}"
    if out.exists():
        print(f"  [skip]  {label}")
        return None
    if dry_run:
        print(f"  [would run]  {label}")
        return None

    base = _VIT_BASE[N]
    key  = jax.random.PRNGKey(seed)
    key, vit_key, sampler_key = jax.random.split(key, 3)

    vit = ViTWaveFunction(n_visible=N, n_layers=base["n_layers"],
                          d_model=base["d_model"], n_heads=base["n_heads"],
                          patch_size=patch_size, key=vit_key, geometry="1d")
    ising   = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    sampler = GenericClassicalSampler(n_warmup=N_WARMUP, n_sweeps=1)
    sampler._key = sampler_key

    config = dict(learning_rate=lr, n_iterations=iterations, n_samples=N_SAMPLES,
                  regularization=REGULARIZATION, seed=seed)
    t0 = time.perf_counter()
    history = TrainerGeneric(vit, ising, sampler, config, args=args).train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    try:
        exact = ising.exact_ground_energy()
        print(f"  {label}  err={abs(final_E - exact) / abs(exact):.4f}  t={elapsed:.1f}s")
    except NotImplementedError:
        print(f"  {label}  E={final_E:.6f}  t={elapsed:.1f}s")
    save_results(args, history, ising, rbm=None)
    return {"final_energy": final_E}


def _run_dwave(N, J2, seed, lr, method, rbm_type, iterations, dry_run=False) -> dict | None:
    args = _rbm_args(N, J2, seed, lr, rbm_type, iterations,
                     sampler="dimod", sampling_method=method, reg=DWAVE_REGULARIZATION)
    args.iterations = iterations
    out  = _result_path(args)
    label = f"N={N:2d}  J2={J2:.2f}  seed={seed}  lr={lr}  method={method}  rbm={rbm_type}"
    if out.exists():
        print(f"  [skip]  {label}")
        return None
    if dry_run:
        print(f"  [would run]  {label}")
        return None

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    if rbm_type == "fullbm":
        wave_fn = FullBoltzmannMachine(N, N, model_key)
    elif rbm_type == "full":
        wave_fn = FullyConnectedRBM(N, N, model_key)
    else:
        wave_fn = DWaveTopologyRBM(N, N, model_key, solver=rbm_type)

    ising   = J1J2Ising1D(N, J1=J1, J2=J2, h=H)
    sampler = DimodSampler(method=method)
    config  = dict(learning_rate=lr, n_iterations=iterations, n_samples=N_SAMPLES,
                   regularization=DWAVE_REGULARIZATION, seed=seed, use_cem=True, cem_interval=5)

    t0 = time.perf_counter()
    history = Trainer(wave_fn, ising, sampler, config, args=args).train()
    elapsed = time.perf_counter() - t0

    final_E = history["energy"][-1]
    try:
        exact = ising.exact_ground_energy()
        print(f"  [D-Wave] {label}  err={abs(final_E - exact) / abs(exact):.4f}  t={elapsed:.1f}s")
    except NotImplementedError:
        print(f"  [D-Wave] {label}  E={final_E:.6f}  t={elapsed:.1f}s")
    save_results(args, history, ising, rbm=wave_fn)
    return {"final_energy": final_E}


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="J1-J2 1D benchmark — RBM, FBM, ViT; classical and D-Wave",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ansatz", nargs="+", choices=["rbm", "fbm", "vit"], default=["rbm", "fbm"],
                   help="Ansätze to run in classical mode (default: rbm fbm)")
    p.add_argument("--mode", choices=["classical", "dwave", "both"], default="classical")
    p.add_argument("--sizes",      type=int,   nargs="+", default=[8, 16])
    p.add_argument("--seeds",      type=int,   nargs="+", default=SEEDS)
    p.add_argument("--j2",         type=float, nargs="+", default=J2_VALUES)
    p.add_argument("--lrs",        type=float, nargs="+", default=LR_VALUES)
    p.add_argument("--iterations", type=int,              default=300)
    p.add_argument("--patch-sizes", type=int,  nargs="+", default=PATCH_SIZES,
                   help="Patch sizes for ViT (default: 1 2)")
    p.add_argument("--dwave-sizes",   type=int,   nargs="+", default=DWAVE_SIZES)
    p.add_argument("--dwave-lrs",     type=float, nargs="+", default=DWAVE_LR_VALUES)
    p.add_argument("--dwave-methods", type=str,   nargs="+", default=DWAVE_SAMPLING_METHODS,
                   choices=["pegasus", "zephyr"])
    p.add_argument("--dwave-rbm",     type=str,   nargs="+", default=DWAVE_RBM_TYPES,
                   choices=["full", "fullbm", "pegasus", "zephyr"])
    p.add_argument("--dwave-iterations", type=int, default=DWAVE_N_ITERATIONS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    cli = _parse_args()
    sizes = sorted(cli.sizes)
    seeds = sorted(cli.seeds)
    j2s   = sorted(cli.j2)
    lrs   = sorted(cli.lrs)

    run_classical = cli.mode in ("classical", "both")
    run_dwave     = cli.mode in ("dwave", "both")

    # ── Classical sweep ───────────────────────────────────────────────────────

    if run_classical:
        done = skipped = failed = 0
        for ansatz in cli.ansatz:
            if ansatz == "vit":
                patch_sizes = sorted(cli.patch_sizes)
                total = len(sizes) * len(patch_sizes) * len(lrs) * len(j2s) * len(seeds)
                print(f"\nViT classical sweep")
                print(f"  Sizes={sizes}  patches={patch_sizes}  J2={j2s}  LRs={lrs}  seeds={seeds}")
                print(f"  Total: {total} runs\n")
                for N in sizes:
                    for ps in patch_sizes:
                        for lr in lrs:
                            for J2 in j2s:
                                for seed in seeds:
                                    try:
                                        r = _run_vit(N, J2, seed, lr, ps, cli.iterations, cli.dry_run)
                                        skipped += r is None; done += r is not None
                                    except Exception as e:
                                        print(f"  [ERROR] N={N} J2={J2} seed={seed} lr={lr} ph={ps}: {e}")
                                        failed += 1
            else:
                rbm_type = "fullbm" if ansatz == "fbm" else "full"
                total = len(sizes) * len(lrs) * len(j2s) * len(seeds)
                print(f"\n{ansatz.upper()} classical sweep")
                print(f"  Sizes={sizes}  J2={j2s}  LRs={lrs}  seeds={seeds}")
                print(f"  Total: {total} runs\n")
                for N in sizes:
                    for lr in lrs:
                        for J2 in j2s:
                            for seed in seeds:
                                try:
                                    r = _run_rbm(N, J2, seed, lr, rbm_type, cli.iterations, cli.dry_run)
                                    skipped += r is None; done += r is not None
                                except Exception as e:
                                    print(f"  [ERROR] N={N} J2={J2} seed={seed} lr={lr}: {e}")
                                    failed += 1

        print(f"\nClassical: done={done}  skipped={skipped}  failed={failed}")

    # ── D-Wave sweep ──────────────────────────────────────────────────────────

    if run_dwave:
        try:
            used_ms = _require_qpu_time_ms()
        except Exception as e:
            print(f"[QPU BUDGET ERROR] {e} — aborting D-Wave sweep.")
            return

        dwave_sizes = sorted(cli.dwave_sizes)
        dwave_lrs   = sorted(cli.dwave_lrs)
        dwave_j2s   = j2s
        dwave_seeds = seeds

        total = (len(dwave_sizes) * len(cli.dwave_methods) * len(cli.dwave_rbm)
                 * len(dwave_lrs) * len(dwave_j2s) * len(dwave_seeds))
        print(f"\nD-Wave sweep")
        print(f"  Sizes={dwave_sizes}  methods={cli.dwave_methods}  rbm={cli.dwave_rbm}")
        print(f"  J2={dwave_j2s}  LRs={dwave_lrs}  seeds={dwave_seeds}")
        print(f"  QPU budget: {DWAVE_BUDGET_MS/60_000:.0f} min total  |  "
              f"used: {used_ms/60_000:.2f} min  |  "
              f"remaining: {max(0.0, DWAVE_BUDGET_MS/60_000 - used_ms/60_000):.2f} min")
        print(f"  Total: {total} runs\n")

        done = skipped = failed = 0
        for N in dwave_sizes:
            for method in cli.dwave_methods:
                for rbm_type in cli.dwave_rbm:
                    if rbm_type not in ("full", "fullbm") and rbm_type != method:
                        continue
                    for lr in dwave_lrs:
                        for J2 in dwave_j2s:
                            for seed in dwave_seeds:
                                try:
                                    if _qpu_budget_exceeded():
                                        print(f"\nD-Wave: done={done}  skipped={skipped}  failed={failed}")
                                        return
                                except Exception as e:
                                    print(f"[QPU BUDGET ERROR] {e} — aborting.")
                                    return
                                try:
                                    r = _run_dwave(N, J2, seed, lr, method, rbm_type,
                                                   cli.dwave_iterations, cli.dry_run)
                                    skipped += r is None; done += r is not None
                                except Exception as e:
                                    print(f"  [D-Wave ERROR] N={N} J2={J2} seed={seed} "
                                          f"lr={lr} method={method} rbm={rbm_type}: {e}")
                                    failed += 1

        print(f"\nD-Wave: done={done}  skipped={skipped}  failed={failed}")
        print(f"Total QPU time used: {_require_qpu_time_ms() / 60_000:.2f} min")


if __name__ == "__main__":
    main()

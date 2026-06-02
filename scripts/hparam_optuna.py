"""
hparam_optuna.py — Optuna hyperparameter search for VMC+RBM quantum spin solvers.

Searches all optimization and architecture hyperparameters (learning rate,
regularization, sampler schedule, CG solver, ansatz type, hidden unit count)
against a chosen Hamiltonian. Each trial result is saved to disk in the same
format as the existing benchmark scripts; an append-only index.jsonl is also
written for easy retrieval.

Usage (from project root):
    python scripts/hparam_optuna.py
    python scripts/hparam_optuna.py --hamiltonian heisenberg_j1j2_1d \\
        --N 8 12 16 --J2 0.1 0.3 0.5 0.7 --n-trials 200 --iterations 150
    python scripts/hparam_optuna.py --hamiltonian j1j2_1d --N 8 --n-trials 50
    python scripts/hparam_optuna.py --resume --study-name my_study \\
        --hamiltonian heisenberg_j1j2_1d

    # pegasus_fast: ~0.3 s per SR iteration → budget ≈ n_trials × iterations × 0.3 s
    # 10 trials × 60 iters = 180 s ≈ 3 min per (N, sweep_val) combo
    python scripts/hparam_optuna.py --hamiltonian heisenberg_j1j2_1d \\
        --N 8 12 --sampling-methods pegasus_fast --ansatz-types rbm \\
        --n-trials 10 --iterations 60

Adding a new Hamiltonian:
    Add an entry to HAMILTONIAN_REGISTRY below — no other changes needed.

Result layout:
    results/hparam_search/{hamiltonian}/{study_name}/
        study.db              Optuna SQLite (resumable, full trial metadata)
        index.jsonl           one JSON line per completed trial; load with
                              pd.read_json(..., lines=True)
        {N}/{custom}/{method}/result_...json   full training history per trial
                                               (written by helpers.save_results)
Retrieval example:
    import pandas as pd
    df = pd.read_json("results/hparam_search/heisenberg_j1j2_1d/my_study/index.jsonl",
                      lines=True)
    best = df.nsmallest(10, "objective")  # lowest relative error first
"""

import argparse
import fcntl
import json
import math
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from encoder import Trainer
from helpers import save_results
from ising import (
    HeisenbergXXZ1D,
    HeisenbergXY1D,
    J1J2HeisenbergXXZ1D,
    J1J2Ising1D,
    LongRangeTFIM1D,
    TransverseFieldIsing1D,
)
from model import FullBoltzmannMachine, FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler


# ---------------------------------------------------------------------------
# Hamiltonian registry
# ---------------------------------------------------------------------------
# Each entry defines:
#   build(N, phys)     → Hamiltonian instance from chain length and physical params
#   model_key          → value for args.model (used in result file naming)
#   defaults           → default physical parameters
#   sweep_params       → suggested values for CLI --sweep-param flags
#   exact_max_N        → largest N with tractable exact_ground_energy()
#   args_extra(phys)   → extra fields to inject into the args Namespace
#
# To add a Hamiltonian: add an entry here.  No other code changes are needed.

HAMILTONIAN_REGISTRY: dict[str, dict] = {
    "heisenberg_j1j2_1d": {
        "model_key": "heisenberg_j1j2_1d",
        "build": lambda N, p: J1J2HeisenbergXXZ1D(
            N, J1=p["J1"], J2=p["J2"], delta=p.get("delta", 1.0)
        ),
        "defaults": {"J1": 1.0, "J2": 0.3, "delta": 1.0},
        "sweep_params": {"J2": [0.0, 0.1, 0.3, 0.45, 0.55, 0.7, 1.0]},
        "exact_max_N": 20,
        "args_extra": lambda p: {
            "J1": p["J1"],
            "J2": p["J2"],
            "delta": p.get("delta", 1.0),
            "h": 0.0,
            "J": p["J1"],
        },
    },
    "j1j2_1d": {
        "model_key": "j1j2_1d",
        "build": lambda N, p: J1J2Ising1D(N, J1=p["J1"], J2=p["J2"], h=p["h"]),
        "defaults": {"J1": 1.0, "J2": 0.5, "h": 0.5},
        "sweep_params": {"J2": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "exact_max_N": 16,
        "args_extra": lambda p: {
            "J1": p["J1"],
            "J2": p["J2"],
            "h": p["h"],
            "J": p["J1"],
            "delta": 1.0,
        },
    },
    "heisenberg_xy_1d": {
        "model_key": "heisenberg_xy_1d",
        "build": lambda N, p: HeisenbergXY1D(N, J=p["J"]),
        "defaults": {"J": 1.0},
        "sweep_params": {"J": [1.0]},  # J just rescales; one combo per N value
        "exact_max_N": 10000,  # JW formula, no practical limit
        "args_extra": lambda p: {
            "J": p["J"],
            "delta": 0.0,
            "h": 0.0,
            "J1": p["J"],
            "J2": 0.0,
        },
    },
    "heisenberg_xxz_1d": {
        "model_key": "heisenberg_xxz_1d",
        "build": lambda N, p: HeisenbergXXZ1D(N, J=p["J"], delta=p["delta"]),
        "defaults": {"J": 1.0, "delta": 1.0},
        "sweep_params": {"delta": [0.5, 1.0, 1.5]},
        "exact_max_N": 20,
        "args_extra": lambda p: {
            "J": p["J"],
            "delta": p["delta"],
            "h": 0.0,
            "J1": p["J"],
            "J2": 0.0,
        },
    },
    "tfim_1d": {
        "model_key": "1d",
        "build": lambda N, p: TransverseFieldIsing1D(N, h=p["h"]),
        "defaults": {"h": 0.5},
        "sweep_params": {"h": [0.2, 0.5, 1.0, 2.0]},
        "exact_max_N": 512,  # closed-form integral, no practical limit
        "args_extra": lambda p: {
            "h": p["h"],
            "J1": 1.0,
            "J2": 0.0,
            "J": 1.0,
            "delta": 1.0,
        },
    },
    "lr_tfim_1d": {
        "model_key": "lr1d",
        "build": lambda N, p: LongRangeTFIM1D(N, h=p["h"], alpha=p["alpha"], J=p["J"]),
        "defaults": {"h": 0.5, "alpha": 2.0, "J": 1.0},
        "sweep_params": {"alpha": [1.0, 1.5, 2.0, 3.0]},
        "exact_max_N": 16,
        "args_extra": lambda p: {
            "h": p["h"],
            "alpha": p["alpha"],
            "J": p["J"],
            "J1": p["J"],
            "J2": 0.0,
            "delta": 1.0,
        },
    },
}

# Sampling methods that conserve Sz — physically correct for XXZ Heisenberg
_SZ_CONSERVING = {"exchange"}
# Methods for which beta is treated as fixed (no CEM beta adaptation)
_BETA_FIXED_METHODS = {"metropolis", "gibbs"}
# Methods that require DimodSampler (D-Wave QPU backend)
_QPU_METHODS = {"pegasus", "zephyr", "pegasus_fast", "zephyr_fast", "pegasus_ra", "zephyr_ra"}


# ---------------------------------------------------------------------------
# args Namespace builder
# ---------------------------------------------------------------------------


def _build_args(
    *,
    N: int,
    hamiltonian: str,
    phys_params: dict,
    n_hidden: int,
    ansatz_type: str,
    lr: float,
    reg: float,
    n_samples: int,
    sampling_method: str,
    n_iterations: int,
    cg_tol: float,
    cg_maxiter: int,
    use_cem: bool,
    cem_interval: int,
    cem_ema_alpha: float,
    seed: int,
    output_dir: Path,
    fast_anneal_time_ns: float = 7.0,
) -> Namespace:
    """Build the args Namespace consumed by save_results, Trainer, and helpers."""
    entry = HAMILTONIAN_REGISTRY[hamiltonian]
    rbm_key = "fullbm" if ansatz_type == "fbm" else "full"

    return Namespace(
        model=entry["model_key"],
        size=N,
        **entry["args_extra"](phys_params),
        # Architecture
        rbm=rbm_key,
        ansatz=ansatz_type,
        n_hidden=n_hidden,
        alpha=n_hidden / N,
        # ViT params — not used for RBM/FBM but expected by _ansatz_str fallback
        d_model=32,
        n_layers=2,
        n_heads=4,
        patch_size=2,
        # Sampler
        sampler="dimod" if sampling_method in _QPU_METHODS else "custom",
        sampling_method=sampling_method,
        n_samples=n_samples,
        # Training
        iterations=n_iterations,
        learning_rate=lr,
        regularization=reg,
        seed=seed,
        # CEM flags — save_results reads args.cem
        cem=use_cem,
        cem_interval=cem_interval,
        # sigma encodes lsb_sigma for LSB runs (used in result filename)
        # for pegasus_fast it encodes fast_anneal_time_ns
        sigma=fast_anneal_time_ns if sampling_method == "pegasus_fast" else 1.0,
        visualize=False,
        output_dir=str(output_dir),
    )


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------


def run_trial(
    *,
    N: int,
    hamiltonian: str,
    phys_params: dict,
    n_hidden: int,
    ansatz_type: str,
    lr: float,
    reg: float,
    n_samples: int,
    sampling_method: str,
    n_iterations: int,
    cg_tol: float,
    cg_maxiter: int,
    use_cem: bool,
    cem_interval: int,
    cem_ema_alpha: float,
    T_initial: float,
    T_final: float,
    n_warmup: int,
    lsb_steps: int,
    lsb_sigma: float,
    lsb_delta: float,
    seed: int,
    output_dir: Path,
    fast_anneal_time_ns: float = 7.0,
) -> dict:
    """
    Build the model, run training, save the full result JSON, return metrics.

    Returns
    -------
    dict with keys: final_energy, exact_energy, rel_error, abs_error, wall_time_s
    """
    entry = HAMILTONIAN_REGISTRY[hamiltonian]
    ising = entry["build"](N, phys_params)

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

    if ansatz_type == "fbm":
        wave_fn = FullBoltzmannMachine(N, n_hidden, model_key)
    else:
        wave_fn = FullyConnectedRBM(N, n_hidden, model_key)

    if sampling_method in _QPU_METHODS:
        sampler = DimodSampler(method=sampling_method)
    else:
        sampler = ClassicalSampler(
            method=sampling_method,
            n_warmup=n_warmup,
            n_sweeps=1,
            T_initial=T_initial,
            T_final=T_final,
        )
        sampler._key = sampler_key

    trainer_config = {
        "learning_rate": lr,
        "n_iterations": n_iterations,
        "n_samples": n_samples,
        "regularization": reg,
        "cg_tol": cg_tol,
        "cg_maxiter": cg_maxiter,
        "use_cem": use_cem,
        "cem_interval": cem_interval,
        "cem_ema_alpha": cem_ema_alpha,
        # Pass SA schedule through Trainer → sampler (sampler reads from config)
        "T_initial": T_initial,
        "T_final": T_final,
        # LSB params — ignored by non-LSB samplers
        "lsb_steps": lsb_steps,
        "lsb_sigma": lsb_sigma,
        "lsb_delta": lsb_delta,
        # fast anneal param — ignored by non-fast-anneal samplers
        "fast_anneal_time_ns": fast_anneal_time_ns,
        "seed": seed,
        "stop_at_convergence": False,
        "save_checkpoints": False,
    }

    args = _build_args(
        N=N,
        hamiltonian=hamiltonian,
        phys_params=phys_params,
        n_hidden=n_hidden,
        ansatz_type=ansatz_type,
        lr=lr,
        reg=reg,
        n_samples=n_samples,
        sampling_method=sampling_method,
        n_iterations=n_iterations,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        use_cem=use_cem,
        cem_interval=cem_interval,
        cem_ema_alpha=cem_ema_alpha,
        seed=seed,
        output_dir=output_dir,
        fast_anneal_time_ns=fast_anneal_time_ns,
    )

    t0 = time.perf_counter()
    trainer = Trainer(wave_fn, ising, sampler, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    # Short-circuit if training diverged (NaN/inf energy at any point)
    if any(not math.isfinite(e) for e in history["energy"]):
        return {"final_energy": float("nan"), "exact_energy": None,
                "rel_error": float("nan"), "abs_error": float("nan"),
                "wall_time_s": time.perf_counter() - t0}

    # Objective: mean energy over the last 20 % of iterations (reduces noise)
    energies = history["energy"]
    tail_start = max(0, int(0.8 * len(energies)))
    tail_mean = float(jnp.mean(jnp.array(energies[tail_start:])))

    try:
        exact = ising.exact_ground_energy()
        rel_error = abs(tail_mean - exact) / abs(exact)
        abs_error = abs(tail_mean - exact)
    except NotImplementedError:
        exact = None
        rel_error = float("inf")
        abs_error = float("inf")

    save_results(args, history, ising, rbm=wave_fn)

    return {
        "final_energy": tail_mean,
        "exact_energy": exact,
        "rel_error": rel_error,
        "abs_error": abs_error,
        "wall_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Optuna objective factory
# ---------------------------------------------------------------------------


def make_objective(cli, study_dir: Path, n_iterations: int, fixed_N: int, fixed_sweep_val: float):
    """
    Return an Optuna objective function for a single (N, sweep_val) combination.

    N and the sweep parameter are fixed — each combo gets its own study so the
    best hyperparameters are optimised independently per physical regime.
    The sweep parameter is determined by the registry (J2 for J1J2 models,
    delta for XXZ, etc.).
    """
    entry = HAMILTONIAN_REGISTRY[cli.hamiltonian]
    phys_defaults = entry["defaults"].copy()
    sweep_keys = list(entry.get("sweep_params", {}).keys())
    _sweep_key = sweep_keys[0] if sweep_keys else "J2"

    def objective(trial: optuna.Trial) -> float:
        # ── Physical config is fixed for this study ───────────────────────
        N = fixed_N
        phys = phys_defaults.copy()
        phys[_sweep_key] = fixed_sweep_val

        # ── Architecture ──────────────────────────────────────────────────
        # n_hidden_alpha = M/N; search in [1, 4]; always at least N hidden units
        alpha_ratio = trial.suggest_float("n_hidden_alpha", 1.0, 4.0)
        n_hidden = max(N, int(math.ceil(alpha_ratio * N)))

        ansatz_type = trial.suggest_categorical("ansatz_type", cli.ansatz_types)

        # FBM has O(N²) extra parameters — prune very large FBM trials early
        if ansatz_type == "fbm" and N > 16:
            raise optuna.exceptions.TrialPruned()

        # ── Sampler (must come before n_samples — range depends on method) ──
        sampling_method = trial.suggest_categorical(
            "sampling_method", cli.sampling_methods
        )

        # ── Optimizer ─────────────────────────────────────────────────────
        lr = trial.suggest_float("learning_rate", 5e-4, 5e-1, log=True)
        reg = trial.suggest_float("regularization", 1e-7, 1e-1, log=True)
        if sampling_method == "pegasus_fast":
            n_samples = 1000
        else:
            n_samples = trial.suggest_int("n_samples", 200, 2000, step=200)

        # ── CG solver ─────────────────────────────────────────────────────
        cg_tol = trial.suggest_float("cg_tol", 1e-10, 1e-5, log=True)
        cg_maxiter = trial.suggest_int("cg_maxiter", 50, 300)

        # Warmup sweeps (classical only — QPU methods have no warmup)
        n_warmup = 0 if sampling_method in _QPU_METHODS else trial.suggest_int("n_warmup", 50, 500, step=50)

        # Sampler-specific hyperparameters
        if sampling_method == "simulated_annealing":
            T_initial = trial.suggest_float("T_initial", 1.0, 20.0)
            T_final = trial.suggest_float("T_final", 0.1, 2.0)
            use_cem = trial.suggest_categorical("use_cem", [True, False])
            if use_cem:
                cem_ema_alpha = trial.suggest_float("cem_ema_alpha", 0.05, 0.5)
                cem_interval = trial.suggest_int("cem_interval", 1, 10)
            else:
                cem_ema_alpha = 0.3
                cem_interval = 5
            fast_anneal_time_ns = 7.0
            lsb_steps, lsb_sigma, lsb_delta = 1000, 1.0, 1.0

        elif sampling_method == "lsb":
            # LSB integration params
            lsb_steps = trial.suggest_int("lsb_steps", 500, 3000, step=500)
            # lsb_sigma is 1/σ² — higher = less noise; log scale covers [0.25, 4]
            lsb_sigma = trial.suggest_float("lsb_sigma", 0.25, 4.0, log=True)
            lsb_delta = trial.suggest_float("lsb_delta", 0.5, 2.0)
            # CEM always active for LSB (beta_fixed=False in Trainer)
            use_cem = True
            cem_ema_alpha = trial.suggest_float("cem_ema_alpha", 0.05, 0.5)
            cem_interval = trial.suggest_int("cem_interval", 1, 10)
            fast_anneal_time_ns = 7.0
            T_initial, T_final = 5.0, 1.0

        elif sampling_method == "pegasus_fast":
            # Fast anneal in the coherent regime — only anneal time is QPU-specific.
            # n_samples upper bound is kept low: QPU latency is dominated by cloud
            # overhead, not anneal time, so diminishing returns beyond ~400 reads.
            fast_anneal_time_ns = trial.suggest_float("fast_anneal_time_ns", 0.5, 20.0, log=True)
            T_initial, T_final = 5.0, 1.0
            use_cem = False
            cem_ema_alpha, cem_interval = 0.3, 5
            lsb_steps, lsb_sigma, lsb_delta = 1000, 1.0, 1.0

        else:
            # metropolis, exchange, gibbs, remaining QPU methods — no schedule, no CEM
            # (gibbs: beta_fixed=True in Trainer so CEM is silently skipped anyway)
            # (QPU: DimodSampler ignores T_initial/T_final entirely)
            fast_anneal_time_ns = 7.0
            T_initial, T_final = 5.0, 1.0
            use_cem = False
            cem_ema_alpha, cem_interval = 0.3, 5
            lsb_steps, lsb_sigma, lsb_delta = 1000, 1.0, 1.0

        # ── Seed (multiple seeds → robust estimate of each config) ────────
        seed = trial.suggest_categorical("seed", cli.seeds)

        # ── Run ───────────────────────────────────────────────────────────
        result = run_trial(
            N=N,
            hamiltonian=cli.hamiltonian,
            phys_params=phys,
            n_hidden=n_hidden,
            ansatz_type=ansatz_type,
            lr=lr,
            reg=reg,
            n_samples=n_samples,
            sampling_method=sampling_method,
            n_iterations=n_iterations,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            use_cem=use_cem,
            cem_interval=cem_interval,
            cem_ema_alpha=cem_ema_alpha,
            T_initial=T_initial,
            T_final=T_final,
            n_warmup=n_warmup,
            lsb_steps=lsb_steps,
            lsb_sigma=lsb_sigma,
            lsb_delta=lsb_delta,
            fast_anneal_time_ns=fast_anneal_time_ns,
            seed=seed,
            output_dir=study_dir,
        )

        # Diverged trials (NaN/inf energy) are pruned so TPE still learns from them
        if not math.isfinite(result["rel_error"]):
            raise optuna.exceptions.TrialPruned()

        # ── Persist trial summary to index ────────────────────────────────
        summary = {
            "trial": trial.number,
            "datetime": datetime.now().isoformat(),
            "hamiltonian": cli.hamiltonian,
            "N": N,
            "phys_params": phys,
            "params": dict(trial.params),
            "n_hidden": n_hidden,
            "objective": result["rel_error"],
            "rel_error": result["rel_error"],
            "abs_error": result["abs_error"],
            "final_energy": result["final_energy"],
            "exact_energy": result["exact_energy"],
            "wall_time_s": result["wall_time_s"],
        }
        _append_index(study_dir / "index.jsonl", summary)

        return result["rel_error"]

    return objective


# ---------------------------------------------------------------------------
# Atomic index append
# ---------------------------------------------------------------------------


def _append_index(path: Path, record: dict):
    """Append one JSON line to the index file, safe for concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for VMC+RBM on quantum spin models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hamiltonian",
        choices=list(HAMILTONIAN_REGISTRY.keys()),
        default="heisenberg_j1j2_1d",
        help="Which Hamiltonian to optimise against (default: heisenberg_j1j2_1d)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Study name; auto-generated from datetime if omitted",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing study by name instead of creating a new one",
    )
    parser.add_argument(
        "--n-trials", type=int, default=200, help="Number of Optuna trials (default: 200)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="SR training iterations per trial (default: 100)",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[8, 12],
        help="Chain lengths to include in the search (default: 8 12)",
    )
    parser.add_argument(
        "--J2",
        type=float,
        nargs="*",
        default=None,
        help="J2 values to sweep (frustrated models); defaults to registry sweep_params",
    )
    parser.add_argument(
        "--ansatz-types",
        nargs="+",
        default=["rbm", "fbm"],
        choices=["rbm", "fbm"],
        help="Ansatz types to include (default: rbm fbm)",
    )
    parser.add_argument(
        "--sampling-methods",
        nargs="+",
        default=["metropolis", "simulated_annealing", "exchange"],
        help="Sampler methods to include (default: metropolis simulated_annealing exchange)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 42],
        help="Random seeds; multiple seeds → objective is noisy but more robust (default: 1 42)",
    )
    parser.add_argument(
        "--optuna-sampler",
        choices=["tpe", "cmaes", "random"],
        default="tpe",
        help="Optuna search algorithm (default: tpe)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "results"),
        help="Base results directory (default: results/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return parser


def main():
    cli = _build_cli().parse_args()

    # Default sweep values from registry when not overridden on CLI.
    # The sweep parameter differs per hamiltonian (J2 for J1J2 models, delta for XXZ, etc.).
    if cli.J2 is None:
        entry = HAMILTONIAN_REGISTRY[cli.hamiltonian]
        sweep_params = entry.get("sweep_params", {})
        cli.J2 = next(iter(sweep_params.values()), [entry["defaults"].get("J2", 0.0)])
    entry = HAMILTONIAN_REGISTRY[cli.hamiltonian]
    sweep_keys = list(entry.get("sweep_params", {}).keys())
    _sweep_key = sweep_keys[0] if sweep_keys else "J2"

    # Warn when N exceeds exact ED limit
    max_N = entry["exact_max_N"]
    over = [n for n in cli.N if n > max_N]
    if over:
        print(
            f"Warning: N={over} exceed exact_max_N={max_N} — "
            "relative error unavailable, abs energy minimised instead."
        )

    # One session name / base directory; sub-studies live under it per combo
    study_name = cli.study_name or (
        f"{cli.hamiltonian}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    base_dir = Path(cli.output_dir) / "hparam_search" / cli.hamiltonian / study_name
    base_dir.mkdir(parents=True, exist_ok=True)

    n_combos = len(cli.N) * len(cli.J2)

    if cli.dry_run:
        print(f"[dry-run] session    : {study_name}")
        print(f"[dry-run] directory  : {base_dir}")
        print(f"[dry-run] hamiltonian: {cli.hamiltonian}")
        print(f"[dry-run] N          : {cli.N}")
        print(f"[dry-run] {_sweep_key:12s}: {cli.J2}")
        print(f"[dry-run] combos     : {n_combos}  ({len(cli.N)} N  ×  {len(cli.J2)} {_sweep_key}) — 1 study each")
        print(f"[dry-run] ansätze    : {cli.ansatz_types}")
        print(f"[dry-run] samplers   : {cli.sampling_methods}")
        print(f"[dry-run] seeds      : {cli.seeds}")
        print(f"[dry-run] trials     : {cli.n_trials}  ×  {cli.iterations} iters each (per combo)")
        return

    print(f"\nOptuna session    : {study_name}")
    print(f"  Hamiltonian     : {cli.hamiltonian}")
    print(f"  N values        : {cli.N}")
    print(f"  {_sweep_key} values    : {cli.J2}")
    print(f"  Combos          : {n_combos}  (separate study per combo)")
    print(f"  Ansätze         : {cli.ansatz_types}")
    print(f"  Samplers        : {cli.sampling_methods}")
    print(f"  Seeds           : {cli.seeds}")
    print(f"  Trials per combo: {cli.n_trials}  ×  {cli.iterations} iters each")
    print(f"  Optuna sampler  : {cli.optuna_sampler}")
    print(f"  Base dir        : {base_dir}\n")

    all_summaries = []

    for N in cli.N:
        for sweep_val in cli.J2:
            combo_key = f"N{N}_{_sweep_key}{sweep_val}"
            combo_dir = base_dir / combo_key
            combo_dir.mkdir(parents=True, exist_ok=True)

            combo_study_name = f"{study_name}_{combo_key}"
            J2 = sweep_val  # keep local alias so combo_summary["J2"] stays meaningful
            db_path = combo_dir / "study.db"
            storage = f"sqlite:///{db_path}"

            if cli.optuna_sampler == "tpe":
                opt_sampler = optuna.samplers.TPESampler(seed=0)
            elif cli.optuna_sampler == "cmaes":
                opt_sampler = optuna.samplers.CmaEsSampler(seed=0)
            else:
                opt_sampler = optuna.samplers.RandomSampler(seed=0)

            study = optuna.create_study(
                study_name=combo_study_name,
                storage=storage,
                direction="minimize",
                sampler=opt_sampler,
                load_if_exists=cli.resume,
            )

            print(f"[{combo_key}]  study : {combo_study_name}")
            print(f"{'':>{len(combo_key)+2}}  db    : {db_path}")
            print(f"{'':>{len(combo_key)+2}}  index : {combo_dir / 'index.jsonl'}")

            objective_fn = make_objective(
                cli, combo_dir, cli.iterations, fixed_N=N, fixed_sweep_val=sweep_val
            )

            study.optimize(
                objective_fn,
                n_trials=cli.n_trials,
                n_jobs=1,           # sequential — single GPU
                catch=(Exception,),
                show_progress_bar=True,
            )

            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

            combo_summary: dict = {
                "combo": combo_key, "N": N, "J2": J2,
                "n_completed": len(completed), "n_failed": len(failed), "n_pruned": len(pruned),
                "best_objective": None, "best_params": None,
            }

            if completed:
                best = study.best_trial
                combo_summary["best_objective"] = best.value
                combo_summary["best_params"] = dict(best.params)
                print(f"{'':>{len(combo_key)+2}}  best  : trial #{best.number}  obj={best.value:.6f}")
            else:
                print(f"{'':>{len(combo_key)+2}}  best  : no completed trials")

            all_summaries.append(combo_summary)
            print()

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"Session complete — {len(all_summaries)} combo(s)")
    print(f"{'Combo':22s}  {'Best obj':>10s}  {'Done':>6s}  {'Fail':>5s}  {'Pruned':>6s}")
    print(f"{'-'*57}")
    for s in all_summaries:
        obj_str = f"{s['best_objective']:.6f}" if s["best_objective"] is not None else "    N/A"
        print(f"  {s['combo']:20s}  {obj_str:>10s}  {s['n_completed']:>6d}  {s['n_failed']:>5d}  {s['n_pruned']:>6d}")

    print(f"\nTo load results for a specific combo (example):")
    print(f"  import pandas as pd")
    if all_summaries:
        ex = all_summaries[0]["combo"]
        print(f"  df = pd.read_json('{base_dir / ex / 'index.jsonl'}', lines=True)")
    print(f"  best = df.nsmallest(10, 'objective')")


if __name__ == "__main__":
    main()

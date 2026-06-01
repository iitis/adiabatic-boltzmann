"""
tts_run.py — Time-To-Solution benchmark using best hyperparameters from hparam search.

For each (hamiltonian, study, N, sweep_val, sampling_method):
  1. Reads index.jsonl → selects best trial per sampling_method (lowest rel_error)
  2. Re-runs that exact config across --n-seeds random seeds (seed is the only variation)
  3. Saves full result JSON per seed via save_results() (same format as all other runs)
  4. Computes TTS(ε): first iteration where rolling-mean rel_error < ε
  5. Writes tts_summary.json with per-method TTS statistics

Output layout:
    results/tts/{hamiltonian}/{study_name}/{combo_key}/
        tts_summary.json                           TTS stats per sampling_method
        {model}/{N}/{sampler}/{method}/            per-seed result JSONs (via save_results)

Usage:
    # specific study:
    python scripts/tts_run.py --hamiltonian j1j2_1d \\
        --study-name j1j2_1d_20260527_170753

    # filter to specific N values and/or samplers:
    python scripts/tts_run.py --hamiltonian j1j2_1d \\
        --study-name j1j2_1d_20260527_170753 \\
        --N 8 12 --sampling-methods exchange metropolis

    # all studies for a hamiltonian:
    python scripts/tts_run.py --hamiltonian heisenberg_j1j2_1d

    # all hamiltonians and studies:
    python scripts/tts_run.py --all

    # dry run to see what would be processed:
    python scripts/tts_run.py --hamiltonian j1j2_1d --dry-run
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from encoder import Trainer
from helpers import save_results
from hparam_optuna import (
    HAMILTONIAN_REGISTRY,
    _QPU_METHODS,
    _build_args,
)
from model import FullBoltzmannMachine, FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler


# ---------------------------------------------------------------------------
# Best-config extraction
# ---------------------------------------------------------------------------


def load_best_per_sampler(
    index_path: Path,
    methods_filter: list[str] | None = None,
) -> dict[str, dict]:
    """
    Read index.jsonl, return {sampling_method: best_row_dict} per method.
    Only considers trials with finite objectives. Optionally restrict to methods_filter.
    """
    df = pd.read_json(index_path, lines=True)
    df = df[df["objective"].apply(math.isfinite)].copy()
    if df.empty:
        return {}
    df["_method"] = df["params"].apply(lambda p: p["sampling_method"])
    if methods_filter:
        df = df[df["_method"].isin(methods_filter)]
    best: dict[str, dict] = {}
    for method, group in df.groupby("_method"):
        best[method] = group.nsmallest(1, "objective").iloc[0].to_dict()
    return best


# ---------------------------------------------------------------------------
# TTS computation
# ---------------------------------------------------------------------------


def compute_tts(
    energies: list[float],
    exact_energy: float,
    epsilon: float,
    window: int,
) -> int | None:
    """
    Return first iteration t (1-indexed) where the causal rolling-mean
    relative error drops below epsilon. Returns None if never reached.
    The window grows naturally from the start (no padding).
    """
    for t in range(len(energies)):
        w_start = max(0, t - window + 1)
        mean_e = sum(energies[w_start : t + 1]) / (t - w_start + 1)
        if abs(mean_e - exact_energy) / abs(exact_energy) < epsilon:
            return t + 1  # 1-indexed
    return None


def _tts_stats(tts_vals: list[int | None], n_iterations: int) -> dict:
    """Compute TTS summary statistics from a list of per-seed TTS values."""
    n_total = len(tts_vals)
    reached = [v for v in tts_vals if v is not None]
    # Treat "never reached" as n_iterations + 1 for percentile computation
    all_tts = sorted(v if v is not None else n_iterations + 1 for v in tts_vals)

    def _pct(p: float) -> int | None:
        if not all_tts:
            return None
        idx = min(int(math.ceil(p * len(all_tts))) - 1, len(all_tts) - 1)
        return all_tts[max(0, idx)]

    return {
        "n_seeds_reached": len(reached),
        "n_seeds_total": n_total,
        "success_fraction": len(reached) / n_total if n_total else 0.0,
        "tts_mean": sum(all_tts) / len(all_tts) if all_tts else None,
        "tts_p50": _pct(0.50),
        "tts_p90": _pct(0.90),
        "tts_p99": _pct(0.99),
        "tts_per_seed": tts_vals,
    }


# ---------------------------------------------------------------------------
# Single TTS trial
# ---------------------------------------------------------------------------


def run_tts_trial(
    *,
    N: int,
    hamiltonian: str,
    phys_params: dict,
    best_row: dict,
    seed: int,
    n_iterations: int,
    output_dir: Path,
) -> tuple[dict, list[float]]:
    """
    Run one TTS trial with fixed hyperparameters and a given seed.
    Saves the full result JSON via save_results(). Returns (metrics, energies).
    """
    entry = HAMILTONIAN_REGISTRY[hamiltonian]
    ising = entry["build"](N, phys_params)

    p = best_row["params"]
    n_hidden = int(best_row["n_hidden"])
    ansatz_type = p["ansatz_type"]
    lr = p["learning_rate"]
    reg = p["regularization"]
    n_samples = p["n_samples"]
    cg_tol = p["cg_tol"]
    cg_maxiter = p["cg_maxiter"]
    sampling_method = p["sampling_method"]

    # Method-specific params — with safe defaults for methods that don't use them
    T_initial = p.get("T_initial", 5.0)
    T_final = p.get("T_final", 1.0)
    n_warmup = int(p.get("n_warmup", 0))
    use_cem = bool(p.get("use_cem", False))
    cem_interval = int(p.get("cem_interval", 5))
    cem_ema_alpha = float(p.get("cem_ema_alpha", 0.3))
    lsb_steps = int(p.get("lsb_steps", 1000))
    lsb_sigma = float(p.get("lsb_sigma", 1.0))
    lsb_delta = float(p.get("lsb_delta", 1.0))

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

    if ansatz_type == "fbm":
        wave_fn = FullBoltzmannMachine(N, n_hidden, model_key)
    else:
        wave_fn = FullyConnectedRBM(N, n_hidden, model_key)

    if sampling_method in _QPU_METHODS:
        sampler_obj = DimodSampler(method=sampling_method)
    else:
        sampler_obj = ClassicalSampler(
            method=sampling_method,
            n_warmup=n_warmup,
            n_sweeps=1,
            T_initial=T_initial,
            T_final=T_final,
        )
        sampler_obj._key = sampler_key

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
        "T_initial": T_initial,
        "T_final": T_final,
        "lsb_steps": lsb_steps,
        "lsb_sigma": lsb_sigma,
        "lsb_delta": lsb_delta,
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
    )

    t0 = time.perf_counter()
    trainer = Trainer(wave_fn, ising, sampler_obj, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    energies = [float(e) for e in history["energy"]]
    diverged = any(not math.isfinite(e) for e in energies)

    try:
        exact = float(ising.exact_ground_energy())
        tail_start = max(0, int(0.8 * len(energies)))
        tail_mean = float(jnp.mean(jnp.array(energies[tail_start:])))
        rel_error = abs(tail_mean - exact) / abs(exact)
        abs_error = abs(tail_mean - exact)
    except NotImplementedError:
        exact = None
        tail_mean = None
        rel_error = float("inf")
        abs_error = float("inf")

    if not diverged:
        save_results(args, history, ising, rbm=wave_fn)

    metrics = {
        "seed": seed,
        "exact_energy": exact,
        "final_energy": energies[-1] if energies else None,
        "tail_mean_energy": tail_mean,
        "rel_error": rel_error,
        "abs_error": abs_error,
        "wall_time_s": elapsed,
        "diverged": diverged,
    }
    return metrics, energies


# ---------------------------------------------------------------------------
# One combo
# ---------------------------------------------------------------------------


def run_combo(
    *,
    index_path: Path,
    hamiltonian: str,
    N: int,
    phys_params: dict,
    combo_key: str,
    study_name: str,
    n_seeds: int,
    n_iterations: int,
    epsilon: list[float],
    window: int,
    methods_filter: list[str] | None,
    output_base: Path,
) -> None:
    best_per_sampler = load_best_per_sampler(index_path, methods_filter)
    if not best_per_sampler:
        print(f"  [{combo_key}] no completed trials — skipping")
        return

    tts_out = output_base / "tts" / hamiltonian / study_name / combo_key
    tts_out.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "hamiltonian": hamiltonian,
        "N": N,
        "phys_params": phys_params,
        "combo_key": combo_key,
        "study_name": study_name,
        "n_seeds": n_seeds,
        "n_iterations": n_iterations,
        "epsilon": epsilon,
        "window": window,
        "methods": {},
    }

    for method, best_row in sorted(best_per_sampler.items()):
        print(
            f"  [{combo_key}] {method:25s}  "
            f"best_obj={best_row['objective']:.6f}  "
            f"n_hidden={int(best_row['n_hidden'])}"
        )

        seed_metrics: list[dict] = []
        seed_energies: list[list[float]] = []

        for seed in range(n_seeds):
            print(f"    seed {seed + 1}/{n_seeds} ...", end="\r", flush=True)
            try:
                metrics, energies = run_tts_trial(
                    N=N,
                    hamiltonian=hamiltonian,
                    phys_params=phys_params,
                    best_row=best_row,
                    seed=seed,
                    n_iterations=n_iterations,
                    output_dir=tts_out,
                )
                seed_metrics.append(metrics)
                seed_energies.append(energies)
            except Exception as exc:
                print(f"\n    seed {seed} FAILED: {exc}")

        n_ok = sum(1 for m in seed_metrics if not m["diverged"])
        print(f"    {n_ok}/{len(seed_metrics)} seeds OK                    ")

        exact = next(
            (m["exact_energy"] for m in seed_metrics if m["exact_energy"] is not None),
            None,
        )

        tts_by_eps: dict = {}
        if exact is not None:
            for eps in epsilon:
                tts_vals: list[int | None] = []
                for m, energies in zip(seed_metrics, seed_energies):
                    if m["diverged"]:
                        tts_vals.append(None)
                        continue
                    tts_vals.append(compute_tts(energies, exact, eps, window))
                tts_by_eps[str(eps)] = {
                    "epsilon": eps,
                    "window": window,
                    **_tts_stats(tts_vals, n_iterations),
                }

        rel_errors = [
            m["rel_error"]
            for m in seed_metrics
            if not m["diverged"] and math.isfinite(m["rel_error"])
        ]
        wall_times = [m["wall_time_s"] for m in seed_metrics]

        summary["methods"][method] = {
            "best_obj_from_hparam": best_row["objective"],
            "best_params": best_row["params"],
            "n_hidden": int(best_row["n_hidden"]),
            "n_seeds_run": len(seed_metrics),
            "n_seeds_diverged": sum(1 for m in seed_metrics if m["diverged"]),
            "exact_energy": exact,
            "mean_rel_error": sum(rel_errors) / len(rel_errors) if rel_errors else None,
            "mean_wall_time_s": sum(wall_times) / len(wall_times) if wall_times else None,
            "tts": tts_by_eps,
        }

    summary_path = tts_out / "tts_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [{combo_key}] summary → {summary_path}\n")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_index_files(
    hamiltonian: str | None,
    study_name: str | None,
    hparam_base: Path,
) -> list[tuple[str, str, Path]]:
    """Return (hamiltonian, study_name, index_path) tuples to process."""
    found: list[tuple[str, str, Path]] = []

    if hamiltonian:
        ham_dirs = [hparam_base / hamiltonian]
    else:
        ham_dirs = sorted(d for d in hparam_base.iterdir() if d.is_dir())

    for ham_dir in ham_dirs:
        if not ham_dir.exists():
            continue
        ham = ham_dir.name

        if study_name:
            study_dirs = [ham_dir / study_name]
        else:
            study_dirs = sorted(d for d in ham_dir.iterdir() if d.is_dir())

        for study_dir in study_dirs:
            if not study_dir.exists():
                continue
            for index_path in sorted(study_dir.rglob("index.jsonl")):
                found.append((ham, study_dir.name, index_path))

    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TTS benchmark using best hyperparameters from hparam search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hamiltonian",
        choices=list(HAMILTONIAN_REGISTRY.keys()),
        help="Hamiltonian to process (omit with --all for every hamiltonian)",
    )
    parser.add_argument(
        "--study-name",
        help="Specific study name under the hamiltonian dir (omit to use all studies)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all hamiltonians and all studies (ignores --hamiltonian / --study-name)",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        metavar="N",
        help="Filter to specific chain lengths (e.g. --N 8 12)",
    )
    parser.add_argument(
        "--sampling-methods",
        nargs="+",
        metavar="METHOD",
        help="Filter to specific sampling methods (e.g. --sampling-methods exchange metropolis)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds per config (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="SR training iterations per run (default: 300)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[0.01, 0.001],
        help="Relative-error threshold(s) for TTS computation (default: 0.01 0.001)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling-mean window size for TTS computation (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "results"),
        help="Base results directory (default: results/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the list of combos to process and exit",
    )
    return parser


def main():
    cli = _build_cli().parse_args()
    output_base = Path(cli.output_dir)
    hparam_base = output_base / "hparam_search"

    hamiltonian = None if cli.all else cli.hamiltonian
    study_name = None if cli.all else cli.study_name

    index_files = discover_index_files(hamiltonian, study_name, hparam_base)
    if not index_files:
        print("No index.jsonl files found — check --hamiltonian / --study-name / --all")
        return

    print("TTS benchmark")
    print(f"  Seeds      : {cli.n_seeds}")
    print(f"  Iterations : {cli.iterations}")
    print(f"  Epsilon(s) : {cli.epsilon}")
    print(f"  Window     : {cli.window}")
    print(f"  Combos     : {len(index_files)}")
    if cli.sampling_methods:
        print(f"  Methods    : {cli.sampling_methods}")
    if cli.N:
        print(f"  N filter   : {cli.N}")
    print()

    if cli.dry_run:
        for ham, study, idx in index_files:
            print(f"  {ham}/{study}/{idx.parent.name}")
        return

    for ham, study, index_path in index_files:
        combo_key = index_path.parent.name  # e.g. "N8_J20.5"

        try:
            df = pd.read_json(index_path, lines=True)
            if df.empty:
                print(f"[{ham}/{study}/{combo_key}] empty index — skipping")
                continue
            row0 = df.iloc[0]
            N = int(row0["N"])
            phys_params = dict(row0["phys_params"])
        except Exception as exc:
            print(f"[{ham}/{study}/{combo_key}] failed to parse index: {exc} — skipping")
            continue

        if cli.N and N not in cli.N:
            continue

        print(f"[{ham}/{study}/{combo_key}]")
        run_combo(
            index_path=index_path,
            hamiltonian=ham,
            N=N,
            phys_params=phys_params,
            combo_key=combo_key,
            study_name=study,
            n_seeds=cli.n_seeds,
            n_iterations=cli.iterations,
            epsilon=cli.epsilon,
            window=cli.window,
            methods_filter=cli.sampling_methods,
            output_base=output_base,
        )


if __name__ == "__main__":
    main()

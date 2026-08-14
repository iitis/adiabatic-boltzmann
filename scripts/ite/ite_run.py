"""
ite_run.py — Iterations-To-Epsilon (ITE) benchmark using globally best hyperparameters.

ITE counts SR training iterations, not wall-clock time: for each seed it finds
the first iteration whose causal rolling-mean energy is within epsilon of the
exact ground state. This is distinct from the dashboard's "Time to Epsilon
(TTE)" metric (scripts/viz/dashboard.py), which is a wall-clock estimate
derived from success probability and mean sampling time.

For each (hamiltonian, N, sweep_val, sampling_method), all trials from ALL
hparam studies are pooled and the single best config (lowest rel_error) is
selected — regardless of which study it came from.

That config is then re-run across --n-seeds random seeds.  Full result JSONs
are saved via save_results() (same format and location as all other runs).
An ite_summary.json is written per (N, sweep_val) combo with, per epsilon
threshold: ITE (iteration count) stats, "time_to_ite_s" (cumulative sampling
time to the ITE iteration), and "energy_wh_to_ite" (measured GPU energy —
see src/energy.py — prorated by the time fraction elapsed at that iteration;
None per-seed where GPU energy wasn't measured). Because this script always
re-runs the single best hyperparameter config, these reflect genuine
seed-to-seed variance — unlike scripts/viz/plot_ttc.py's loader, which pools
every historical run regardless of hyperparameters.

Output layout:
    results/{model}/{N}/{sampler}/{method}/result_...json   (normal per-seed results)
    results/ite/{hamiltonian}/{combo_key}/ite_summary.json  (ITE stats)

Usage:
    python scripts/ite/ite_run.py --hamiltonian j1j2_1d
    python scripts/ite/ite_run.py --hamiltonian heisenberg_j1j2_1d --N 8 12
    python scripts/ite/ite_run.py --hamiltonian j1j2_1d \\
        --sampling-methods exchange metropolis --epsilon 0.01 0.001 0.0001
    python scripts/ite/ite_run.py --all
    python scripts/ite/ite_run.py --hamiltonian j1j2_1d --dry-run
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "scripts" / "hparam"))

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
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler


# ---------------------------------------------------------------------------
# Global best extraction
# ---------------------------------------------------------------------------


def load_all_trials(hamiltonian: str, hparam_base: Path) -> pd.DataFrame:
    """Pool every index.jsonl under results/hparam_search/{hamiltonian}/ into one DataFrame."""
    ham_dir = hparam_base / hamiltonian
    if not ham_dir.exists():
        return pd.DataFrame()
    frames = []
    for index_path in sorted(ham_dir.rglob("index.jsonl")):
        try:
            df = pd.read_json(index_path, lines=True)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"  Warning: could not read {index_path}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def global_best_per_combo_sampler(
    all_trials: pd.DataFrame,
    methods_filter: list[str] | None = None,
    N_filter: list[int] | None = None,
) -> dict[tuple, dict]:
    """
    Return {(N, phys_key, sampling_method): best_row} — one entry per
    (N, physical params, sampler) combination, globally optimal across all studies.
    phys_key is a JSON string of the phys_params dict (sorted keys).
    """
    df = all_trials[all_trials["objective"].apply(math.isfinite)].copy()
    if df.empty:
        return {}
    df["_method"] = df["params"].apply(lambda p: p["sampling_method"])
    def _clean_phys_key(p: dict) -> str:
        cleaned = {k: round(v, 10) if isinstance(v, float) else v for k, v in dict(p).items()}
        return json.dumps(cleaned, sort_keys=True)

    df["_phys_key"] = df["phys_params"].apply(_clean_phys_key)
    if methods_filter:
        df = df[df["_method"].isin(methods_filter)]
    if N_filter:
        df = df[df["N"].isin(N_filter)]

    best: dict[tuple, dict] = {}
    for (N, phys_key, method), group in df.groupby(["N", "_phys_key", "_method"]):
        best[(int(N), phys_key, method)] = (
            group.nsmallest(1, "objective").iloc[0].to_dict()
        )
    return best


def _fmt_float(v) -> str:
    """Format a float without floating-point noise (0.3 not 0.30000000000000004)."""
    if isinstance(v, float):
        return f"{round(v, 10):g}"
    return str(v)


def _combo_key(hamiltonian: str, N: int, phys_params: dict) -> str:
    """Build a human-readable combo key, e.g. 'N8_J20.5', matching hparam convention."""
    entry = HAMILTONIAN_REGISTRY.get(hamiltonian, {})
    sweep_keys = list(entry.get("sweep_params", {}).keys())
    if sweep_keys:
        k = sweep_keys[0]
        return f"N{N}_{k}{_fmt_float(phys_params.get(k, '?'))}"
    return f"N{N}"


# ---------------------------------------------------------------------------
# ITE computation
# ---------------------------------------------------------------------------


def compute_ite(
    energies: list[float],
    exact_energy: float,
    epsilon: float,
    window: int,
) -> int | None:
    """
    First iteration t (1-indexed) where the causal rolling-mean relative error
    drops below epsilon. Window grows naturally from iteration 0. Returns None
    if never reached within the recorded curve.
    """
    for t in range(len(energies)):
        w_start = max(0, t - window + 1)
        mean_e = sum(energies[w_start : t + 1]) / (t - w_start + 1)
        if abs(mean_e - exact_energy) / abs(exact_energy) < epsilon:
            return t + 1  # 1-indexed
    return None


def _ite_stats(ite_vals: list[int | None], n_iterations: int) -> dict:
    n_total = len(ite_vals)
    reached = [v for v in ite_vals if v is not None]
    all_ite = sorted(v if v is not None else n_iterations + 1 for v in ite_vals)

    def _pct(p: float) -> int | None:
        if not all_ite:
            return None
        return all_ite[min(int(math.ceil(p * len(all_ite))) - 1, len(all_ite) - 1)]

    return {
        "n_seeds_reached": len(reached),
        "n_seeds_total": n_total,
        "success_fraction": len(reached) / n_total if n_total else 0.0,
        "ite_mean": sum(all_ite) / len(all_ite) if all_ite else None,
        "ite_p50": _pct(0.50),
        "ite_p90": _pct(0.90),
        "ite_p99": _pct(0.99),
        "ite_per_seed": ite_vals,
    }


def _continuous_stats(vals: list[float | None]) -> dict:
    """
    Percentile stats over the reached subset only — unlike _ite_stats, there
    is no natural "never reached" sentinel for a continuous quantity (time,
    energy), so unreached seeds are excluded rather than penalized.
    """
    reached = sorted(v for v in vals if v is not None)

    def _pct(p: float) -> float | None:
        if not reached:
            return None
        return reached[min(int(math.ceil(p * len(reached))) - 1, len(reached) - 1)]

    return {
        "mean": sum(reached) / len(reached) if reached else None,
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p99": _pct(0.99),
        "per_seed": vals,
    }


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------


def run_ite_trial(
    *,
    N: int,
    hamiltonian: str,
    phys_params: dict,
    best_row: dict,
    seed: int,
    n_iterations: int,
    output_dir: Path,
) -> tuple[dict, list[float], list[float]]:
    """Run one seed with the fixed best hyperparameters. Returns (metrics, energies, times)."""
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

    T_initial = p.get("T_initial", 5.0)
    T_final = p.get("T_final", 1.0)
    n_warmup = int(p.get("n_warmup", 0))
    use_cem = bool(p.get("use_cem", False))
    cem_interval = int(p.get("cem_interval", 5))
    cem_ema_alpha = float(p.get("cem_ema_alpha", 0.3))
    lsb_steps = int(p.get("lsb_steps", 1000))
    lsb_sigma = float(p.get("lsb_sigma", 1.0))
    lsb_delta = float(p.get("lsb_delta", 1.0))
    fast_anneal_time_ns = float(p.get("fast_anneal_time_ns", 7.0))

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

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
    trainer = Trainer(wave_fn, ising, sampler_obj, trainer_config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    energies = [float(e) for e in history["energy"]]
    times = [float(t) for t in history["total_sampling_time_s"]]
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
        save_results(args, history, ising, rbm=wave_fn, energy_j=trainer.total_energy_j)

    return {
        "seed": seed,
        "exact_energy": exact,
        "final_energy": energies[-1] if energies else None,
        "tail_mean_energy": tail_mean,
        "rel_error": rel_error,
        "abs_error": abs_error,
        "wall_time_s": elapsed,
        "energy_j": trainer.total_energy_j,
        "diverged": diverged,
    }, energies, times


# ---------------------------------------------------------------------------
# One (N, phys_params) combo — all methods
# ---------------------------------------------------------------------------


def run_combo(
    *,
    hamiltonian: str,
    N: int,
    phys_params: dict,
    combo_key: str,
    best_per_sampler: dict[str, dict],
    n_seeds: int,
    n_iterations: int,
    epsilon: list[float],
    window: int,
    output_base: Path,
) -> None:
    ite_out = output_base / "ite" / hamiltonian / combo_key
    ite_out.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "hamiltonian": hamiltonian,
        "N": N,
        "phys_params": phys_params,
        "combo_key": combo_key,
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
        seed_times: list[list[float]] = []

        for seed in range(n_seeds):
            print(f"    seed {seed + 1}/{n_seeds} ...", end="\r", flush=True)
            try:
                metrics, energies, times = run_ite_trial(
                    N=N,
                    hamiltonian=hamiltonian,
                    phys_params=phys_params,
                    best_row=best_row,
                    seed=seed,
                    n_iterations=n_iterations,
                    output_dir=output_base,
                )
                seed_metrics.append(metrics)
                seed_energies.append(energies)
                seed_times.append(times)
            except Exception as exc:
                print(f"\n    seed {seed} FAILED: {exc}")

        n_ok = sum(1 for m in seed_metrics if not m["diverged"])
        print(f"    {n_ok}/{len(seed_metrics)} seeds OK                    ")

        exact = next(
            (m["exact_energy"] for m in seed_metrics if m["exact_energy"] is not None),
            None,
        )

        ite_by_eps: dict = {}
        if exact is not None:
            for eps in epsilon:
                ite_vals: list[int | None] = [
                    None if m["diverged"] else compute_ite(e, exact, eps, window)
                    for m, e in zip(seed_metrics, seed_energies)
                ]
                time_vals: list[float | None] = []
                energy_vals: list[float | None] = []
                for m, t, it in zip(seed_metrics, seed_times, ite_vals):
                    if it is None:
                        time_vals.append(None)
                        energy_vals.append(None)
                        continue
                    cum_time = sum(t[:it])
                    time_vals.append(cum_time)
                    total_time = sum(t)
                    energy_j = m.get("energy_j")
                    if energy_j is not None and total_time > 0:
                        energy_vals.append((energy_j / 3600.0) * (cum_time / total_time))
                    else:
                        energy_vals.append(None)

                ite_by_eps[str(eps)] = {
                    "epsilon": eps,
                    "window": window,
                    **_ite_stats(ite_vals, n_iterations),
                    "time_to_ite_s": _continuous_stats(time_vals),
                    "energy_wh_to_ite": _continuous_stats(energy_vals),
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
            "ite": ite_by_eps,
        }

    summary_path = ite_out / "ite_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [{combo_key}] summary → {summary_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterations-to-epsilon (ITE) benchmark using globally best hparam configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hamiltonian",
        choices=list(HAMILTONIAN_REGISTRY.keys()),
        help="Hamiltonian to process (use --all for every hamiltonian)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all hamiltonians",
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
        help="Filter to specific sampling methods",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Random seeds per config (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="SR iterations per run (default: 300)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[0.01, 0.001],
        help="Rel-error threshold(s) for ITE (default: 0.01 0.001)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling-mean window for ITE computation (default: 10)",
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

    if cli.all:
        hamiltonians = [d.name for d in sorted(hparam_base.iterdir()) if d.is_dir()]
    elif cli.hamiltonian:
        hamiltonians = [cli.hamiltonian]
    else:
        print("Specify --hamiltonian <name> or --all")
        return

    for hamiltonian in hamiltonians:
        print(f"\n=== {hamiltonian} ===")
        all_trials = load_all_trials(hamiltonian, hparam_base)
        if all_trials.empty:
            print("  No trials found — skipping")
            continue

        best_map = global_best_per_combo_sampler(
            all_trials,
            methods_filter=cli.sampling_methods,
            N_filter=cli.N,
        )
        if not best_map:
            print("  No completed trials after filtering — skipping")
            continue

        combos: dict[tuple, dict[str, dict]] = {}
        for (N, phys_key, method), best_row in best_map.items():
            combos.setdefault((N, phys_key), {})[method] = best_row

        print(f"  {len(all_trials)} total trials  |  {len(combos)} (N, phys) combos  |  {len(best_map)} (combo, method) entries")
        print(f"  Seeds={cli.n_seeds}  Iterations={cli.iterations}  Epsilon={cli.epsilon}  Window={cli.window}")

        if cli.dry_run:
            for (N, phys_key), methods in sorted(combos.items()):
                phys_params = json.loads(phys_key)
                key = _combo_key(hamiltonian, N, phys_params)
                print(f"  {key}: {sorted(methods)}")
            continue

        for (N, phys_key), best_per_sampler in sorted(combos.items()):
            phys_params = json.loads(phys_key)
            key = _combo_key(hamiltonian, N, phys_params)
            print(f"\n[{key}]")
            run_combo(
                hamiltonian=hamiltonian,
                N=N,
                phys_params=phys_params,
                combo_key=key,
                best_per_sampler=best_per_sampler,
                n_seeds=cli.n_seeds,
                n_iterations=cli.iterations,
                epsilon=cli.epsilon,
                window=cli.window,
                output_base=output_base,
            )


if __name__ == "__main__":
    main()

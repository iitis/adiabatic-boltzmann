#!/usr/bin/env python3
"""
j1j2_convergence_median.py — paper figure: median RBM convergence curves on the
frustrated J1-J2 Heisenberg chain (replaces the old best-of-run figure
fig_convergence_best from scripts/viz/plot_j1j2_convergence_curves.py).

Statistically fair protocol (referee point 5):

  Tuning set      All Optuna trials pooled from results/hparam_search/
                  heisenberg_j1j2_1d/ (same pool ite_run.py uses).
  Selection       For each (N, J2) cell, the gibbs trial with the lowest
                  tuning rel_error — via ite_run.global_best_per_combo_sampler,
                  i.e. the SAME rule that produced the existing production
                  sweeps, so the selected configuration is derived from the
                  tuning data, never from the evaluation runs. The selected
                  config (study, trial, hyperparameters) is written to the
                  summary JSON so it can be reported alongside the figure.
  Evaluation      Fresh seeds disjoint from every seed used during tuning
                  (tuning used seeds {1, 42, 123}; those are excluded from
                  both the run list and the plot). Each cell is topped up to
                  --n-seeds evaluation seeds; existing result files that match
                  the selected config exactly are reused, missing seeds are
                  run in-process (Trainer/ClassicalSampler, one JAX process,
                  serial — never shell out to scripts/main.py).
  Failures        Seeds that diverge (non-finite energy) produce no result
                  file (save_results is skipped, matching ite_run.py), but
                  every attempt and its outcome is recorded in the summary
                  JSON and the per-panel n in the figure reports
                  plotted/attempted, so failed runs are retained as censored
                  observations rather than silently dropped.
  Aggregation     Per panel: pointwise median E/N across seeds with the
                  interquartile band (p25-p75); no best-of anywhere.

J2 values are matched exactly (tolerance 1e-6). The old figure bucketed by
round(J2, 1), which silently mixed J2=0.45 tuning probes into the J2=0.5
panel; this script does not.

Output:
    results/j1j2_convergence_median/summary.json   (protocol + per-seed outcomes)
    plots/j1j2/fig_convergence_median.{pdf,png}

Usage:
    python scripts/j1j2/j1j2_convergence_median.py             # top up + plot
    python scripts/j1j2/j1j2_convergence_median.py --plot-only
    python scripts/j1j2/j1j2_convergence_median.py --dry-run
"""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "scripts" / "hparam"))
sys.path.insert(0, str(_REPO / "scripts" / "ite"))
sys.path.insert(0, str(_REPO / "scripts" / "viz"))

import numpy as np

# ---------------------------------------------------------------------------
# Figure grid and protocol constants
# ---------------------------------------------------------------------------

HAMILTONIAN = "heisenberg_j1j2_1d"
METHOD = "gibbs"
N_ROWS = [8, 12]
J2_COLS = [0.3, 0.5, 0.7, 0.9]

# Excluded from evaluation to keep tuning/eval disjoint.
TUNING_SEEDS = {1, 42, 123}

# Fields an existing result file must match to count as this configuration.
_MATCH_KEYS = ("learning_rate", "regularization", "n_samples", "n_hidden")

SUMMARY_DIR = _REPO / "results" / "j1j2_convergence_median"
PLOTS_DIR = _REPO / "plots" / "j1j2"


def eval_seeds(n_seeds: int) -> list[int]:
    """First n_seeds non-negative integers that were never used in tuning."""
    seeds, s = [], 0
    while len(seeds) < n_seeds:
        if s not in TUNING_SEEDS:
            seeds.append(s)
        s += 1
    return seeds


# ---------------------------------------------------------------------------
# Config selection (tuning data only)
# ---------------------------------------------------------------------------

def select_configs(hparam_base: Path) -> dict[tuple[int, float], dict]:
    """
    {(N, J2): best_row} for METHOD, selected by lowest tuning objective —
    exactly ite_run.py's selection rule applied to this figure's grid.
    """
    from ite_run import global_best_per_combo_sampler, load_all_trials

    trials = load_all_trials(HAMILTONIAN, hparam_base)
    if trials.empty:
        raise RuntimeError(f"No tuning trials under {hparam_base / HAMILTONIAN}")

    best = global_best_per_combo_sampler(
        trials, methods_filter=[METHOD], N_filter=N_ROWS
    )
    configs: dict[tuple[int, float], dict] = {}
    for (N, phys_key, _method), row in best.items():
        j2 = json.loads(phys_key).get("J2")
        for j2_target in J2_COLS:
            if j2 is not None and abs(j2 - j2_target) < 1e-6:
                configs[(N, j2_target)] = row
    missing = [(N, j2) for N in N_ROWS for j2 in J2_COLS if (N, j2) not in configs]
    if missing:
        raise RuntimeError(f"No {METHOD} tuning trials for cells: {missing}")
    return configs


# ---------------------------------------------------------------------------
# Existing evaluation runs
# ---------------------------------------------------------------------------

def load_existing_runs(results_root: Path, N: int, j2: float,
                       best_row: dict) -> dict[int, dict]:
    """
    {seed: {energies, exact_energy}} for result files whose config matches
    the selected configuration exactly (and J2 exactly — no rounding).
    """
    run_dir = results_root / HAMILTONIAN / str(N) / "custom" / METHOD
    if not run_dir.exists():
        return {}

    p = best_row["params"]
    expected = {
        "learning_rate": p["learning_rate"],
        "regularization": p["regularization"],
        "n_samples": p["n_samples"],
        "n_hidden": int(best_row["n_hidden"]),
    }

    runs: dict[int, dict] = {}
    for f in sorted(run_dir.glob("*.json.gz")):
        try:
            with gzip.open(f) as fp:
                d = json.load(fp)
        except Exception as exc:
            print(f"  [skip] {f.name}: {exc}")
            continue
        cfg = d.get("config", {})
        if abs(float(cfg.get("J2", math.inf)) - j2) > 1e-6:
            continue
        if any(not _close(cfg.get(k), expected[k]) for k in _MATCH_KEYS):
            continue
        energies = d.get("history", {}).get("energy")
        exact = d.get("exact_energy")
        if not energies or exact is None:
            continue
        seed = int(cfg.get("seed", -1))
        runs[seed] = {"energies": [float(e) for e in energies],
                      "exact_energy": float(exact)}
    return runs


def _close(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(b, float):
        return math.isclose(float(a), b, rel_tol=1e-12, abs_tol=0.0)
    return a == b


# ---------------------------------------------------------------------------
# Top-up runs (in-process, serial)
# ---------------------------------------------------------------------------

def top_up_cell(N: int, j2: float, best_row: dict, seeds_to_run: list[int],
                iterations: int, results_root: Path) -> list[dict]:
    """Run missing evaluation seeds for one cell. Returns per-seed outcomes."""
    from ite_run import run_ite_trial

    outcomes = []
    for seed in seeds_to_run:
        print(f"  running N={N} J2={j2} seed={seed} ...", flush=True)
        metrics, _energies, _times = run_ite_trial(
            N=N,
            hamiltonian=HAMILTONIAN,
            phys_params=best_row["phys_params"],
            best_row=best_row,
            seed=seed,
            n_iterations=iterations,
            output_dir=results_root,
        )
        status = "diverged" if metrics["diverged"] else "ok"
        print(f"    -> {status}  rel_error={metrics['rel_error']:.4g}  "
              f"({metrics['wall_time_s']:.0f}s)", flush=True)
        outcomes.append({
            "seed": seed,
            "diverged": metrics["diverged"],
            "rel_error": None if math.isinf(metrics["rel_error"]) else metrics["rel_error"],
            "wall_time_s": metrics["wall_time_s"],
        })
    return outcomes


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(cell_runs: dict[tuple[int, float], dict[int, dict]],
         cell_attempts: dict[tuple[int, float], dict],
         out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from plot_style import setup_style

    setup_style(fontsize=16, scale=1.0)
    fig, axes = plt.subplots(
        len(N_ROWS), len(J2_COLS),
        figsize=(20, 6),
        sharex=False,
        gridspec_kw={"hspace": 0.52, "wspace": 0.42},
    )

    for col, j2 in enumerate(J2_COLS):
        for row, N in enumerate(N_ROWS):
            ax = axes[row][col]
            runs = {s: r for s, r in cell_runs.get((N, j2), {}).items()
                    if s not in TUNING_SEEDS}

            if col == 0:
                ax.set_title(rf"$N={N}$", pad=4)
            ax.text(0.04, 0.96, rf"$J_2/J_1={j2}$",
                    ha="left", va="top", transform=ax.transAxes,
                    fontsize="small", color="#111")
            if row == len(N_ROWS) - 1:
                ax.set_xlabel("Iteration")
            if col == 0:
                ax.set_ylabel("$E/N$")

            if not runs:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#888")
                continue

            exact = next(iter(runs.values()))["exact_energy"] / N
            t_min = min(len(r["energies"]) for r in runs.values())
            curves = np.array([r["energies"][:t_min] for r in runs.values()]) / N
            iters = np.arange(t_min)

            med = np.median(curves, axis=0)
            p25 = np.percentile(curves, 25, axis=0)
            p75 = np.percentile(curves, 75, axis=0)

            ax.fill_between(iters, p25, p75, color="#93c5fd", alpha=0.55,
                            linewidth=0, label="IQR")
            ax.plot(iters, med, color="#1d4ed8", lw=1.6, label="median")
            ax.axhline(exact, color="#dc2626", lw=1.3, ls="--")

            # Median tail (last 20%) relative error across seeds
            tail = curves[:, int(0.8 * t_min):].mean(axis=1)
            rel_errs = np.abs(tail - exact) / np.abs(exact)
            eps_med = float(np.median(rel_errs))

            # Y limits: converge region around exact energy and median tail
            e_min = min(exact, float(med[-max(1, t_min // 3):].min()))
            e_max = float(np.percentile(med, 90))
            margin = abs(exact) * 0.08
            ax.set_ylim(e_min - margin, max(e_max, exact + margin))

            attempted = cell_attempts.get((N, j2), {})
            n_attempted = attempted.get("n_attempted", len(runs))
            n_diverged = attempted.get("n_diverged", 0)
            n_note = (f"$n={len(runs)}/{n_attempted}$"
                      if n_diverged else f"$n={len(runs)}$")
            ax.text(0.96, 0.96,
                    n_note + "\n" + rf"$\varepsilon_{{\rm med}}={eps_med:.4f}$",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize="small", color="#333",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.5))

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"fig_convergence_median.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Evaluation seeds per cell (tuning seeds excluded)")
    parser.add_argument("--iterations", type=int, default=300,
                        help="SR iterations per run (matches existing sweeps)")
    parser.add_argument("--results", type=Path, default=_REPO / "results",
                        help="Base results directory")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip top-up runs; plot existing data only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selected configs and missing seeds, then exit")
    cli = parser.parse_args()

    hparam_base = cli.results / "hparam_search"
    configs = select_configs(hparam_base)
    target_seeds = eval_seeds(cli.n_seeds)

    # Per-cell attempt ledger; must survive across --plot-only runs.
    summary_path = SUMMARY_DIR / "summary.json"
    prev_attempts: dict[str, dict[str, str]] = {}
    if summary_path.exists():
        prev = json.loads(summary_path.read_text())
        for cell, d in prev.get("cells", {}).items():
            prev_attempts[cell] = dict(d.get("attempts", {}))

    summary = {
        "hamiltonian": HAMILTONIAN,
        "method": METHOD,
        "protocol": {
            "selection": "lowest tuning rel_error per (N, J2) over all pooled "
                         "Optuna studies (ite_run.global_best_per_combo_sampler)",
            "tuning_seeds_excluded_from_eval": sorted(TUNING_SEEDS),
            "eval_seeds": target_seeds,
            "iterations": cli.iterations,
        },
        "cells": {},
    }

    cell_runs: dict[tuple[int, float], dict[int, dict]] = {}
    cell_attempts: dict[tuple[int, float], dict] = {}

    for N in N_ROWS:
        for j2 in J2_COLS:
            row = configs[(N, j2)]
            existing = load_existing_runs(cli.results, N, j2, row)
            eval_existing = {s: r for s, r in existing.items()
                             if s not in TUNING_SEEDS}
            missing = [s for s in target_seeds if s not in eval_existing]

            p = row["params"]
            print(f"\nN={N} J2={j2}: selected trial {row['trial']} "
                  f"(tuning rel_error={row['rel_error']:.4g}) "
                  f"lr={p['learning_rate']:.4g} reg={p['regularization']:.4g} "
                  f"ns={p['n_samples']} nh={int(row['n_hidden'])}")
            print(f"  existing eval seeds: {sorted(eval_existing)}")
            print(f"  missing eval seeds : {missing}")

            outcomes = []
            if missing and not cli.plot_only and not cli.dry_run:
                outcomes = top_up_cell(N, j2, row, missing,
                                       cli.iterations, cli.results)
                existing = load_existing_runs(cli.results, N, j2, row)
                eval_existing = {s: r for s, r in existing.items()
                                 if s not in TUNING_SEEDS}

            cell_key = f"N{N}_J2{j2}"
            attempts = dict(prev_attempts.get(cell_key, {}))
            attempts.update({str(s): "ok" for s in eval_existing})
            attempts.update({
                str(o["seed"]): "diverged" if o["diverged"] else "ok"
                for o in outcomes
            })
            n_diverged = sum(1 for v in attempts.values() if v == "diverged")

            cell_runs[(N, j2)] = eval_existing
            cell_attempts[(N, j2)] = {
                "n_attempted": len(attempts),
                "n_diverged": n_diverged,
            }
            summary["cells"][cell_key] = {
                "selected_trial": {
                    "trial": int(row["trial"]),
                    "n_hidden": int(row["n_hidden"]),
                    "params": row["params"],
                    "tuning_rel_error": row["rel_error"],
                },
                "eval_seeds_plotted": sorted(eval_existing),
                "attempts": attempts,
                "n_diverged": n_diverged,
            }

    if cli.dry_run:
        print("\n[dry-run] no runs executed, no plot written.")
        return

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to {summary_path}")

    plot(cell_runs, cell_attempts, PLOTS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

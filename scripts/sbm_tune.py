#!/usr/bin/env python3
"""
SBM hyperparameter grid search for Ising RBM VMC.

Phase 1  — Grid search over discrete parameters:
    mode        : discrete | ballistic
    heated      : True | False
    max_steps   : configurable list
    cem         : True | False

Phase 2 (optional, --optuna) — Optuna over continuous sb.set_env parameters:
    time_step, pressure_slope, heat_coefficient
  using the best (mode, heated, max_steps) found in Phase 1.

All experiments run in parallel.  Results saved to results/sbm_tune.json and
a summary CSV + plot in plots/sbm_tune/.

Run from repo root:
    python scripts/sbm_tune.py
    python scripts/sbm_tune.py --workers 4 --no-2d
    python scripts/sbm_tune.py --optuna --optuna-trials 60
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import ClassicalSampler

# ── grid definition ───────────────────────────────────────────────────────────

SB_MODES      = ["discrete", "ballistic"]
SB_HEATED     = [False, True]
SB_MAX_STEPS  = [100, 300, 500, 1000, 2000, 5000]
CEM_OPTIONS   = [False, True]

# Test cases: (model, size, h)
TEST_CASES_1D = [
    ("1d",  8, 0.5), ("1d",  8, 1.0), ("1d",  8, 2.0),
    ("1d", 16, 0.5), ("1d", 16, 1.0), ("1d", 16, 2.0),
]
TEST_CASES_2D = [
    ("2d", 4, 0.5), ("2d", 4, 1.0), ("2d", 4, 2.0),
    ("2d", 6, 0.5), ("2d", 6, 1.0), ("2d", 6, 2.0),
]

SEEDS       = [1, 42]
ITERATIONS  = 50   # short run — enough to rank parameters, not for publication
N_SAMPLES   = 1000
LR          = 0.1
REG         = 1e-5
CEM_INTERVAL   = 5
CEM_N_SAMPLES  = 200

OUTPUT_JSON = ROOT / "results" / "sbm_tune.json"
PLOTS_DIR   = ROOT / "plots" / "sbm_tune"

_all_results: list[dict] = []
_done = 0
_failed = 0


# ── helpers ───────────────────────────────────────────────────────────────────


def tlog(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _unique_ratio(v: np.ndarray) -> float:
    return len(set(map(tuple, v.tolist()))) / max(len(v), 1)


# ── single experiment ─────────────────────────────────────────────────────────


def run_experiment(
    mode: str,
    heated: bool,
    max_steps: int,
    use_cem: bool,
    model: str,
    size: int,
    h: float,
    seed: int,
) -> dict:
    """
    Run one short VMC training and return metrics dict.
    All stdout from the training loop is suppressed to avoid interleaved output.
    """
    np.random.seed(seed)
    n_visible = size if model == "1d" else size * size
    n_hidden  = n_visible

    if model == "1d":
        ising = TransverseFieldIsing1D(size, h)
    else:
        ising = TransverseFieldIsing2D(size, h)

    rbm = FullyConnectedRBM(n_visible, n_hidden)
    sampler = ClassicalSampler(
        method="sbm",
        sb_mode=mode,
        sb_heated=heated,
        sb_max_steps=max_steps,
    )

    # ── initial diversity (untrained RBM) ─────────────────────────────────────
    captured = io.StringIO()
    stderr_sink = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(stderr_sink):
        v_init, _ = sampler._sbm_sample(rbm, N_SAMPLES, {})
    init_unique = _unique_ratio(v_init)

    # ── training ──────────────────────────────────────────────────────────────
    trainer_config = {
        "learning_rate":  LR,
        "n_iterations":   ITERATIONS,
        "n_samples":      N_SAMPLES,
        "regularization": REG,
        "use_cem":        use_cem,
        "cem_interval":   CEM_INTERVAL,
        "cem_n_samples":  CEM_N_SAMPLES,
    }
    trainer = Trainer(rbm, ising, sampler, trainer_config)

    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(stderr_sink):
        history = trainer.train()

    # ── final diversity (trained RBM) ─────────────────────────────────────────
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(stderr_sink):
        v_final, _ = sampler._sbm_sample(rbm, N_SAMPLES, {})
    final_unique = _unique_ratio(v_final)

    # ── metrics ───────────────────────────────────────────────────────────────
    exact      = float(ising.exact_ground_energy())
    final_e    = float(history["energy"][-1])
    rel_error  = abs(final_e - exact) / abs(exact) * 100

    beta_eff_vals = [v for v in history.get("beta_eff_cem", []) if v is not None]
    beta_eff_mean = float(np.mean(beta_eff_vals))  if beta_eff_vals else None
    beta_eff_std  = float(np.std(beta_eff_vals))   if beta_eff_vals else None

    # Diversity trend: mean unique ratio estimated from unique counts logged
    # in captured stdout (lines containing "[SBM]   mode=")
    sbm_lines = [l for l in captured.getvalue().splitlines() if "[SBM]" in l]
    unique_counts = []
    for line in sbm_lines:
        try:
            # format: "  [SBM]   mode=X heated=Y unique=Z/W"
            part = line.split("unique=")[1]
            num, denom = part.strip().split("/")
            unique_counts.append(int(num) / int(denom))
        except (IndexError, ValueError):
            pass
    mean_unique_ratio = float(np.mean(unique_counts)) if unique_counts else init_unique

    return {
        "mode":             mode,
        "heated":           heated,
        "max_steps":        max_steps,
        "cem":              use_cem,
        "model":            model,
        "size":             size,
        "h":                h,
        "seed":             seed,
        "rel_error":        rel_error,
        "final_energy":     final_e,
        "exact_energy":     exact,
        "init_unique":      init_unique,
        "final_unique":     final_unique,
        "mean_unique":      mean_unique_ratio,
        "beta_eff_mean":    beta_eff_mean,
        "beta_eff_std":     beta_eff_std,
        "energy_curve":     [float(e) for e in history["energy"]],
    }


def worker(exp: dict, idx: int, total: int) -> dict | None:
    label = (
        f"[{idx:>4}/{total}] "
        f"mode={exp['mode']:<10} heated={str(exp['heated']):<5} "
        f"steps={exp['max_steps']:<5} cem={str(exp['use_cem']):<5} "
        f"{exp['model']} N={exp['size']} h={exp['h']} seed={exp['seed']}"
    )
    try:
        result = run_experiment(**exp)
        tlog(f"OK  {label}  → err={result['rel_error']:.2f}%  "
             f"unique={result['mean_unique']:.3f}")
        return result
    except Exception:
        tlog(f"FAIL {label}\n{traceback.format_exc()}")
        return None


# ── Phase 1: grid search ──────────────────────────────────────────────────────


def build_grid(test_cases: list[tuple]) -> list[dict]:
    experiments = []
    for mode, heated, max_steps, cem, (model, size, h), seed in itertools.product(
        SB_MODES, SB_HEATED, SB_MAX_STEPS, CEM_OPTIONS, test_cases, SEEDS
    ):
        experiments.append(dict(
            mode=mode, heated=heated, max_steps=max_steps, use_cem=cem,
            model=model, size=size, h=h, seed=seed,
        ))
    return experiments


def run_grid(experiments: list[dict], workers: int) -> None:
    global _all_results, _done, _failed
    total = len(experiments)
    tlog(f"Phase 1 — grid search: {total} experiments on {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker, exp, i + 1, total): exp
            for i, exp in enumerate(experiments)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                _all_results.append(result)
                _done += 1
            else:
                _failed += 1


# ── Phase 2: Optuna continuous search ────────────────────────────────────────


def run_optuna(best_params: dict, test_cases: list[tuple], n_trials: int,
               workers: int) -> None:
    try:
        import optuna
        import simulated_bifurcation as sb
    except ImportError as e:
        tlog(f"Phase 2 skipped — missing dependency: {e}")
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mode      = best_params["mode"]
    heated    = best_params["heated"]
    max_steps = best_params["max_steps"]
    use_cem   = best_params["cem"]

    def objective(trial: optuna.Trial) -> float:
        time_step       = trial.suggest_float("time_step",       0.01, 2.0,  log=True)
        pressure_slope  = trial.suggest_float("pressure_slope",  0.001, 0.5, log=True)
        heat_coefficient= trial.suggest_float("heat_coefficient",0.001, 0.5, log=True)

        sb.set_env(
            time_step=time_step,
            pressure_slope=pressure_slope,
            heat_coefficient=heat_coefficient,
        )

        errors = []
        for model, size, h in test_cases[:4]:   # small subset for speed
            for seed in SEEDS:
                np.random.seed(seed)
                n_visible = size if model == "1d" else size * size
                ising = (TransverseFieldIsing1D(size, h) if model == "1d"
                         else TransverseFieldIsing2D(size, h))
                rbm     = FullyConnectedRBM(n_visible, n_visible)
                sampler = ClassicalSampler(method="sbm", sb_mode=mode,
                                           sb_heated=heated, sb_max_steps=max_steps)
                trainer_config = {
                    "learning_rate": LR, "n_iterations": ITERATIONS,
                    "n_samples": N_SAMPLES, "regularization": REG,
                    "use_cem": use_cem, "cem_interval": CEM_INTERVAL,
                    "cem_n_samples": CEM_N_SAMPLES,
                }
                trainer = Trainer(rbm, ising, sampler, trainer_config)
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    history = trainer.train()
                exact  = float(ising.exact_ground_energy())
                errors.append(abs(history["energy"][-1] - exact) / abs(exact) * 100)

        sb.reset_env()
        mean_err = float(np.mean(errors))
        tlog(f"  [optuna] trial {trial.number}  "
             f"dt={time_step:.3f} ps={pressure_slope:.3f} hc={heat_coefficient:.3f}"
             f"  → err={mean_err:.2f}%")
        return mean_err

    tlog(f"Phase 2 — Optuna ({n_trials} trials) with best params: {best_params}")
    study = optuna.create_study(
        direction="minimize",
        study_name="sbm_env_tune",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=min(workers, 4))

    best = study.best_trial
    tlog(f"  Best env params: {best.params}  → err={best.value:.2f}%")

    optuna_results = {
        "best_params":  best.params,
        "best_value":   best.value,
        "fixed_params": best_params,
        "all_trials": [
            {"number": t.number, "params": t.params, "value": t.value}
            for t in study.trials if t.value is not None
        ],
    }
    out = OUTPUT_JSON.with_name("sbm_tune_optuna.json")
    out.write_text(json.dumps(optuna_results, indent=2))
    tlog(f"Optuna results → {out}")


# ── summary + plot ────────────────────────────────────────────────────────────


def summarise(results: list[dict]) -> None:
    if not results:
        return

    import pandas as pd  # type: ignore

    df = pd.DataFrame(results)

    # Aggregate over seeds and test cases → mean rel_error and mean_unique per
    # (mode, heated, max_steps, cem) combo
    group_cols = ["mode", "heated", "max_steps", "cem"]
    agg = (
        df.groupby(group_cols)
        .agg(
            rel_error_mean=("rel_error",  "mean"),
            rel_error_std =("rel_error",  "std"),
            unique_mean   =("mean_unique","mean"),
            n_runs        =("rel_error",  "count"),
        )
        .reset_index()
        .sort_values("rel_error_mean")
    )

    print("\n" + "=" * 90)
    print("SBM HYPERPARAMETER TUNING — LEADERBOARD  (sorted by mean rel. energy error)")
    print("=" * 90)
    print(f"{'mode':<12} {'heated':<7} {'steps':<7} {'cem':<6} "
          f"{'err_mean%':>10} {'err_std':>8} {'diversity':>10} {'n':>4}")
    print("-" * 90)
    for _, row in agg.iterrows():
        print(
            f"{row['mode']:<12} {str(row['heated']):<7} {int(row['max_steps']):<7} "
            f"{str(row['cem']):<6} {row['rel_error_mean']:>10.3f} "
            f"{row['rel_error_std']:>8.3f} {row['unique_mean']:>10.4f} "
            f"{int(row['n_runs']):>4}"
        )
    print("=" * 90)

    best = agg.iloc[0]
    print(f"\nBest combo: mode={best['mode']}  heated={best['heated']}  "
          f"max_steps={int(best['max_steps'])}  cem={best['cem']}")
    print(f"  mean error = {best['rel_error_mean']:.3f}%  "
          f"diversity = {best['unique_mean']:.4f}\n")

    # Save summary CSV
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(PLOTS_DIR / "summary.csv", index=False)

    # Plot: error vs max_steps, one line per (mode, heated, cem) combo
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("SBM hyperparameter sweep — averaged over models, sizes, h, seeds",
                     fontsize=13, fontweight="bold")

        styles = {
            ("discrete",  False, False): ("tab:blue",   "-",  "disc / cold / no-CEM"),
            ("discrete",  False, True ): ("tab:blue",   "--", "disc / cold / CEM"),
            ("discrete",  True,  False): ("tab:orange", "-",  "disc / heated / no-CEM"),
            ("discrete",  True,  True ): ("tab:orange", "--", "disc / heated / CEM"),
            ("ballistic", False, False): ("tab:green",  "-",  "ball / cold / no-CEM"),
            ("ballistic", False, True ): ("tab:green",  "--", "ball / cold / CEM"),
            ("ballistic", True,  False): ("tab:red",    "-",  "ball / heated / no-CEM"),
            ("ballistic", True,  True ): ("tab:red",    "--", "ball / heated / CEM"),
        }

        for (mode, heated, cem), (color, ls, label) in styles.items():
            sub = agg[(agg.mode == mode) & (agg.heated == heated) & (agg.cem == cem)]
            if sub.empty:
                continue
            sub = sub.sort_values("max_steps")
            ax1.plot(sub["max_steps"], sub["rel_error_mean"],
                     color=color, linestyle=ls, marker="o", linewidth=2, label=label)
            ax2.plot(sub["max_steps"], sub["unique_mean"],
                     color=color, linestyle=ls, marker="o", linewidth=2, label=label)

        for ax, ylabel, title in [
            (ax1, "mean rel. error (%)", "Energy error vs SBM steps"),
            (ax2, "mean unique sample ratio", "Sample diversity vs SBM steps"),
        ]:
            ax.set_xscale("log")
            ax.set_xlabel("max_steps", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        fname = PLOTS_DIR / "sbm_tune_summary.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        tlog(f"Summary plot → {fname}")

    except ImportError:
        tlog("matplotlib not available — skipping plot")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SBM hyperparameter grid search")
    parser.add_argument("--workers",  type=int, default=os.cpu_count(),
                        help="Parallel workers (default: cpu count)")
    parser.add_argument("--no-2d",   action="store_true",
                        help="Skip 2D models (faster)")
    parser.add_argument("--no-cem",  action="store_true",
                        help="Skip CEM variants (halves number of experiments)")
    parser.add_argument("--optuna",  action="store_true",
                        help="Run Phase 2 Optuna search after grid search")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment list without running")
    args = parser.parse_args()

    test_cases = TEST_CASES_1D + ([] if args.no_2d else TEST_CASES_2D)
    global CEM_OPTIONS
    if args.no_cem:
        CEM_OPTIONS = [False]

    experiments = build_grid(test_cases)
    total = len(experiments)

    tlog("=" * 70)
    tlog(f"SBM tune started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    tlog(f"Workers          : {args.workers}")
    tlog(f"Experiments      : {total}")
    tlog(f"  modes          : {SB_MODES}")
    tlog(f"  heated         : {SB_HEATED}")
    tlog(f"  max_steps      : {SB_MAX_STEPS}")
    tlog(f"  CEM            : {CEM_OPTIONS}")
    tlog(f"  test cases     : {len(test_cases)}  ×  seeds={SEEDS}")
    tlog(f"  iterations/run : {ITERATIONS}")
    tlog("=" * 70)

    if args.dry_run:
        for i, exp in enumerate(experiments):
            print(f"  [{i+1:>4}] {exp}")
        return

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    run_grid(experiments, args.workers)

    # Save raw results
    OUTPUT_JSON.write_text(json.dumps(_all_results, indent=2))
    tlog(f"Raw results → {OUTPUT_JSON}  ({len(_all_results)} records)")

    summarise(_all_results)

    if args.optuna and _all_results:
        # Pick best (mode, heated, max_steps, cem) from Phase 1 for Phase 2
        best_by_error = sorted(_all_results, key=lambda r: r["rel_error"])
        best_params = {
            "mode":      best_by_error[0]["mode"],
            "heated":    best_by_error[0]["heated"],
            "max_steps": best_by_error[0]["max_steps"],
            "cem":       best_by_error[0]["cem"],
        }
        run_optuna(best_params, test_cases, args.optuna_trials, args.workers)

    tlog("=" * 70)
    tlog(f"Done — {_done} succeeded, {_failed} failed")
    tlog("=" * 70)


if __name__ == "__main__":
    main()

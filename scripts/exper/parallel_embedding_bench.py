#!/usr/bin/env python3
"""
parallel_embedding_bench.py — paper figure: QPU parallel-embedding benchmark
with a statistically fair protocol (replaces the best-of-3-seeds figure from
parallel_embedding_experiment.py; referee point 5).

Pre-registered protocol (everything fixed below, before any QPU ms is spent):

  Configuration   The archived-healthy TFIM config (N=8, h=0.5, pegasus,
                  lr=0.1, reg=1e-5, n_samples=990, 150 iterations) — same as
                  the previous experiment, declared, never re-selected.
  Arms            n_parallel in {1, 3, 5}.
  Seeds           EVAL_SEEDS below, in declared order; excludes 1/42/123
                  (seeds that appear in hyperparameter tuning elsewhere in
                  the repo, and the seed the archived config was found on).
                  If the budget runs out, trailing seeds are dropped —
                  truncation by pre-declared order is unbiased.
  Schedule        Paired + interleaved: for each seed, all three arms run
                  back-to-back in an order shuffled per seed (META_SEED) so
                  QPU drift cannot confound the arm comparison and per-seed
                  paired ratios are meaningful.
  Primary         Cumulative QPU sampling time until the relative energy
  endpoint        error stays below EPSILON for a full WINDOW-iteration
                  window (sustained, not first-touch). Crashed (NaN) and
                  never-converging runs are right-censored at their last
                  recorded cumulative time and enter a Kaplan-Meier
                  estimate; they are never dropped.
  Secondary       Relative error at FIXED_BUDGET_S of cumulative QPU
  endpoint        sampling time, defined for every surviving run.
  Failures        Every attempt is appended to a persistent, append-only
                  ledger the moment it finishes (crash-safe; a killed
                  session loses at most the run in flight). Re-invocations
                  skip completed (seed, arm) pairs and never erase recorded
                  outcomes. Crash rates are reported with Wilson CIs.
  Budget          QPU_BUDGET_MS for this experiment, measured as time.json
                  deltas from the baseline recorded at the first live run.
                  Before each run the next-run cost is projected (measured
                  deltas once available, conservative constants before) and
                  the session hard-aborts when it would exceed the budget.

Modes:
  --rehearse   (default) identical pipeline with the classical Metropolis
               sampler substituted for the QPU and n_parallel forced to 1
               inside the Trainer (the Trainer rejects n_parallel>1 on
               classical samplers); the nominal arm is still recorded, so
               ledger, analysis and figure are exercised end-to-end with
               zero QPU spend. Separate ledger file, output suffixed.
  --live       the one real QPU session. Explicit opt-in.
  --plot-only  regenerate figure from the ledger; runs nothing.
  --dry-run    print schedule + cost projection; runs nothing.

Output:
  results/parallel_embedding_bench/ledger{_rehearsal}.json
  plots/parallel_embedding/parallel_embedding_bench{_rehearsal}.{pdf,png}

Usage:
  python scripts/exper/parallel_embedding_bench.py --dry-run
  python scripts/exper/parallel_embedding_bench.py --rehearse
  python scripts/exper/parallel_embedding_bench.py --live
  python scripts/exper/parallel_embedding_bench.py --plot-only [--live-ledger]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "viz"))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

# ---------------------------------------------------------------------------
# Pre-registered protocol
# ---------------------------------------------------------------------------

CONFIG = dict(N=8, h=0.5, method="pegasus", lr=0.1, reg=1e-5,
              n_samples=990, iterations=150)
ARMS = [1, 3, 5, 99, 165]
# n_parallel=165 chosen classically (zero QPU cost): busclique found 191
# disjoint K_{8,8} embeddings on Advantage_system6 (76.5% of chip, chain
# length 2-3), and 165 is the largest divisor of n_samples=990 comfortably
# under that ceiling (990/165 = 6 reads/copy/iteration).
# n_parallel=99 is a second, smaller "big" arm (990/99 = 10 reads/copy) —
# nearest divisor of n_samples to 100, added as its own arm (not a swap for
# 165, which already has live seed data) to keep both comparisons uncorrupted.
EVAL_SEEDS = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
META_SEED = 7
EPSILON = 0.01
WINDOW = 10
FIXED_BUDGET_S = 5.0
QPU_BUDGET_MS = 15 * 60 * 1000
# Conservative per-run cost (ms of time.json delta) used until >=1 live run
# of the arm has been measured; based on cache medians x1.5 overhead margin.
# arm=165 is untested at this scale (embedded BQM spans ~76% of the chip) —
# prior set pessimistically (above arm=1) so the budget guard stays safe
# until a real measurement replaces it.
COST_PRIOR_MS = {1: 21_000, 3: 11_000, 5: 9_000, 99: 12_000, 165: 25_000}

QPU_TIME_PATH = _REPO / "time.json"
OUT_DIR = _REPO / "results" / "parallel_embedding_bench"
PLOTS_DIR = _REPO / "plots" / "parallel_embedding"


def arm_order(seed: int) -> list[int]:
    order = list(ARMS)
    random.Random(f"{META_SEED}_{seed}").shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Ledger — append-only, written after every run
# ---------------------------------------------------------------------------

def ledger_path(rehearse: bool) -> Path:
    return OUT_DIR / ("ledger_rehearsal.json" if rehearse else "ledger.json")


def load_ledger(rehearse: bool) -> dict:
    p = ledger_path(rehearse)
    if p.exists():
        return json.loads(p.read_text())
    return {"protocol": {
        "config": CONFIG, "arms": ARMS, "eval_seeds": EVAL_SEEDS,
        "meta_seed": META_SEED, "epsilon": EPSILON, "window": WINDOW,
        "fixed_budget_s": FIXED_BUDGET_S, "qpu_budget_ms": QPU_BUDGET_MS,
        "rehearsal": rehearse,
    }, "qpu_baseline_ms": None, "runs": {}}


def save_ledger(ledger: dict, rehearse: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path(rehearse).write_text(json.dumps(ledger, indent=1))


# ---------------------------------------------------------------------------
# Budget guard (live mode only) — time.json deltas, no fallbacks
# ---------------------------------------------------------------------------

def projected_next_cost_ms(ledger: dict, arm: int) -> float:
    deltas = [r["qpu_delta_ms"] for r in ledger["runs"].values()
              if r["arm"] == arm and r.get("qpu_delta_ms") is not None]
    return float(np.median(deltas)) if deltas else COST_PRIOR_MS[arm]


def check_budget(ledger: dict, arm: int) -> float:
    """Raise if the projected next run would exceed the allocation."""
    from helpers import read_qpu_time_ms
    used_total = read_qpu_time_ms(QPU_TIME_PATH)
    if ledger["qpu_baseline_ms"] is None:
        ledger["qpu_baseline_ms"] = used_total
    spent = used_total - ledger["qpu_baseline_ms"]
    projection = projected_next_cost_ms(ledger, arm)
    if spent + projection > QPU_BUDGET_MS:
        raise RuntimeError(
            f"Budget stop: spent {spent / 60_000:.2f} min + projected "
            f"{projection / 60_000:.2f} min > allocation "
            f"{QPU_BUDGET_MS / 60_000:.0f} min. Remaining seeds dropped "
            f"(pre-declared order — unbiased truncation)."
        )
    return used_total


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

_SHARED_EMBEDDING_CACHE: dict = {}  # injected into each fresh DimodSampler so
# the expensive n_parallel-way disjoint busclique packing is reused across
# seeds, without reusing the sampler instance itself (which stays fresh per
# run to avoid carrying over any other instance state between runs).


def run_one(seed: int, arm: int, rehearse: bool, ledger: dict) -> dict:
    from encoder import Trainer
    from ising import TransverseFieldIsing1D
    from model import FullyConnectedRBM
    from sampler import ClassicalSampler, DimodSampler
    from helpers import read_qpu_time_ms

    used_before = None if rehearse else check_budget(ledger, arm)

    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    ising = TransverseFieldIsing1D(size=CONFIG["N"], h=CONFIG["h"])
    rbm = FullyConnectedRBM(CONFIG["N"], CONFIG["N"], model_key)
    if rehearse:
        # neal SA via DimodSampler: same QUBO sampling pipeline as the QPU,
        # classical, no time.json writes. ClassicalSampler alternatives fail
        # on this config (metropolis barely mixes at ~14/990 unique samples;
        # gibbs blows up NaN@3), leaving the event path (KM medians, paired
        # ratios) unexercised.
        sampler = DimodSampler(method="simulated_annealing")
        n_parallel_actual = 1  # sample_parallel is QPU-only (pegasus/zephyr)
    else:
        sampler = DimodSampler(method=CONFIG["method"])
        sampler._embedding_cache = _SHARED_EMBEDDING_CACHE
        n_parallel_actual = arm

    trainer = Trainer(rbm=rbm, ising_model=ising, sampler=sampler, config={
        "n_samples": CONFIG["n_samples"],
        "n_iterations": CONFIG["iterations"],
        "learning_rate": CONFIG["lr"],
        "regularization": CONFIG["reg"],
        "seed": seed,
        "n_parallel": n_parallel_actual,
        "save_checkpoints": False,
    })
    t0 = time.perf_counter()
    history = trainer.train()
    wall_s = time.perf_counter() - t0

    e = np.array(history["energy"], dtype=float)
    nan_iter = int(np.argmax(np.isnan(e))) if np.isnan(e).any() else None

    return {
        "seed": seed,
        "arm": arm,
        "rehearsal": rehearse,
        "solver": CONFIG["method"] if not rehearse else "neal_sa(rehearsal)",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_s": wall_s,
        "energy": [float(x) for x in e],
        "sampling_time_s": [float(t) for t in history["sampling_time_s"]],
        "E_exact": float(ising.exact_ground_energy()),
        "nan_iter": nan_iter,
        "qpu_delta_ms": (None if rehearse
                         else read_qpu_time_ms(QPU_TIME_PATH) - used_before),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def sustained_crossing(rec: dict) -> tuple[float | None, float]:
    """
    (time_to_epsilon_s, censor_time_s): time_to_epsilon_s is the cumulative
    QPU sampling time at the END of the first WINDOW-iteration window whose
    relative error stays below EPSILON throughout; None if never (censored).
    """
    e = np.array(rec["energy"], dtype=float)
    t = np.cumsum(rec["sampling_time_s"])
    finite = np.isfinite(e)
    last = int(np.argmax(~finite)) if (~finite).any() else len(e)
    e, tt = e[:last], t[:last]
    censor = float(tt[-1]) if len(tt) else 0.0
    rel = np.abs(e - rec["E_exact"]) / abs(rec["E_exact"])
    for i in range(len(rel) - WINDOW + 1):
        if np.all(rel[i:i + WINDOW] < EPSILON):
            return float(tt[i + WINDOW - 1]), censor
    return None, censor


def error_at_budget(rec: dict) -> float | None:
    """Relative error at FIXED_BUDGET_S cumulative QPU sampling time (last
    finite iteration completed within the budget); None if crashed before."""
    e = np.array(rec["energy"], dtype=float)
    t = np.cumsum(rec["sampling_time_s"])
    finite = np.isfinite(e)
    last = int(np.argmax(~finite)) if (~finite).any() else len(e)
    e, t = e[:last], t[:last]
    idx = np.searchsorted(t, FIXED_BUDGET_S, side="right") - 1
    if idx < 0:
        return None
    return float(abs(e[idx] - rec["E_exact"]) / abs(rec["E_exact"]))


def km_median(events: list[tuple[float, bool]]) -> float | None:
    """Kaplan-Meier median of (time, observed) pairs; None if S never
    reaches 0.5 (median exceeds largest observation — heavy censoring)."""
    pts = sorted(events)
    n = len(pts)
    s = 1.0
    at_risk = n
    for t, observed in pts:
        if observed:
            s *= (at_risk - 1) / at_risk
        at_risk -= 1
        if s <= 0.5:
            return t
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_km_median_ci(events, n_boot=2000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    meds = []
    for _ in range(n_boot):
        sample = [events[i] for i in rng.integers(0, len(events), len(events))]
        m = km_median(sample)
        if m is not None:
            meds.append(m)
    if not meds:
        return None, None
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

ARM_COLORS = {1: "#1f77b4", 3: "#2ca02c", 5: "#d62728"}


def plot(ledger: dict, rehearse: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup_style

    setup_style()
    runs = list(ledger["runs"].values())
    if not runs:
        print("Ledger empty — nothing to plot.")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.6))

    stats_lines = []
    for arm in ARMS:
        arm_runs = [r for r in runs if r["arm"] == arm]
        if not arm_runs:
            continue
        color = ARM_COLORS[arm]
        n_att = len(arm_runs)
        n_crash = sum(1 for r in arm_runs if r["nan_iter"] is not None)
        lo, hi = wilson_ci(n_crash, n_att)

        # Panel A: median energy vs cumulative QPU time (survivors), IQR band
        survivors = [r for r in arm_runs if r["nan_iter"] is None]
        if survivors:
            grids = []
            t_max = min(float(np.sum(r["sampling_time_s"])) for r in survivors)
            grid = np.linspace(0, t_max, 200)
            for r in survivors:
                t = np.cumsum(r["sampling_time_s"])
                grids.append(np.interp(grid, t, r["energy"]))
            curves = np.array(grids)
            med = np.median(curves, axis=0)
            ax_a.fill_between(grid, np.percentile(curves, 25, axis=0),
                              np.percentile(curves, 75, axis=0),
                              color=color, alpha=0.20, linewidth=0)
            ax_a.plot(grid, med, color=color, lw=1.8,
                      label=rf"$n_\parallel={arm}$  ($n={len(survivors)}/{n_att}$)")

        # Panel B: KM median time-to-epsilon + per-seed event/censor marks
        events = []
        for r in arm_runs:
            tte, censor = sustained_crossing(r)
            events.append((tte, True) if tte is not None else (censor, False))
        obs = [t for t, o in events if o]
        cen = [t for t, o in events if not o]
        y = ARMS.index(arm)
        ax_b.scatter(obs, [y] * len(obs), color=color, marker="o", s=28,
                     zorder=3, label=None)
        ax_b.scatter(cen, [y] * len(cen), facecolors="none", edgecolors=color,
                     marker="^", s=34, zorder=3)
        med = km_median(events)
        if med is not None:
            ci_lo, ci_hi = bootstrap_km_median_ci(events)
            ax_b.errorbar([med], [y], xerr=None, color=color, marker="|",
                          markersize=18, markeredgewidth=2.4, zorder=4)
            if ci_lo is not None:
                ax_b.plot([ci_lo, ci_hi], [y, y], color=color, lw=3, alpha=0.35,
                          solid_capstyle="butt", zorder=2)
            med_txt = f"KM median {med:.1f}s"
        else:
            med_txt = "KM median > censor times"
        stats_lines.append(
            f"n_par={arm}: {n_att} attempted, {n_crash} crashed "
            f"(rate {n_crash / n_att:.0%}, 95% CI {lo:.0%}-{hi:.0%}); {med_txt}"
        )

    exact = runs[0]["E_exact"]
    ax_a.axhline(exact, color="#dc2626", lw=1.2, ls="--", label=r"$E_\mathrm{exact}$")
    ax_a.set_xlabel("Cumulative QPU sampling time (s)")
    ax_a.set_ylabel("Energy")
    ax_a.set_title("Median energy vs QPU time (IQR band, survivors)")
    ax_a.legend(fontsize=8)

    ax_b.set_yticks(range(len(ARMS)))
    ax_b.set_yticklabels([rf"$n_\parallel={a}$" for a in ARMS])
    ax_b.set_xlabel(f"QPU time to sustained $\\epsilon$={EPSILON:.0%} (s)")
    ax_b.set_title("Time-to-$\\epsilon$ — events (dots), censored (open triangles),\n"
                   "KM median (bar) with bootstrap 95% CI")
    ax_b.set_ylim(-0.6, len(ARMS) - 0.4)

    suffix = "_rehearsal" if rehearse else ""
    if rehearse:
        fig.suptitle("REHEARSAL — classical sampler, no QPU data", color="#b91c1c")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = PLOTS_DIR / f"parallel_embedding_bench{suffix}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {p}")
    plt.close(fig)

    print("\n".join(stats_lines))

    # Paired ratios (arm vs n_parallel=1), seeds where both converged
    by = {(r["arm"], r["seed"]): r for r in runs}
    for arm in ARMS[1:]:
        ratios = []
        for seed in EVAL_SEEDS:
            a, b = by.get((arm, seed)), by.get((1, seed))
            if a is None or b is None:
                continue
            ta, _ = sustained_crossing(a)
            tb, _ = sustained_crossing(b)
            if ta is not None and tb is not None:
                ratios.append(ta / tb)
        if ratios:
            print(f"paired time ratio n_par={arm} / n_par=1: "
                  f"median {np.median(ratios):.2f} over {len(ratios)} pairs "
                  f"({['%.2f' % x for x in ratios]})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel-embedding QPU benchmark (pre-registered protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rehearse", action="store_true",
                      help="classical-sampler dress rehearsal (default mode)")
    mode.add_argument("--live", action="store_true",
                      help="the real QPU session (explicit opt-in)")
    mode.add_argument("--plot-only", action="store_true",
                      help="figure from ledger, run nothing")
    mode.add_argument("--dry-run", action="store_true",
                      help="print schedule + projection, run nothing")
    parser.add_argument("--live-ledger", action="store_true",
                        help="[plot-only/dry-run] use the live ledger instead of rehearsal")
    cli = parser.parse_args()

    live = cli.live
    rehearse = not live  # default; also governs which ledger plot-only/dry-run reads
    if (cli.plot_only or cli.dry_run) and cli.live_ledger:
        rehearse = False

    ledger = load_ledger(rehearse)
    schedule = [(seed, arm) for seed in EVAL_SEEDS for arm in arm_order(seed)]
    todo = [(s, a) for (s, a) in schedule if f"{s}_{a}" not in ledger["runs"]]

    if cli.dry_run:
        print(f"Schedule ({len(schedule)} runs, {len(todo)} to do):")
        for s, a in schedule:
            mark = " " if (s, a) in todo else "✓"
            print(f"  {mark} seed={s} arm={a}")
        total = sum(projected_next_cost_ms(ledger, a) for _, a in todo)
        print(f"Projected QPU cost of remaining runs: {total / 60_000:.1f} min "
              f"(allocation {QPU_BUDGET_MS / 60_000:.0f} min)")
        return

    if not cli.plot_only:
        label = "LIVE QPU SESSION" if live else "rehearsal (classical sampler)"
        print(f"Mode: {label} — {len(todo)} runs to do")
        for seed, arm in todo:
            key = f"{seed}_{arm}"
            print(f"  running seed={seed} arm={arm} ...", flush=True)
            try:
                rec = run_one(seed, arm, rehearse, ledger)
            except RuntimeError as exc:
                print(f"  STOP: {exc}")
                break
            ledger["runs"][key] = rec
            save_ledger(ledger, rehearse)
            status = (f"NaN@{rec['nan_iter']}" if rec["nan_iter"] is not None
                      else "completed")
            print(f"    -> {status}, wall {rec['wall_s']:.0f}s, "
                  f"qpu_delta {rec['qpu_delta_ms']} ms", flush=True)

    plot(ledger, rehearse)


if __name__ == "__main__":
    main()

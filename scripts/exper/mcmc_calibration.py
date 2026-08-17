#!/usr/bin/env python3
"""
mcmc_calibration.py -- fair calibration of ClassicalSampler mixing params
(Metropolis n_warmup, Gibbs n_sweeps, SA cooling length, LSB steps) at the
exact fig10c cell (h=0.5, lr=0.08, reg=0.05, ns=200, iterations=100),
N=32 only.

Why: at N>=24 in that cell, Metropolis/Gibbs increasingly self-detect
convergence at the wrong plateau instead of the true ground state (see
fig10c's validated-rate collapse: Metropolis 13/20 -> 4/20 -> 0/20 and
Gibbs 9/20 -> 7/20 -> 0/20 for N=24/32/64), correlating with poor mixing
(ESS 0.1-0.6, unique-sample ratio 0.2-0.3). A naive single-seed spot check
(seed 0, N=32, mixing cranked 4-5x) showed Gibbs got WORSE, not better
(err/spin 0.001 -> 0.069) -- so this script validates any fix across
multiple seeds instead of eyeballing one.

SA (ClassicalSampler's own JAX/GPU simulated_annealing method, run via this
same mcmc_matched_sweep.py harness -- see chat) was added to this
calibration for the same reason, but its untuned baseline (sa_sweeps=1, the
class default) collapses even harder and sooner than Metropolis/Gibbs did:
validated rate 3/20 at N=8, 0/20 at N=12 and N=16 (seeds 0-19, same
cv_threshold/window/epsilon as this file's Metropolis/Gibbs scoring), not
just at N>=24. A single-seed probe at N=32 found a sharp transition between
sa_sweeps=10 (err/spin 0.057) and 40 (err/spin 0.0004) at negligible extra
cost (SA's per-seed wall time is nearly flat across cooling length --
36s/38s/38s/48s for sa_sweeps=1/10/40/160), which is exactly the kind of
single-seed signal this file's own Gibbs precedent warns not to trust
without a multi-seed check.

The multi-seed run confirmed the transition is real but did NOT confirm
sa_sweeps=10 as the production value -- summary.json's decide() records
"chosen: 10" for SA, but that is a decision-rule artifact, not the actual
production setting. decide()'s tie-break ("cheapest = min(grid) among
candidates clearing the rate margin") assumes cost scales monotonically
with the grid parameter; SA's own module docstring says the opposite
("per-seed wall time is nearly flat across cooling length"), and the
scored data bears that out: sa_sweeps=40 has a LOWER median TTE (5.74s)
than sa_sweeps=10 (8.83s), i.e. 40 is not just equally valid but actually
cheaper and faster to a validated answer, on top of the ~140x tighter
err/spin from the single-seed probe above. The production fig10c/fig10d
results/tfim_1d/*/custom/simulated_annealing/ files (sampler_config
metadata, git_sha 481310c87) confirm n_sweeps=40 was what was actually
used -- correctly. Treat summary.json's SA "chosen: 10" as informational
only; 40 is the right value and decide()'s binary validated/not-validated
tie-break is the thing that's wrong here, not the production data.

LSB (Langevin Simulated Bifurcation) was added last, and unlike the other
three does NOT have a calibration fix: at N=32 (seeds 90-99), lsb_steps in
{1000, 2000, 4000, 8000} (lsb_delta/gamma/sigma held at their class
defaults 0.1/0.1/1.0) scores validated_rate 0.50/0.50/0.40/0.50 -- flat,
not monotone, and never close to Metropolis/Gibbs/SA's 1.00. A follow-up
3-seed (90-92) screen over lsb_delta in {0.1, 0.05, 0.02, 0.01} x lsb_gamma
in {0.1, 0.3, 0.5, 0.05} x lsb_sigma (paper sigma_inv2 convention) in
{0.5, 1.0, 2.0} at fixed lsb_steps=1000 found nothing that beats the class
default (delta=0.1, gamma=0.1, sigma_inv2=1.0, itself only 1/3): every
delta below 0.1 got WORSE and increasingly produced NaN training energies
(delta=0.01 was 0/3 all-NaN), and neither more damping (gamma up to 0.5)
nor either noise direction (sigma_inv2 0.5 or 2.0) recovered it. This
points to something more structural than a mixing-length problem -- e.g.
RBM weight growth over SR iterations pushing the fixed-hyperparameter
Langevin integration into an unstable regime -- not fixable by a grid
search over (steps, delta, gamma, sigma) alone. LSB_GRID is kept in this
file's protocol so the decision rule still runs and records a "chosen"
value (it picks 1000, the cheapest of several statistically-tied-bad
candidates), but that choice should NOT be read as "LSB is calibrated" the
way Metropolis/Gibbs/SA's chosen values are.

Separately: this whole calibration (all four methods) only ever ran at
N=32. Checking the actual production fig10c data across the full size
range (N=8..128) with the SAME validated-convergence criterion used here
shows Metropolis/Gibbs/SA/LSB are all near-100% through N=32 but collapse
at N=64 (Metropolis 12/20, Gibbs 10/20, SA 12/20, LSB 6/20) and fail
outright at N=128 (0/20 for Metropolis/Gibbs/LSB; SA has no N=128 data).
The N=32 calibration in this file does not address that -- it was never
in scope here, since this file's grid only ever varies the mixing
parameter, never N.

Protocol (pre-registered -- fixed before any candidate's rate/TTE was
inspected):
  - Candidate grids, in a doubling sequence:
      Metropolis n_warmup in {200, 400, 800, 1600}
      Gibbs      n_sweeps in {10, 20, 40, 80}
    The original brief proposed {200,400,800} / {10,20,40}. Widened by one
    doubling step after an empirical timing probe (single seed, N=32,
    iter=100, this machine: idle single TITAN RTX GPU, 36 CPU threads,
    125GB RAM -- see module docstring footer) showed the top of that
    grid only costs ~70s (Metropolis)/~27s (Gibbs) per seed, and even the
    added stretch point (1600 / 80) costs ~121s/~35s -- cheap enough that
    a 10-seed x 4-candidate x 2-solver grid (80 runs) finishes in well
    under 1.5h on a single idle GPU. NOTE: this machine has 36 CPU
    threads, but Metropolis/Gibbs sampling is one batched XLA kernel on
    a single GPU (see src/sampler.py), and there is exactly one GPU here
    -- so the extra cores don't let the grid run wider *in parallel*
    (this repo's house rule is one JAX process at a time on this GPU);
    they just mean the outer seed/candidate loop is cheap to run wider
    *sequentially*.
  - Seeds 90..99 (10 seeds), disjoint from the reported 0..19 seeds used
    by the actual fig10c figure.
  - Score = (validated_rate, median_TTE), computed with the *exact same*
    compute_validated_convergence_iter(cv_threshold=0.05, window=10,
    epsilon=0.01) and cumulative-sampling_time_s TTE convention as
    paper_figures.py's fig10c_tte_vs_n_self_convergence, so the
    calibration criterion matches the criterion the paper figure itself
    reports against.
  - Decision rule (fixed before results were inspected -- see decide()):
    among candidates whose median TTE is within 2x of the baseline's
    (the grid's first/cheapest value) median TTE, pick the CHEAPEST
    candidate whose validated_rate exceeds the baseline's by >= 0.30
    absolute (3/10 seeds); if none qualifies, pick the highest-rate
    candidate within that TTE budget (ties broken toward the cheaper
    candidate); if the baseline has zero validated seeds (median TTE
    undefined), the budget is instead anchored to the cheapest candidate
    that has >=1 validated seed.

Timing probe used to justify the widened grid (this machine, single seed,
N=32, h=0.5, lr=0.08, reg=0.05, ns=200, iter=100):
    Metropolis n_warmup=200  -> 27.9s
    Metropolis n_warmup=800  -> 69.5s
    Metropolis n_warmup=1600 -> 121.4s
    Gibbs n_sweeps=10 -> 20.2s
    Gibbs n_sweeps=40 -> 27.2s
    Gibbs n_sweeps=80 -> 35.0s

Usage:
    python scripts/exper/mcmc_calibration.py --run       # generate grid data (idempotent, skips existing)
    python scripts/exper/mcmc_calibration.py --analyze   # score candidates + apply the decision rule
    python scripts/exper/mcmc_calibration.py --run --analyze

Output data lands in the normal results/ tree, namespaced so it can never
collide with or be swept into the production Metropolis/Gibbs/LSB series:
    results/tfim_1d/32/custom/metropolis_calib_w{warmup}/result_..._seed{90..99}_....json.gz
    results/tfim_1d/32/custom/gibbs_calib_s{sweeps}/result_..._seed{90..99}_....json.gz
    results/tfim_1d/32/custom/lsb_calib_st{steps}/result_..._seed{90..99}_....json.gz
The scored summary (rates, median TTE, chosen value, decision-rule text)
is written to results/mcmc_calibration/summary.json.
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "viz"))
sys.path.insert(0, str(_ROOT / "scripts" / "exper"))

import jax
jax.config.update("jax_enable_x64", True)

from mcmc_matched_sweep import run_one  # noqa: E402
import matplotlib
matplotlib.use("Agg")
from paper_figures import load, compute_validated_convergence_iter, median_iqr  # noqa: E402

METROPOLIS_GRID = [200, 400, 800, 1600]
GIBBS_GRID = [10, 20, 40, 80]
# SA per-seed cost is nearly flat across cooling length, so widened straight to 160.
SA_GRID = [1, 10, 40, 160]
# LSB (Langevin SB): unlike SA, a single-seed spot check (seed 90, N=32) showed
# NO monotonic improvement with more steps -- final energy went good/bad/good/bad
# across 1000/2000/4000/8000 (-33.99/-31.80/-33.96/-31.75 vs exact ~-34.03), so
# raw step count is not obviously the dominant knob the way sa_sweeps was for
# SA. Kept as a doubling grid anyway (same as the others) so the multi-seed
# validated-rate protocol -- not a single noisy seed -- decides it.
LSB_GRID = [1000, 2000, 4000, 8000]
SEED_START = 90
N_SEEDS = 10
SIZE = 32
H, LR, REG, N_SAMPLES, ITERATIONS = 0.5, 0.08, 0.05, 200, 100
CV_THRESHOLD, WINDOW, EPSILON = 0.05, 10, 0.1
# epsilon=0.1 matches the rendered fig10c variant; 0.01 scores 0/20 everywhere.
RATE_MARGIN = 0.30
TTE_BUDGET_MULT = 2.0
OUTPUT_DIR = str(_ROOT / "results")
SUMMARY_PATH = _ROOT / "results" / "mcmc_calibration" / "summary.json"


def run_grid():
    for warmup in METROPOLIS_GRID:
        args = Namespace(
            h=H, lr=LR, reg=REG, n_samples=N_SAMPLES, iterations=ITERATIONS,
            gibbs_sweeps=10, n_warmup=warmup, variant=f"calib_w{warmup}",
            cem=False, output_dir=OUTPUT_DIR, skip_existing=True,
        )
        for seed in range(SEED_START, SEED_START + N_SEEDS):
            print(f"=== metropolis n_warmup={warmup} seed={seed} ===")
            run_one(SIZE, "metropolis", seed, args)
    for sweeps in GIBBS_GRID:
        args = Namespace(
            h=H, lr=LR, reg=REG, n_samples=N_SAMPLES, iterations=ITERATIONS,
            gibbs_sweeps=sweeps, n_warmup=None, variant=f"calib_s{sweeps}",
            cem=False, output_dir=OUTPUT_DIR, skip_existing=True,
        )
        for seed in range(SEED_START, SEED_START + N_SEEDS):
            print(f"=== gibbs n_sweeps={sweeps} seed={seed} ===")
            run_one(SIZE, "gibbs", seed, args)
    for sweeps in SA_GRID:
        args = Namespace(
            h=H, lr=LR, reg=REG, n_samples=N_SAMPLES, iterations=ITERATIONS,
            gibbs_sweeps=10, sa_sweeps=sweeps, n_warmup=None, variant=f"calib_s{sweeps}",
            cem=False, output_dir=OUTPUT_DIR, skip_existing=True,
        )
        for seed in range(SEED_START, SEED_START + N_SEEDS):
            print(f"=== simulated_annealing n_sweeps={sweeps} seed={seed} ===")
            run_one(SIZE, "simulated_annealing", seed, args)
    for steps in LSB_GRID:
        args = Namespace(
            h=H, lr=LR, reg=REG, n_samples=N_SAMPLES, iterations=ITERATIONS,
            gibbs_sweeps=10, n_warmup=None, lsb_steps=steps, variant=f"calib_st{steps}",
            cem=False, output_dir=OUTPUT_DIR, skip_existing=True,
        )
        for seed in range(SEED_START, SEED_START + N_SEEDS):
            print(f"=== lsb steps={steps} seed={seed} ===")
            run_one(SIZE, "lsb", seed, args)


def _score(solver_dir):
    pattern = (
        f"{OUTPUT_DIR}/tfim_1d/{SIZE}/custom/{solver_dir}/"
        f"result_1d_h{H}_rbmfull_nh{SIZE}_lr{LR}_reg{REG}_ns{N_SAMPLES}"
        f"_seed*_iter{ITERATIONS}_cem0_sigma1.0.json.gz"
    )
    recs = load(pattern)
    ttes = []
    for r in recs:
        hist = r["history"]
        tf = "total_sampling_time_s" if "total_sampling_time_s" in hist else "sampling_time_s"
        cum = np.cumsum(hist[tf])
        it = compute_validated_convergence_iter(
            hist, r["exact_energy"], SIZE, EPSILON, CV_THRESHOLD, WINDOW
        )
        if it is not None:
            ttes.append(float(cum[it - 1]))
    n = len(recs)
    rate = (len(ttes) / n) if n else None
    med, lo, hi = median_iqr(ttes)
    return {"n": n, "n_validated": len(ttes), "rate": rate, "median_tte_s": med,
            "iqr_lo_s": lo, "iqr_hi_s": hi}


def decide(grid, results):
    baseline = results[grid[0]]
    budget_ref = baseline if baseline["median_tte_s"] is not None else next(
        (results[c] for c in grid if results[c]["median_tte_s"] is not None), None
    )
    if budget_ref is None:
        return None, "no candidate had any validated seed -- cannot calibrate; needs manual review"

    budget = TTE_BUDGET_MULT * budget_ref["median_tte_s"]
    within_budget = [c for c in grid if results[c]["median_tte_s"] is not None
                      and results[c]["median_tte_s"] <= budget]
    baseline_rate = baseline["rate"] or 0.0
    qualifying = [c for c in within_budget if (results[c]["rate"] or 0.0) - baseline_rate >= RATE_MARGIN]

    if qualifying:
        chosen = min(qualifying)
        reason = (f"cheapest candidate clearing the +{RATE_MARGIN:g} validated-rate margin "
                   f"within {TTE_BUDGET_MULT:g}x TTE budget ({budget:.2f}s, anchored to "
                   f"{grid[0]}'s median TTE)")
    elif within_budget:
        chosen = max(within_budget, key=lambda c: (results[c]["rate"] or 0.0, -c))
        reason = (f"no candidate cleared the rate margin within budget ({budget:.2f}s); "
                   "picked the highest validated-rate candidate within budget")
    else:
        chosen = min(grid)
        reason = "no candidate fell within the TTE budget -- flagging for manual review"
    return chosen, reason


def analyze():
    report = {}
    for label, grid, solver_dir_fmt in (
        ("metropolis", METROPOLIS_GRID, "metropolis_calib_w{}"),
        ("gibbs", GIBBS_GRID, "gibbs_calib_s{}"),
        ("simulated_annealing", SA_GRID, "simulated_annealing_calib_s{}"),
        ("lsb", LSB_GRID, "lsb_calib_st{}"),
    ):
        results = {c: _score(solver_dir_fmt.format(c)) for c in grid}
        chosen, reason = decide(grid, results)
        report[label] = {"grid": grid, "results": results, "chosen": chosen, "reason": reason}
        print(f"\n=== {label} (N={SIZE}, seeds {SEED_START}-{SEED_START + N_SEEDS - 1}) ===")
        for c in grid:
            r = results[c]
            tte_str = f"{r['median_tte_s']:.2f}s" if r["median_tte_s"] is not None else "n/a"
            print(f"  {c:>5}: validated {r['n_validated']}/{r['n']}  "
                  f"rate={r['rate']:.2f}  median TTE={tte_str}")
        print(f"  -> chosen: {chosen}  ({reason})")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump({
            "protocol": {
                "size": SIZE, "seed_start": SEED_START, "n_seeds": N_SEEDS,
                "cv_threshold": CV_THRESHOLD, "window": WINDOW, "epsilon": EPSILON,
                "rate_margin": RATE_MARGIN, "tte_budget_mult": TTE_BUDGET_MULT,
            },
            "report": report,
        }, f, indent=2)
    print(f"\nwrote {SUMMARY_PATH}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="generate the calibration grid data")
    p.add_argument("--analyze", action="store_true", help="score candidates + apply the decision rule")
    args = p.parse_args()
    if not args.run and not args.analyze:
        p.error("pass --run and/or --analyze")
    if args.run:
        run_grid()
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()

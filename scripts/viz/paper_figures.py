#!/usr/bin/env python3
"""
paper_figures.py — a small set of paper-grade ITE / TTE / energy-to-solution
figures built only from result cells with enough independent seeds to support
a real statistical claim (no best-of, no cherry-picking, censoring reported
where runs don't reach threshold within their iteration budget).

Each figure documents, in its own header comment, exactly which result files
back it and why that cell was chosen (matched hyperparameters across the
compared arms, consistent physics instance, seed depth).

Output: plots/paper_figures/*.{png,pdf}
"""

import glob
import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_style import setup_style

OUT_DIR = "plots/paper_figures"

COLOR_BLUE = "#2a78d6"
COLOR_GREEN = "#008300"
COLOR_MAGENTA = "#e87ba4"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


# ---------------------------------------------------------------------------
# Shared statistics helpers
# ---------------------------------------------------------------------------

def compute_ite(energies, exact_energy, size, epsilon, window=10):
    """First iteration (1-indexed) where the causal rolling-mean energy error
    PER SPIN (|E-E_exact|/N, not relative to |E_exact|) drops below epsilon.
    None if never reached within the recorded run.

    Per-spin error is the convention used elsewhere in this repo (dashboard.py,
    plot_hparam_search.py, plot_sparsity_ablation_floor.py) and is the fairer
    one when a comparison spans different h (energy density varies with h, so
    relative error implicitly rescales by a different reference at every h;
    per-spin error does not)."""
    for t in range(len(energies)):
        w_start = max(0, t - window + 1)
        mean_e = sum(energies[w_start:t + 1]) / (t - w_start + 1)
        if abs(mean_e - exact_energy) / size < epsilon:
            return t + 1
    return None


def compute_convergence_iter(history, cv_threshold, window=10):
    """Self-referential, oracle-free convergence check -- no exact_energy
    needed. First iteration after which the per-iteration coefficient of
    variation CV = std(E_loc)/|mean(E_loc)| stays below cv_threshold for
    `window` consecutive iterations. None if never reached.

    This is a retrospective version of Trainer's live stop_at_convergence
    (src/encoder.py) -- same zero-variance-principle idea (Var(E_loc) -> 0
    at an exact eigenstate) -- but Trainer thresholds the RAW Var(E_loc)
    with no N-normalization, and a naive per-spin normalization (Var/N^2)
    turns out to be scale-broken: Var(E_loc) itself scales ~N even at the
    untrained state (verified empirically: Var/N ~ 1.0 at iteration 1
    across N=8..128), so a fixed Var/N^2 threshold is trivially satisfied
    at large N regardless of training progress. CV cancels that scaling
    (both std and |E| grow ~sqrt(N)/~N respectively) and is empirically
    stable across N at convergence (~0.01-0.02 band, checked N=8..128),
    which is why it's used here instead."""
    energies = history["energy"]
    stds = history["error"]
    consecutive = 0
    for t, (e, std) in enumerate(zip(energies, stds)):
        cv = std / abs(e) if e != 0 else float("inf")
        if cv < cv_threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= window:
            return t - window + 2
    return None


def compute_cv_self_convergence_iter(history, exact_energy, size, epsilon, cv_threshold, window=10):
    """CV-based self-detected + validated convergence point: the run must (1)
    self-detect convergence (see compute_convergence_iter -- oracle-free, this
    is what a real deployment without a known answer would use to decide
    "we're done") AND (2) the value it stopped at must actually be within
    epsilon energy-error-per-spin of the true answer (same convention as
    compute_ite). A run that self-detects convergence at the WRONG plateau
    (verified to happen often for uncalibrated D-Wave sampling, see
    conversation) does NOT count -- it's censored, same as a run that never
    stabilizes at all.

    This is the internal-tuning criterion used by mcmc_calibration.py to pick
    solver mixing/integration parameters. It is NOT the report's published TTE
    criterion -- see compute_validated_convergence_iter below for that (report
    sec:exper:tte, Figure 15 / fig10c-fig10d).

    Returns the self-detected convergence iteration if validated, else None.
    """
    conv_iter = compute_convergence_iter(history, cv_threshold, window)
    if conv_iter is None:
        return None
    energies = history["energy"]
    # plateau is energies[conv_iter-1 : conv_iter-1+window] (0-indexed)
    plateau = energies[conv_iter - 1: conv_iter - 1 + window]
    mean_e = sum(plateau) / len(plateau)
    if abs(mean_e - exact_energy) / size < epsilon:
        return conv_iter
    return None


def compute_validated_convergence_iter(energies, exact_energy, size, epsilon, window=10):
    """The report's published TTE criterion (sec:exper:tte, Figure 15): the
    causal rolling-window (size `window`) mean energy error per spin,
    |mean(E)-E_exact|/size, must first drop below epsilon AND then stay below
    epsilon for `window` consecutive iterations -- the crossing iteration is
    the first element of that sustained window. Oracle-based (uses
    exact_energy directly, unlike the CV self-detection criterion above).

    Requiring `window` consecutive passes rather than a single crossing
    discards runs that briefly and spuriously dip below epsilon before moving
    back away from it (observed in some D-Wave trajectories).

    Returns the 1-indexed iteration of that first sustained-below-epsilon
    window, or None if the run never satisfies the criterion within its
    recorded iteration budget (censored).
    """
    n = len(energies)
    errors = []
    for t in range(n):
        w_start = max(0, t - window + 1)
        mean_e = sum(energies[w_start:t + 1]) / (t - w_start + 1)
        errors.append(abs(mean_e - exact_energy) / size)
    for t in range(n):
        if errors[t] < epsilon:
            end = t + window
            if end <= n and all(e < epsilon for e in errors[t:end]):
                return t + 1
    return None


def tte99_from_validated(ttes, total, p_target=0.99):
    """Convert a validated-convergence sample (per-seed TTEs from
    compute_validated_convergence_iter, evaluated at the fixed iteration
    budget) into the literature-standard probabilistic TTE (arXiv:2401.07184,
    generalizing TTS/TTS99 from exact-ground-state success to
    within-epsilon success): TTE99 = T_r * ln(1-p_target)/ln(1-p), where
    p = len(ttes)/total is the validated (success) fraction and T_r is a
    representative single-run time. This is the expected total wall-clock
    time to reach epsilon with p_target confidence, if a run that fails to
    validate within its budget is discarded and restarted from scratch with
    a new seed.

    Applied identically to the median and IQR bounds (R depends only on p,
    a single scalar per cell, not on the individual seed's own TTE).

    Returns (None, None, None, p) if no seed validated (p=0, undefined --
    infinitely many restarts would be needed); returns (T_r, T_r, T_r, 1.0)
    unchanged if every seed validated (p=1, R=1, no repeats needed).
    """
    p = len(ttes) / total if total else 0.0
    if not ttes:
        return None, None, None, p
    m, lo, hi = median_iqr(ttes)
    if p >= 1.0:
        return m, lo, hi, p
    R = np.log(1 - p_target) / np.log(1 - p)
    return m * R, lo * R, hi * R, p


def fit_powerlaw_exponent(xs, ys):
    """Least-squares exponent of a power-law fit y = c * x**p in log-log space.

    None if fewer than 2 usable points.
    """
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def median_iqr(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    return (
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 75)),
    )


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with gzip.open(f) as fh:
            out.append(json.load(fh))
    return out


def fpga_glob(n):
    """FPGA (sweeps100 campaign) result glob for TFIM 1D at size n.

    Lives under results/tfim_1d/{n}/fpga/fpga/ alongside every other TFIM 1D
    solver -- migrated out of results/sweeps100/tfim_1d/{n}/fpga/ so this is
    the only copy. The sweeps2000 FPGA campaign (different num_sweeps, same
    filenames -- see results/custom/velox_default_h0.5.json's description)
    is a genuinely separate dataset and still lives under
    results/sweeps2000/tfim_1d/{n}/fpga/; it is NOT merged here.
    """
    return f"results/tfim_1d/{n}/fpga/*/result_*_seed*_iter*"


def style_axes(ax):
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color=GRID, linewidth=0.4, zorder=0, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def log_x_with_ticks(ax, sizes, rotation=0):
    """Log-x axis labeled only at the actual N values (no autogenerated
    3x10^1-style minor tick labels colliding with them)."""
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes], rotation=rotation)


# ---------------------------------------------------------------------------
# Figure 1 — ITE vs N, tfim_1d, TFIM critical point
#
# Metropolis/Gibbs are a true sampler-only match (cem=False for both).
# Epsilon=0.01 energy error per spin, rolling window=10. Runs that never
# cross epsilon within the 300-iteration budget are censored (hollow marker
# at the iteration budget), not dropped.
# ---------------------------------------------------------------------------

def fig1_ite_vs_n_tfim1d():
    sizes = [25, 36, 49, 64, 81, 100, 121, 144, 169, 196]
    samplers = [
        ("metropolis", "Metropolis (cem=False)", COLOR_BLUE, "o", "-"),
        ("gibbs", "Gibbs (cem=False)", COLOR_GREEN, "s", "-"),
    ]
    epsilon = 0.01
    n_iterations = 300

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method, label, color, marker, linestyle in samplers:
        med, lo, hi, ns, censored_x, censored_n = [], [], [], [], [], []
        for n in sizes:
            recs = load(f"results/tfim_1d/{n}/custom/{method}/result_1d_h1.0_rbmfull_nh{n}_lr0.01_reg1e-05_ns1000_seed*_iter*")
            if not recs:
                med.append(None); lo.append(None); hi.append(None); ns.append(0)
                continue
            ites = [compute_ite(r["history"]["energy"], r["exact_energy"], n, epsilon) for r in recs]
            reached = [v for v in ites if v is not None]
            ns.append(len(ites))
            if reached:
                m, l, h = median_iqr(reached)
                med.append(m); lo.append(l); hi.append(h)
            else:
                med.append(None); lo.append(None); hi.append(None)
            if len(reached) < len(ites):
                censored_x.append(n)
                censored_n.append(f"{len(reached)}/{len(ites)}")

        xs = [n for n, m in zip(sizes, med) if m is not None]
        ys = [m for m in med if m is not None]
        lo_v = [l for l in lo if l is not None]
        hi_v = [h for h in hi if h is not None]
        if xs:
            yerr = [[y - l for y, l in zip(ys, lo_v)], [h - y for y, h in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                        markersize=7, linewidth=1.6, capsize=3, zorder=3, linestyle=linestyle)

        # censored sizes: draw hollow marker at the iteration budget
        cx = [n for n in sizes if n not in xs]
        if cx:
            ax.scatter(cx, [n_iterations] * len(cx), marker=marker, facecolors="none",
                       edgecolors=color, s=55, linewidth=1.4, zorder=3)

    ax.axhline(n_iterations, color=MUTED, linestyle=":", linewidth=1)
    ax.text(sizes[-1], n_iterations * 1.03, "iteration budget (censored above)",
            fontsize=8, color=MUTED, ha="right")

    ax.set_yscale("log")
    log_x_with_ticks(ax, sizes, rotation=45)
    ax.set_xlabel("System size N")
    ax.set_ylabel(f"ITE — iterations to {epsilon:.3g} energy error/spin (median, IQR)")
    ax.set_title("Iterations-to-epsilon vs. system size\nTFIM 1D, critical point h=1.0, matched hyperparameters, no best-of")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig1_ite_vs_n_tfim1d")


# ---------------------------------------------------------------------------
# Figure 2 — Wall-clock time-to-epsilon vs N, VeloxQ vs FPGA
#
# sweeps100/sweeps2000 pooled (verified independent, not duplicated runs).
# The re-tuned N=128 VeloxQ point (sweeps100_v2/sweeps2000_v2) is shown
# separately, not connected to the matched velox/fpga line, since FPGA has
# no equivalently re-tuned config at N=128.
# Epsilon=0.01 energy error per spin (see compute_ite docstring).
# ---------------------------------------------------------------------------

def fig2_tte_vs_n_velox_fpga():
    sizes = [8, 12, 16, 24, 32, 64, 128]
    epsilon = 0.01

    fig, ax = plt.subplots(figsize=(8, 5.5))

    solvers = [("velox", "VeloxQ (SA)", COLOR_BLUE, "o"), ("fpga", "FPGA", COLOR_GREEN, "s")]

    for solver, label, color, marker in solvers:
        med, lo, hi, censored_x = [], [], [], []
        # FPGA's sweeps100 campaign now lives under results/tfim_1d/{n}/fpga/
        # (see fpga_glob docstring); sweeps2000 is a separate campaign, still
        # under results/sweeps2000/. velox is unaffected -- both campaigns
        # still live under results/sweeps{100,2000}/.
        sweeps100_pattern = (lambda n: fpga_glob(n)) if solver == "fpga" \
            else (lambda n: f"results/sweeps100/tfim_1d/{n}/{solver}/*/result_*_seed*_iter*")
        for n in sizes:
            recs = [
                r for r in
                load(sweeps100_pattern(n))
                + load(f"results/sweeps2000/tfim_1d/{n}/{solver}/*/result_*_seed*_iter*")
                if r["config"]["n_hidden"] == n
                and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9
                and r["config"]["n_samples"] == 200
            ]
            if not recs:
                med.append(None); lo.append(None); hi.append(None)
                continue
            times = []
            for r in recs:
                ite = compute_ite(r["history"]["energy"], r["exact_energy"], n, epsilon)
                if ite is not None:
                    cum_t = np.cumsum(r["history"]["sampling_time_s"])
                    times.append(float(cum_t[ite - 1]))
            n_total = len(recs)
            if times:
                m, l, h = median_iqr(times)
                med.append(m); lo.append(l); hi.append(h)
            else:
                med.append(None); lo.append(None); hi.append(None)
            if len(times) < n_total:
                censored_x.append(n)

        xs = [n for n, m in zip(sizes, med) if m is not None]
        ys = [m for m in med if m is not None]
        lo_v = [l for l in lo if l is not None]
        hi_v = [h for h in hi if h is not None]
        if xs:
            yerr = [[y - l for y, l in zip(ys, lo_v)], [h - y for y, h in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                        markersize=7, linewidth=1.6, capsize=3, zorder=3)
        cx = [n for n in sizes if n not in xs]
        if cx:
            # place censored markers just above the highest plotted value for visibility
            ymax = max(ys) if ys else 1.0
            ax.scatter(cx, [ymax * 3] * len(cx), marker=marker, facecolors="none",
                       edgecolors=color, s=55, linewidth=1.4, zorder=3)

    # re-tuned N=128 velox point (sweeps100_v2 + sweeps2000_v2)
    recs = load("results/sweeps100_v2/tfim_1d/128/velox/simulated_annealing/result_*seed*") + \
        load("results/sweeps2000_v2/tfim_1d/128/velox/simulated_annealing/result_*seed*")
    times = []
    for r in recs:
        ite = compute_ite(r["history"]["energy"], r["exact_energy"], 128, epsilon)
        if ite is not None:
            cum_t = np.cumsum(r["history"]["sampling_time_s"])
            times.append(float(cum_t[ite - 1]))
    if times:
        m, lo_r, hi_r = median_iqr(times)
        ax.errorbar([128], [m], yerr=[[m - lo_r], [hi_r - m]], marker="*", color="#eb6834",
                    markersize=16, linewidth=1.6, capsize=3, zorder=4,
                    label=f"VeloxQ, re-tuned (n={len(times)}/{len(recs)} reached)")
        ax.annotate("re-tuned\nhyperparameters\n(not matched to FPGA)",
                    xy=(128, m), xytext=(40, m * 0.35),
                    fontsize=8, color=MUTED, ha="center",
                    arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8))

    ax.set_yscale("log")
    log_x_with_ticks(ax, sizes)
    ax.set_xlabel("System size N")
    ax.set_ylabel(f"TTE — wall-clock time to {epsilon:.3g} energy error/spin [s] (median, IQR)")
    ax.set_title("Time-to-epsilon vs. system size\nTFIM 1D, h=0.5, matched hyperparameters, VeloxQ vs. FPGA")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig2_tte_vs_n_velox_fpga")


# ---------------------------------------------------------------------------
# Figure 3 — Energy-to-solution (final energy error per spin) vs N, two panels
#
# Panel A: same tfim_1d custom-sampler cells as Figure 1 (h=1.0, 300 iters).
# Panel B: same velox/fpga matched cells as Figure 2 (h=0.5, 100 iters),
#          plus the re-tuned N=128 VeloxQ point.
# ---------------------------------------------------------------------------

def fig3_energy_to_solution_vs_n():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A
    ax = axes[0]
    sizes = [25, 36, 49, 64, 81, 100, 121, 144, 169, 196]
    samplers = [
        ("metropolis", "Metropolis (cem=False)", COLOR_BLUE, "o", "-"),
        ("gibbs", "Gibbs (cem=False)", COLOR_GREEN, "s", "-"),
    ]
    for method, label, color, marker, linestyle in samplers:
        med, lo, hi = [], [], []
        for n in sizes:
            recs = load(f"results/tfim_1d/{n}/custom/{method}/result_1d_h1.0_rbmfull_nh{n}_lr0.01_reg1e-05_ns1000_seed*_iter*")
            errs = [abs(r["final_energy"] - r["exact_energy"]) / n for r in recs]
            m, l, h = median_iqr(errs)
            med.append(m); lo.append(l); hi.append(h)
        xs = [n for n, m in zip(sizes, med) if m is not None]
        ys = [m for m in med if m is not None]
        yerr = [[y - l for y, l in zip(ys, [v for v in lo if v is not None])],
                [h - y for y, h in zip(ys, [v for v in hi if v is not None])]]
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                    markersize=7, linewidth=1.6, capsize=3, zorder=3, linestyle=linestyle)
    ax.set_yscale("log")
    log_x_with_ticks(ax, sizes, rotation=45)
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"Final energy error per spin  $|E_\mathrm{final}-E_\mathrm{exact}|/N$")
    ax.set_title("(A) TFIM critical point (h=1.0), 300 iterations")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # Panel B
    ax = axes[1]
    sizes = [8, 12, 16, 24, 32, 64, 128]
    solvers = [("velox", "VeloxQ (SA)", COLOR_BLUE, "o"), ("fpga", "FPGA", COLOR_GREEN, "s")]
    for solver, label, color, marker in solvers:
        med, lo, hi = [], [], []
        # FPGA's sweeps100 campaign now lives under results/tfim_1d/{n}/fpga/
        # (see fpga_glob docstring); sweeps2000 is a separate campaign, still
        # under results/sweeps2000/. velox is unaffected -- both campaigns
        # still live under results/sweeps{100,2000}/.
        sweeps100_pattern = (lambda n: fpga_glob(n)) if solver == "fpga" \
            else (lambda n: f"results/sweeps100/tfim_1d/{n}/{solver}/*/result_*_seed*_iter*")
        for n in sizes:
            recs = [
                r for r in
                load(sweeps100_pattern(n))
                + load(f"results/sweeps2000/tfim_1d/{n}/{solver}/*/result_*_seed*_iter*")
                if r["config"]["n_hidden"] == n
                and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9
                and r["config"]["n_samples"] == 200
            ]
            errs = [abs(r["final_energy"] - r["exact_energy"]) / n for r in recs]
            m, l, h = median_iqr(errs)
            med.append(m); lo.append(l); hi.append(h)
        xs = [n for n, m in zip(sizes, med) if m is not None]
        ys = [m for m in med if m is not None]
        yerr = [[y - l for y, l in zip(ys, [v for v in lo if v is not None])],
                [h - y for y, h in zip(ys, [v for v in hi if v is not None])]]
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                    markersize=7, linewidth=1.6, capsize=3, zorder=3)

    recs = load("results/sweeps100_v2/tfim_1d/128/velox/simulated_annealing/result_*seed*") + \
        load("results/sweeps2000_v2/tfim_1d/128/velox/simulated_annealing/result_*seed*")
    errs = [abs(r["final_energy"] - r["exact_energy"]) / 128 for r in recs]
    m, l, h = median_iqr(errs)
    ax.errorbar([128], [m], yerr=[[m - l], [h - m]], marker="*", color="#eb6834",
                markersize=16, linewidth=1.6, capsize=3, zorder=4,
                label="VeloxQ, re-tuned")

    ax.set_yscale("log")
    log_x_with_ticks(ax, sizes)
    ax.set_xlabel("System size N")
    ax.set_title("(B) TFIM, h=0.5, 100 iterations")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("Energy-to-solution vs. system size (median, IQR; no best-of)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_energy_to_solution_vs_n")


# ---------------------------------------------------------------------------
# Figure 4 — Ground-state success fraction vs N at the Majumdar-Ghosh point
#
# J1-J2 chain at J2=0.5, 28-30 seeds per cell, 300 SR iterations. "Success"
# = final energy error per spin < 0.05. Wilson score CI. Each solver shows
# a different finite-size crossover (Gibbs solves it at every N tested,
# Exchange only from N>=12, SA jumps from 0% at N=16).
# ---------------------------------------------------------------------------

def fig4_success_fraction_heisenberg():
    sizes = [8, 12, 16]
    solvers = [
        ("exchange", "Exchange", COLOR_BLUE, "o"),
        ("gibbs", "Gibbs", COLOR_GREEN, "s"),
        ("simulated_annealing", "Simulated annealing", COLOR_MAGENTA, "^"),
    ]
    success_threshold = 0.05

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for method, label, color, marker in solvers:
        fracs, los, his, ns = [], [], [], []
        for n in sizes:
            recs = load(f"results/heisenberg_j1j2_1d/{n}/custom/{method}/result_*_J20.5_*_seed*_iter*")
            errs = [abs(r["final_energy"] - r["exact_energy"]) / n for r in recs]
            k = sum(1 for e in errs if e < success_threshold)
            n_total = len(errs)
            ns.append(n_total)
            if n_total == 0:
                fracs.append(None); los.append(None); his.append(None)
                continue
            lo_ci, hi_ci = wilson_ci(k, n_total)
            fracs.append(k / n_total); los.append(lo_ci); his.append(hi_ci)

        xs = [n for n, f in zip(sizes, fracs) if f is not None]
        ys = [f for f in fracs if f is not None]
        yerr = [[y - l for y, l in zip(ys, [v for v in los if v is not None])],
                [h - y for y, h in zip(ys, [v for v in his if v is not None])]]
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                    markersize=8, linewidth=1.8, capsize=4, zorder=3)
        for x, y, n_total in zip(xs, ys, ns):
            ax.annotate(f"n={n_total}", (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, color=MUTED, ha="center")

    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("System size N")
    ax.set_ylabel(f"Ground-state success fraction (energy error/spin < {success_threshold:.2g}), Wilson 95% CI")
    ax.set_title(
        "Solver-dependent finite-size crossover at the Majumdar-Ghosh point\n"
        "J1-J2 Heisenberg chain, J2/J1 = 0.5, 300 SR iterations"
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="center right", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig4_success_fraction_heisenberg_mg_point")


# ---------------------------------------------------------------------------
# Figure 5 — Single-instance convergence, TFIM 1D, includes real D-Wave QPU
# sampling (pegasus, zephyr), no new QPU time used (archived runs only).
#
# N=8 and N=16, h=0.5, rbmfull, lr=0.1, reg=0.001, ns=1000, seed=42, iter=300.
# Exact filenames (not globs) pin the specific archived run matching the
# D-Wave runs' naming, since some nominally-identical cells hold more than
# one run under the same seed label. VeloxQ has no archived N=8 run at this
# config. Single seed per solver -- qualitative comparison, not a
# statistical claim. D-Wave QPU runs have no sampling_time_s, so this is
# energy-vs-iteration only, not a TTE plot.
# ---------------------------------------------------------------------------

def fig5_convergence_dwave_tfim1d():
    panels = [
        (8, [
            ("results/tfim_1d/8/custom/metropolis/result_1d_h0.5_rbmfull_nh8_lr0.1_reg0.001_ns1000_seed42_iter300_cem0.json.gz", "Metropolis", COLOR_BLUE, "-"),
            ("results/tfim_1d/8/dimod/pegasus/result_1d_h0.5_rbmfull_nh8_lr0.1_reg0.001_ns1000_seed42_iter300_cem0.json.gz", "D-Wave Pegasus (QPU)", COLOR_MAGENTA, "-"),
            ("results/tfim_1d/8/dimod/zephyr/result_1d_h0.5_rbmfull_nh8_lr0.1_reg0.001_ns1000_seed42_iter300_cem0.json.gz", "D-Wave Zephyr (QPU)", "#eb6834", "-"),
        ]),
        (16, [
            ("results/tfim_1d/16/custom/metropolis/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed42_iter300.json.gz", "Metropolis", COLOR_BLUE, "-"),
            ("results/tfim_1d/16/velox/velox/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed42_iter300.json.gz", "VeloxQ (SA)", COLOR_GREEN, "-"),
            ("results/tfim_1d/16/dimod/pegasus/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed42_iter300.json.gz", "D-Wave Pegasus (QPU)", COLOR_MAGENTA, "-"),
            ("results/tfim_1d/16/dimod/zephyr/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed42_iter300.json.gz", "D-Wave Zephyr (QPU)", "#eb6834", "-"),
        ]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (n, runs) in zip(axes, panels):
        for path, label, color, linestyle in runs:
            with gzip.open(path) as fh:
                r = json.load(fh)
            energies = np.array(r["history"]["energy"])
            exact = r["exact_energy"]
            err_per_spin = np.abs(energies - exact) / n
            n_budget = r["config"]["iterations"]
            plot_label = label if len(err_per_spin) == n_budget else f"{label} (stopped @ {len(err_per_spin)}/{n_budget})"
            ax.plot(np.arange(1, len(err_per_spin) + 1), err_per_spin, color=color, label=plot_label,
                    linewidth=1.6, linestyle=linestyle, zorder=3)
            if len(err_per_spin) < n_budget:
                ax.scatter([len(err_per_spin)], [err_per_spin[-1]], color=color, marker="x", s=40, zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel("SR iteration")
        ax.set_title(f"N = {n}")
        style_axes(ax)
        ax.legend(frameon=False, loc="upper right", fontsize=9)

    axes[0].set_ylabel(r"Energy error per spin  $|E-E_\mathrm{exact}|/N$")
    fig.suptitle(
        "Single-instance convergence, TFIM 1D, h=0.5, matched hyperparameters\n"
        "(seed=42, single run per solver — not a statistical comparison)", y=1.03
    )
    fig.tight_layout()
    _save(fig, "fig5_convergence_dwave_tfim1d")


# ---------------------------------------------------------------------------
# Figure 6 — Energy-to-solution vs N, TFIM 2D, includes real D-Wave QPU
# sampling (pegasus, zephyr), no new QPU time used (archived runs only).
#
# rbmfull, lr=0.1, reg=0.001, ns=1000, iter=300, h=0.5/1.0/2.0, sizes 4/6/8.
# D-Wave has 2 QPU runs per cell (seed 1, 42); metropolis has 2-3. With only
# 2-3 samples per point the shown band is a min/max-ish spread, not a real
# IQR. Metric is energy error per spin, N_spins=n*n.
# ---------------------------------------------------------------------------

def fig6_energy_vs_n_tfim2d_dwave():
    sizes = [4, 6, 8]
    hs = [0.5, 1.0, 2.0]
    solvers = [
        ("custom/metropolis", "Metropolis", COLOR_BLUE, "o"),
        ("dimod/pegasus", "D-Wave Pegasus (QPU)", COLOR_MAGENTA, "^"),
        ("dimod/zephyr", "D-Wave Zephyr (QPU)", "#eb6834", "s"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, h in zip(axes, hs):
        for subdir, label, color, marker in solvers:
            med, lo, hi, ns = [], [], [], []
            for n in sizes:
                nh = n * n
                recs = [
                    r for r in load(f"results/tfim_2d/{n}/{subdir}/result_2d_h{h}_rbmfull_nh{nh}_lr0.1_reg0.001_ns1000_seed*_iter300*")
                    if r["config"]["n_hidden"] == nh
                ]
                errs = [abs(r["final_energy"] - r["exact_energy"]) / nh for r in recs]
                ns.append(len(errs))
                m, l, hh = median_iqr(errs)
                med.append(m); lo.append(l); hi.append(hh)
            xs = [n for n, m in zip(sizes, med) if m is not None]
            ys = [m for m in med if m is not None]
            lo_v = [v for v in lo if v is not None]
            hi_v = [v for v in hi if v is not None]
            if xs:
                yerr = [[y - l for y, l in zip(ys, lo_v)], [hh - y for y, hh in zip(ys, hi_v)]]
                ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                            markersize=7, linewidth=1.6, capsize=3, zorder=3)
            for x, n_total in zip(sizes, ns):
                if n_total:
                    ax.annotate(f"n={n_total}", (x, med[sizes.index(x)]), textcoords="offset points",
                                xytext=(0, 7), fontsize=6.5, color=color, ha="center")
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel("Linear size N (lattice N×N)")
        ax.set_title(f"h = {h}")
        style_axes(ax)

    axes[0].set_ylabel(r"Final energy error per spin  $|E_\mathrm{final}-E_\mathrm{exact}|/N_\mathrm{spins}$")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Energy-to-solution vs. system size, TFIM 2D, matched hyperparameters\n"
        "Metropolis vs. D-Wave QPU (Pegasus, Zephyr) — n=2-3 samples per point", y=1.04
    )
    fig.tight_layout()
    _save(fig, "fig6_energy_vs_n_tfim2d_dwave")


# ---------------------------------------------------------------------------
# Figure 7 — ITE and energy-to-solution vs N, classical solver comparison,
# with real statistical power (10-15 seeds per point, not 2-3).
#
# h=1.0, rbmfull, lr=0.1, reg=1e-05, ns=1000, iter=100, cem=0.
# dimod/simulated_annealing and dimod/tabu run through the same DimodSampler
# codepath as the D-Wave QPU runs, executed on CPU instead of QPU.
# Metric is energy error per spin (see compute_ite docstring).
#
# Panel A: final energy error per spin vs. N (4, 8, 16). An ITE panel was
# tried here first and dropped -- at this cell's short iteration budget,
# per-seed histories are unstable enough that ITE can spuriously flag an
# early iteration as converged.
#
# Panel B: same solvers/seeds, swept across h at fixed N=8. No solver
# dominates across the whole field range (Metropolis best at h=0.5, Tabu
# best at h=1.5-2.0).
# ---------------------------------------------------------------------------

def fig7_classical_scaling_tfim1d():
    solvers = [
        ("custom/metropolis", "Metropolis (MCMC)", COLOR_BLUE, "o"),
        ("dimod/simulated_annealing", "Simulated annealing (classical, dimod/neal)", COLOR_GREEN, "s"),
        ("dimod/tabu", "Tabu search (classical, dimod)", COLOR_MAGENTA, "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: final energy error per spin vs N, h=1.0
    ax = axes[0]
    sizes = [4, 8, 16]
    for solver, label, color, marker in solvers:
        med, lo, hi, ns = [], [], [], []
        for n in sizes:
            recs = load(f"results/tfim_1d/{n}/{solver}/result_1d_h1.0_rbmfull_nh{n}_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz")
            ns.append(len(recs))
            errs = [abs(r["final_energy"] - r["exact_energy"]) / n for r in recs]
            m, l, h = median_iqr(errs)
            med.append(m); lo.append(l); hi.append(h)
        xs = [n for n, m in zip(sizes, med) if m is not None]
        ys = [m for m in med if m is not None]
        yerr = [[y - l for y, l in zip(ys, [v for v in lo if v is not None])],
                [h - y for y, h in zip(ys, [v for v in hi if v is not None])]]
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                    markersize=7, linewidth=1.6, capsize=3, zorder=3)
        for x, n_total in zip(sizes, ns):
            ax.annotate(f"n={n_total}", (x, med[sizes.index(x)]), textcoords="offset points",
                        xytext=(8, 0), fontsize=7, color=color, ha="left")
    ax.set_yscale("log")
    log_x_with_ticks(ax, sizes)
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"Final energy error per spin  $|E_\mathrm{final}-E_\mathrm{exact}|/N$")
    ax.set_title("(A) vs. system size, h=1.0 (critical point)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    # Panel B: final energy error per spin vs h, N=8 fixed
    ax = axes[1]
    hs = [0.5, 1.0, 1.5, 2.0]
    for solver, label, color, marker in solvers:
        med, lo, hi, ns = [], [], [], []
        for h_val in hs:
            recs = load(f"results/tfim_1d/8/{solver}/result_1d_h{h_val}_rbmfull_nh8_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz")
            ns.append(len(recs))
            errs = [abs(r["final_energy"] - r["exact_energy"]) / 8 for r in recs]
            m, l, hh = median_iqr(errs)
            med.append(m); lo.append(l); hi.append(hh)
        xs = [x for x, m in zip(hs, med) if m is not None]
        ys = [m for m in med if m is not None]
        yerr = [[y - l for y, l in zip(ys, [v for v in lo if v is not None])],
                [hh - y for y, hh in zip(ys, [v for v in hi if v is not None])]]
        ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                    markersize=7, linewidth=1.6, capsize=3, zorder=3)
        for x, n_total in zip(hs, ns):
            ax.annotate(f"n={n_total}", (x, med[hs.index(x)]), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, color=color, ha="center")
    ax.set_yscale("log")
    ax.set_xticks(hs)
    ax.set_xlabel("Transverse field h")
    ax.set_ylabel(r"Final energy error per spin  $|E_\mathrm{final}-E_\mathrm{exact}|/N$")
    ax.set_title("(B) vs. field strength, N=8 fixed")
    style_axes(ax)
    ax.legend(frameon=False, loc="center left", fontsize=8)

    fig.suptitle(
        "Classical solver comparison, TFIM 1D, matched hyperparameters, 100 iterations\n"
        "MCMC vs. classical annealing/tabu (10-15 independent seeds per point, no best-of)", y=1.03
    )
    fig.tight_layout()
    _save(fig, "fig7_classical_scaling_tfim1d")


# ---------------------------------------------------------------------------
# Figure 8 — All solvers archived at N=16, TFIM 1D, each at its own
# best-available operating point (NOT hyperparameter-matched -- no config
# is shared by more than 3 of these solvers anywhere in the archive).
#
# Per-solver cell (all rbm=full, n_hidden=16 unless noted):
#   Metropolis / dimod-SA / dimod-Tabu : h=1.0, lr=0.1,  reg=1e-05, ns=1000, iter=100,  n=10 each
#   Gibbs                              : h=0.5, lr=0.01, reg=1e-05, ns=1000, iter=300, cem=False, n=6
#   VeloxQ (SA) / FPGA                 : h=0.5, lr=0.08, reg=0.05, ns=200, sweeps100+sweeps2000 pooled, n=40
#   D-Wave Pegasus / Zephyr (QPU)      : h=0.5, rbmfull, lr=0.1, reg=0.001, ns=1000, iter=300, n=2
#
# Metric: energy error per spin, |E_final-E_exact|/16.
# ---------------------------------------------------------------------------

def fig8_all_solvers_n16():
    rows = [
        ("Metropolis (MCMC)", "results/tfim_1d/16/custom/metropolis/result_1d_h1.0_rbmfull_nh16_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz", None, COLOR_BLUE),
        ("Gibbs (MCMC)", "results/tfim_1d/16/custom/gibbs/result_1d_h0.5_rbmfull_nh16_lr0.01_reg1e-05_ns1000_seed*_iter300_cem0_sigma1.0.json.gz", None, COLOR_BLUE),
        ("Simulated annealing (dimod/neal)", "results/tfim_1d/16/dimod/simulated_annealing/result_1d_h1.0_rbmfull_nh16_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz", None, COLOR_GREEN),
        ("Tabu search (dimod)", "results/tfim_1d/16/dimod/tabu/result_1d_h1.0_rbmfull_nh16_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz", None, COLOR_GREEN),
        ("VeloxQ (SA hardware)", None, "velox", "#eb6834"),
        ("FPGA", None, "fpga", "#eb6834"),
        ("D-Wave Pegasus (QPU)", "results/tfim_1d/16/dimod/pegasus/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed*_iter300.json.gz", None, COLOR_MAGENTA),
        ("D-Wave Zephyr (QPU)", "results/tfim_1d/16/dimod/zephyr/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed*_iter300.json.gz", None, COLOR_MAGENTA),
    ]

    def sweeps_recs(solver):
        # FPGA's sweeps100 campaign now lives under results/tfim_1d/16/fpga/
        # (see fpga_glob docstring); sweeps2000 is a separate campaign, still
        # under results/sweeps2000/. velox is unaffected -- both campaigns
        # still live under results/sweeps{100,2000}/.
        patterns = (
            [fpga_glob(16), "results/sweeps2000/tfim_1d/16/fpga/*/result_*_seed*_iter*"]
            if solver == "fpga" else
            [f"results/{campaign}/tfim_1d/16/{solver}/*/result_*_seed*_iter*" for campaign in ("sweeps100", "sweeps2000")]
        )
        out = []
        for pattern in patterns:
            for r in load(pattern):
                c = r["config"]
                if c["n_hidden"] == 16 and abs(c["learning_rate"] - 0.08) < 1e-9 \
                        and abs(c["regularization"] - 0.05) < 1e-9 and c["n_samples"] == 200:
                    out.append(r)
        return out

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ylabels = []
    for i, (label, pattern, sweep_solver, color) in enumerate(rows):
        recs = sweeps_recs(sweep_solver) if sweep_solver else load(pattern)
        errs = [abs(r["final_energy"] - r["exact_energy"]) / 16 for r in recs]
        m, lo, hi = median_iqr(errs)
        y = len(rows) - i
        if m is not None:
            ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], marker="o", color=color,
                        markersize=8, linewidth=1.6, capsize=4, zorder=3)
        ax.annotate(f"n={len(errs)}", (max(errs) if errs else 1, y), textcoords="offset points",
                    xytext=(8, 0), fontsize=7.5, color=color, va="center")
        ylabels.append(label)

    ax.set_yticks([len(rows) - i for i in range(len(rows))])
    ax.set_yticklabels(ylabels)
    ax.set_xscale("log")
    ax.set_xlabel(r"Final energy error per spin  $|E_\mathrm{final}-E_\mathrm{exact}|/N$ (median, IQR)")
    ax.set_title(
        "All archived solvers at N=16, each at its own best-available operating point\n"
        "NOT hyperparameter-matched -- see script comments for each row's (h, config, n)",
        fontsize=11
    )
    style_axes(ax)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.8, zorder=0)

    fig.tight_layout()
    _save(fig, "fig8_all_solvers_n16")


# ---------------------------------------------------------------------------
# Figure 9 — ITE and TTE, all solvers at N=16 (companion to Figure 8, same
# per-solver cells).
#
# Panel A (ITE): iterations to 0.01 energy error per spin, rolling window=10.
# Solvers that never reach threshold within their budget are censored
# (hollow marker at the recorded budget), not dropped.
#
# Panel B (TTE): wall-clock time to threshold. D-Wave Pegasus/Zephyr (QPU)
# have no per-iteration timing field and are omitted rather than assigned a
# fabricated duration. Uses 'total_sampling_time_s' where present (includes
# CEM time), else 'sampling_time_s'.
# ---------------------------------------------------------------------------

def fig9_ite_tte_all_solvers_n16():
    epsilon = 0.01

    rows = [
        ("Metropolis (MCMC)", "results/tfim_1d/16/custom/metropolis/result_1d_h1.0_rbmfull_nh16_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz", None, COLOR_BLUE),
        ("Gibbs (MCMC)", "results/tfim_1d/16/custom/gibbs/result_1d_h0.5_rbmfull_nh16_lr0.01_reg1e-05_ns1000_seed*_iter300_cem0_sigma1.0.json.gz", None, COLOR_BLUE),
        ("Simulated annealing (dimod/neal)", "results/tfim_1d/16/dimod/simulated_annealing/result_1d_h1.0_rbmfull_nh16_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz", None, COLOR_GREEN),
        ("Tabu search (dimod)", "results/tfim_1d/16/dimod/tabu/result_1d_h1.0_rbmfull_nh16_lr0.1_reg1e-05_ns1000_seed*_iter100_cem0.json.gz", None, COLOR_GREEN),
        ("VeloxQ (SA hardware)", None, "velox", "#eb6834"),
        ("FPGA", None, "fpga", "#eb6834"),
        ("D-Wave Pegasus (QPU)", "results/tfim_1d/16/dimod/pegasus/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed*_iter300.json.gz", None, COLOR_MAGENTA),
        ("D-Wave Zephyr (QPU)", "results/tfim_1d/16/dimod/zephyr/result_1d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed*_iter300.json.gz", None, COLOR_MAGENTA),
    ]

    def sweeps_recs(solver):
        # FPGA's sweeps100 campaign now lives under results/tfim_1d/16/fpga/
        # (see fpga_glob docstring); sweeps2000 is a separate campaign, still
        # under results/sweeps2000/. velox is unaffected -- both campaigns
        # still live under results/sweeps{100,2000}/.
        patterns = (
            [fpga_glob(16), "results/sweeps2000/tfim_1d/16/fpga/*/result_*_seed*_iter*"]
            if solver == "fpga" else
            [f"results/{campaign}/tfim_1d/16/{solver}/*/result_*_seed*_iter*" for campaign in ("sweeps100", "sweeps2000")]
        )
        out = []
        for pattern in patterns:
            for r in load(pattern):
                c = r["config"]
                if c["n_hidden"] == 16 and abs(c["learning_rate"] - 0.08) < 1e-9 \
                        and abs(c["regularization"] - 0.05) < 1e-9 and c["n_samples"] == 200:
                    out.append(r)
        return out

    def time_field(r):
        h = r["history"]
        if "total_sampling_time_s" in h:
            return "total_sampling_time_s"
        if "sampling_time_s" in h:
            return "sampling_time_s"
        return None

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ylabels = []
    for i, (label, pattern, sweep_solver, color) in enumerate(rows):
        recs = sweeps_recs(sweep_solver) if sweep_solver else load(pattern)
        y = len(rows) - i
        ylabels.append(label)

        ites = [compute_ite(r["history"]["energy"], r["exact_energy"], 16, epsilon) for r in recs]
        reached_ite = [v for v in ites if v is not None]
        budgets = [len(r["history"]["energy"]) for r in recs]

        ax = axes[0]
        if reached_ite:
            m, lo, hi = median_iqr(reached_ite)
            ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], marker="o", color=color,
                        markersize=8, linewidth=1.6, capsize=4, zorder=3)
        if len(reached_ite) < len(recs) and budgets:
            ax.scatter([max(budgets)], [y], marker="x", color=color, s=60, zorder=4)
        ax.annotate(f"n={len(reached_ite)}/{len(recs)}", (max(budgets) if budgets else 1, y),
                    textcoords="offset points", xytext=(8, 0), fontsize=7.5, color=color, va="center")

        ax = axes[1]
        tf = time_field(recs[0]) if recs else None
        if tf is None:
            ax.annotate(f"{label}: no timing recorded", (0.02, y), xycoords=("axes fraction", "data"),
                        fontsize=8, color=MUTED, va="center", style="italic")
            continue
        cum_times = [np.cumsum(r["history"][tf]) for r in recs]
        ttes = [float(ct[ite - 1]) for ct, ite in zip(cum_times, ites) if ite is not None]
        totals = [float(ct[-1]) for ct in cum_times]
        if ttes:
            m, lo, hi = median_iqr(ttes)
            ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], marker="o", color=color,
                        markersize=8, linewidth=1.6, capsize=4, zorder=3)
        if len(ttes) < len(recs) and totals:
            ax.scatter([max(totals)], [y], marker="x", color=color, s=60, zorder=4)
        ax.annotate(f"n={len(ttes)}/{len(recs)}", (max(totals) if totals else 1, y),
                    textcoords="offset points", xytext=(8, 0), fontsize=7.5, color=color, va="center")

    for ax, xlabel, title in [
        (axes[0], f"ITE — iterations to {epsilon:.3g} energy error/spin (median, IQR; x = censored)", "(A) Iterations-to-epsilon"),
        (axes[1], f"TTE — wall-clock seconds to {epsilon:.3g} energy error/spin (median, IQR; x = censored)", "(B) Time-to-epsilon (QPU omitted, no timing recorded)"),
    ]:
        ax.set_yticks([len(rows) - i for i in range(len(rows))])
        ax.set_yticklabels(ylabels if ax is axes[0] else [])
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0.3, len(rows) + 0.7)
        style_axes(ax)
        ax.grid(axis="x", which="major", color=GRID, linewidth=0.8, zorder=0)

    fig.suptitle(
        "All archived solvers at N=16, each at its own best-available operating point\n"
        "NOT hyperparameter-matched -- see Figure 8 for each row's (h, config, n)", y=1.03
    )
    fig.tight_layout()
    _save(fig, "fig9_ite_tte_all_solvers_n16")


# ---------------------------------------------------------------------------
# Figure 10 — ITE and TTE vs. system size N, all solvers
# ---------------------------------------------------------------------------

def fig10_ite_tte_vs_n_all_solvers(epsilon=0.01):
    def mcmc_recs(solver, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/custom/{solver}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == n and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    def fpga_recs(n):
        out = []
        for r in load(fpga_glob(n)):
            c = r["config"]
            if c["n_hidden"] == n and abs(c["learning_rate"] - 0.08) < 1e-9 \
                    and abs(c["regularization"] - 0.05) < 1e-9 and c["n_samples"] == 200:
                out.append(r)
        return out[:20]

    def dwave_recs(method, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/dimod/{method}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == n and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    def velox_untuned_recs(n):
        with open("results/custom/velox_default_h0.5.json") as f:
            default = json.load(f)
        cfg = default["config"]
        out = []
        for campaign in default["source_campaigns"]:
            pattern = default["source_path_template"].format(campaign=campaign, n=n)
            for r in load(pattern):
                c = r["config"]
                if c["n_hidden"] == n and abs(c["learning_rate"] - cfg["learning_rate"]) < 1e-9 \
                        and abs(c["regularization"] - cfg["regularization"]) < 1e-9 and c["n_samples"] == cfg["n_samples"]:
                    out.append(r)
        return out[:20]

    _sizes = [8, 12, 16, 24, 32, 64, 128]
    _dwave_sizes = [16, 32, 64]  # Zephyr's dense biclique embedding tops out here -- no K_128,128
    _pegasus_sizes = [16, 32, 64, 128]  # Pegasus (Advantage_system6) does embed K_128,128
    series = [
        ("Metropolis", _sizes, lambda n: mcmc_recs("metropolis", n), COLOR_BLUE, "o", "-"),
        ("Gibbs", _sizes, lambda n: mcmc_recs("gibbs", n), COLOR_GREEN, "s", "-"),
        ("VeloxQ (SA, untuned)", _sizes, lambda n: velox_untuned_recs(n), "#eb6834", "P", "-"),
        ("FPGA", _sizes, lambda n: fpga_recs(n), "#ffa600", "X", "-"),
        ("Pegasus (QPU)", _pegasus_sizes, lambda n: dwave_recs("pegasus", n, cem=0), "#bc5090", "*", ":"),
        ("Pegasus (+CEM)", _pegasus_sizes, lambda n: dwave_recs("pegasus", n, cem=1), "#bc5090", "D", "--"),
        ("Zephyr (QPU)", _dwave_sizes, lambda n: dwave_recs("zephyr", n, cem=0), "#ef5675", "*", ":"),
        ("Zephyr (+CEM)", _dwave_sizes, lambda n: dwave_recs("zephyr", n, cem=1), "#ef5675", "h", "--"),
    ]

    def _plot_tte_series(ax, series_idx, label, sizes, get_recs, color, marker, linestyle,
                         annotate=True, markersize=10, linewidth=2.0, capsize=4,
                         annotate_fontsize=9, annotate_offset=(5, 7)):
        tte_med, tte_lo, tte_hi, tte_n, tte_budget = [], [], [], [], []
        for n in sizes:
            recs = get_recs(n)
            timed_recs = [r for r in recs if "sampling_time_s" in r["history"] or "total_sampling_time_s" in r["history"]]
            if timed_recs:
                tf = "total_sampling_time_s" if "total_sampling_time_s" in timed_recs[0]["history"] else "sampling_time_s"
                cum_times = [np.cumsum(r["history"][tf]) for r in timed_recs]
                ites_timed = [compute_ite(r["history"]["energy"], r["exact_energy"], n, epsilon) for r in timed_recs]
                ttes = [float(ct[ite - 1]) for ct, ite in zip(cum_times, ites_timed) if ite is not None]
                m, l, h = median_iqr(ttes) if ttes else (None, None, None)
                tte_med.append(m); tte_lo.append(l); tte_hi.append(h)
                tte_n.append((len(ttes), len(recs)))
                tte_budget.append(max((float(ct[-1]) for ct in cum_times), default=None))
            else:
                tte_med.append(None); tte_lo.append(None); tte_hi.append(None)
                tte_n.append((0, 0)); tte_budget.append(None)

        xs = [n for n, m in zip(sizes, tte_med) if m is not None]
        ys = [m for m in tte_med if m is not None]
        lo_v = [v for v in tte_lo if v is not None]
        hi_v = [v for v in tte_hi if v is not None]
        if xs:
            yerr = [[y - l for y, l in zip(ys, lo_v)], [h - y for y, h in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                        markersize=markersize, linewidth=linewidth, capsize=capsize, zorder=3,
                        linestyle=linestyle)
        else:
            ax.plot([], [], marker=marker, color=color, linestyle=linestyle, label=label)
        cx = [n for n, m, b in zip(sizes, tte_med, tte_budget) if m is None and b is not None]
        cb = [b * (1.0 + 0.05 * series_idx) for m, b in zip(tte_med, tte_budget) if m is None and b is not None]
        if cx:
            ax.scatter(cx, cb, marker=marker, facecolors="none", edgecolors=color,
                       s=(markersize ** 2), linewidth=1.6, zorder=3)
        if annotate:
            for n, (r, total) in zip(sizes, tte_n):
                if total and r < total:
                    y_pos = tte_budget[sizes.index(n)] if tte_med[sizes.index(n)] is None else tte_med[sizes.index(n)]
                    if y_pos is not None:
                        ax.annotate(f"{r}/{total}", (n, y_pos), textcoords="offset points",
                                    xytext=annotate_offset, fontsize=annotate_fontsize,
                                    color=color, ha="left")

    setup_style(fontsize=13)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))

    for series_idx, (label, sizes, get_recs, color, marker, linestyle) in enumerate(series):
        _plot_tte_series(ax, series_idx, label, sizes, get_recs, color, marker, linestyle)

    ax.set_yscale("log")
    ax.set_xscale("log")
    log_x_with_ticks(ax, _sizes)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel(f"TTE to $\\epsilon={epsilon:.3g}$ [s]\n(median, IQR)")
    ax.legend(loc="upper left", fontsize=11, ncol=2, handlelength=1.8, borderpad=0.5)
    ax.set_title(f"Time-to-$\\epsilon$ vs. system size $N$ — all solvers "
                 f"(hyperparameter-matched)", fontsize=13)

    fig.tight_layout()
    _save(fig, f"fig10_ite_tte_vs_n_all_solvers_eps{epsilon:g}")


# ---------------------------------------------------------------------------
# Figure 10b — same as Figure 10, QPUs only (Pegasus/Zephyr, with/without CEM)
# ---------------------------------------------------------------------------

def fig10b_ite_tte_vs_n_qpu_only(epsilon=0.01):
    # Zephyr's embedding fails past N=64; Pegasus supports up to N=128
    def dwave_recs(method, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/dimod/{method}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == n and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    _dwave_sizes = [16, 32, 64]
    _pegasus_sizes = [16, 32, 64, 128]
    series = [
        ("Pegasus (QPU)", _pegasus_sizes, lambda n: dwave_recs("pegasus", n, cem=0), "#bc5090", "*", ":"),
        ("Pegasus (+CEM)", _pegasus_sizes, lambda n: dwave_recs("pegasus", n, cem=1), "#bc5090", "D", "--"),
        ("Zephyr (QPU)", _dwave_sizes, lambda n: dwave_recs("zephyr", n, cem=0), "#ef5675", "*", ":"),
        ("Zephyr (+CEM)", _dwave_sizes, lambda n: dwave_recs("zephyr", n, cem=1), "#ef5675", "h", "--"),
    ]

    def _plot_tte_series(ax, series_idx, label, sizes, get_recs, color, marker, linestyle,
                         annotate=True, markersize=10, linewidth=2.0, capsize=4,
                         annotate_fontsize=9, annotate_offset=(5, 7)):
        tte_med, tte_lo, tte_hi, tte_n, tte_budget = [], [], [], [], []
        for n in sizes:
            recs = get_recs(n)
            timed_recs = [r for r in recs if "sampling_time_s" in r["history"] or "total_sampling_time_s" in r["history"]]
            if timed_recs:
                tf = "total_sampling_time_s" if "total_sampling_time_s" in timed_recs[0]["history"] else "sampling_time_s"
                cum_times = [np.cumsum(r["history"][tf]) for r in timed_recs]
                ites_timed = [compute_ite(r["history"]["energy"], r["exact_energy"], n, epsilon) for r in timed_recs]
                ttes = [float(ct[ite - 1]) for ct, ite in zip(cum_times, ites_timed) if ite is not None]
                m, l, h = median_iqr(ttes) if ttes else (None, None, None)
                tte_med.append(m); tte_lo.append(l); tte_hi.append(h)
                tte_n.append((len(ttes), len(recs)))
                tte_budget.append(max((float(ct[-1]) for ct in cum_times), default=None))
            else:
                tte_med.append(None); tte_lo.append(None); tte_hi.append(None)
                tte_n.append((0, 0)); tte_budget.append(None)

        xs = [n for n, m in zip(sizes, tte_med) if m is not None]
        ys = [m for m in tte_med if m is not None]
        lo_v = [v for v in tte_lo if v is not None]
        hi_v = [v for v in tte_hi if v is not None]
        if xs:
            yerr = [[y - l for y, l in zip(ys, lo_v)], [h - y for y, h in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                        markersize=markersize, linewidth=linewidth, capsize=capsize, zorder=3,
                        linestyle=linestyle)
        else:
            ax.plot([], [], marker=marker, color=color, linestyle=linestyle, label=label)
        cx = [n for n, m, b in zip(sizes, tte_med, tte_budget) if m is None and b is not None]
        cb = [b * (1.0 + 0.05 * series_idx) for m, b in zip(tte_med, tte_budget) if m is None and b is not None]
        if cx:
            ax.scatter(cx, cb, marker=marker, facecolors="none", edgecolors=color,
                       s=(markersize ** 2), linewidth=1.6, zorder=3)
        if annotate:
            for n, (r, total) in zip(sizes, tte_n):
                if total and r < total:
                    y_pos = tte_budget[sizes.index(n)] if tte_med[sizes.index(n)] is None else tte_med[sizes.index(n)]
                    if y_pos is not None:
                        ax.annotate(f"{r}/{total}", (n, y_pos), textcoords="offset points",
                                    xytext=annotate_offset, fontsize=annotate_fontsize,
                                    color=color, ha="left")

    setup_style(fontsize=13)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))

    for series_idx, (label, sizes, get_recs, color, marker, linestyle) in enumerate(series):
        _plot_tte_series(ax, series_idx, label, sizes, get_recs, color, marker, linestyle)

    ax.set_yscale("log")
    ax.set_xscale("log")
    log_x_with_ticks(ax, _pegasus_sizes)
    ax.set_xlabel("System size $N$")
    ax.set_ylabel(f"TTE to $\\epsilon={epsilon:.3g}$ [s]\n(median, IQR)")
    ax.legend(loc="upper left", fontsize=11, handlelength=1.8, borderpad=0.5)
    ax.set_title(f"Time-to-$\\epsilon$ vs. system size $N$ — D-Wave QPUs only "
                 f"(Pegasus/Zephyr, with and without CEM)", fontsize=13)

    fig.tight_layout()
    _save(fig, f"fig10b_ite_tte_vs_n_qpu_only_eps{epsilon:g}")


# ---------------------------------------------------------------------------
# Figure 10c — same as Figure 10, but using the report's published TTE99
# criterion (report.tex sec:exper:tte, Figure 15): the per-seed rolling-window
# sustained crossing of epsilon (compute_validated_convergence_iter) gives,
# per cell, a validated fraction p and a median single-run time T_r; TTE99
# converts these into the literature-standard 99%-confidence repeated-trials
# time (tte99_from_validated), generalizing TTS/TTS99 (arXiv:2401.07184).
# Raw (non-CEM) Pegasus/Zephyr are omitted here: their low validated
# fractions make TTE99 blow up by 1-2 orders of magnitude, which would
# dominate the shared log-scale axis; the CEM-corrected series already make
# the CEM-matters point by contrast.
# ---------------------------------------------------------------------------

def fig10c_tte_vs_n_validated_convergence(window=10, epsilon=0.1, p_target=0.99):
    def mcmc_recs(solver, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/custom/{solver}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == n and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    def fpga_recs(n):
        out = []
        for r in load(fpga_glob(n)):
            c = r["config"]
            if c["n_hidden"] == n and abs(c["learning_rate"] - 0.08) < 1e-9 \
                    and abs(c["regularization"] - 0.05) < 1e-9 and c["n_samples"] == 200:
                out.append(r)
        return out[:20]

    def dwave_recs(method, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/dimod/{method}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == n and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    # Capped at N=64 -- no solver reaches epsilon by N=128 within budget
    _sizes = [8, 12, 16, 24, 32, 64]
    _dwave_sizes = [8, 16, 32, 64]
    _pegasus_sizes = [8, 16, 32, 64]

    # Split into three groups sharing a y-axis: classical MCMC, classical
    # physics-inspired heuristics, and quantum hardware (QPU).
    groups = [
        ("(a) Classical samplers", [
            ("Metropolis", _sizes, lambda n: mcmc_recs("metropolis", n), COLOR_BLUE, "o", "-"),
            ("Gibbs", _sizes, lambda n: mcmc_recs("gibbs", n), COLOR_GREEN, "s", "-"),
        ]),
        ("(b) Classical, physics-inspired", [
            ("Simulated Annealing", _sizes, lambda n: mcmc_recs("simulated_annealing", n), "#6a3d9a", "v", "-."),
            ("FPGA", _sizes, lambda n: fpga_recs(n), "#ffa600", "X", "-"),
        ]),
        ("(c) Quantum annealers (QPU)", [
            ("Pegasus (+CEM)", _pegasus_sizes, lambda n: dwave_recs("pegasus", n, cem=1), "#bc5090", "D", "--"),
            ("Zephyr (+CEM)", _dwave_sizes, lambda n: dwave_recs("zephyr", n, cem=1), "#ef5675", "h", "--"),
        ]),
    ]

    def _plot_tte_series(ax, series_idx, label, sizes, get_recs, color, marker, linestyle):
        tte_med, tte_lo, tte_hi, tte_n, tte_budget = [], [], [], [], []
        for n in sizes:
            recs = get_recs(n)
            timed_recs = [r for r in recs if "sampling_time_s" in r["history"] or "total_sampling_time_s" in r["history"]]
            if timed_recs:
                tf = "total_sampling_time_s" if "total_sampling_time_s" in timed_recs[0]["history"] else "sampling_time_s"
                cum_times = [np.cumsum(r["history"][tf]) for r in timed_recs]
                conv_iters = [compute_validated_convergence_iter(
                    r["history"]["energy"], r["exact_energy"], n, epsilon, window
                ) for r in timed_recs]
                ttes = [float(ct[it - 1]) for ct, it in zip(cum_times, conv_iters) if it is not None]
                m, l, h, _p = tte99_from_validated(ttes, len(recs), p_target)
                tte_med.append(m); tte_lo.append(l); tte_hi.append(h)
                tte_n.append((len(ttes), len(recs)))
                tte_budget.append(max((float(ct[-1]) for ct in cum_times), default=None))
            else:
                tte_med.append(None); tte_lo.append(None); tte_hi.append(None)
                tte_n.append((0, 0)); tte_budget.append(None)

        xs = [n for n, m in zip(sizes, tte_med) if m is not None]
        ys = [m for m in tte_med if m is not None]
        lo_v = [v for v in tte_lo if v is not None]
        hi_v = [v for v in tte_hi if v is not None]
        if xs:
            exp = fit_powerlaw_exponent(xs, ys)
            lbl = f"{label} ($\\propto N^{{{exp:.2f}}}$)" if exp is not None else label
            yerr = [[y - l for y, l in zip(ys, lo_v)], [h - y for y, h in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=lbl,
                        markersize=10, linewidth=2.0, capsize=4, zorder=3, linestyle=linestyle)
        else:
            ax.plot([], [], marker=marker, color=color, linestyle=linestyle, label=label)
        cx = [n for n, m, b in zip(sizes, tte_med, tte_budget) if m is None and b is not None]
        cb = [b * (1.0 + 0.05 * series_idx) for m, b in zip(tte_med, tte_budget) if m is None and b is not None]
        if cx:
            ax.scatter(cx, cb, marker=marker, facecolors="none", edgecolors=color, s=100, linewidth=1.6, zorder=3)
        for n, (r, total) in zip(sizes, tte_n):
            if total and r < total:
                y_pos = tte_budget[sizes.index(n)] if tte_med[sizes.index(n)] is None else tte_med[sizes.index(n)]
                if y_pos is not None:
                    ax.annotate(f"{r}/{total}", (n, y_pos), textcoords="offset points",
                                xytext=(5, 7), fontsize=9, color=color, ha="left")

    setup_style(fontsize=13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # top-left works for (a)/(b); (c)'s QPU data needs top-right instead
    legend_locs = ["upper left", "upper left", "upper right"]

    for ax, (panel_title, panel_series), legend_loc in zip(axes, groups, legend_locs):
        for series_idx, (label, sizes, get_recs, color, marker, linestyle) in enumerate(panel_series):
            _plot_tte_series(ax, series_idx, label, sizes, get_recs, color, marker, linestyle)
        ax.set_yscale("log")
        ax.set_xscale("log")
        log_x_with_ticks(ax, _sizes)
        ax.set_xlabel("System size $N$")
        ax.set_title(panel_title, fontsize=13)
        ax.legend(loc=legend_loc, fontsize=10, handlelength=1.6, borderpad=0.4)

    axes[0].set_ylabel(f"TTE$_{{{p_target*100:.0f}}}$ to $\\epsilon={epsilon:.3g}$ [s]\n(median, IQR)")
    for ax in axes[1:]:
        ax.label_outer()

    fig.suptitle(f"TTE$_{{{p_target*100:.0f}}}$ at convergence (h=0.5, lr=0.08, reg=0.05, ns=200)", fontsize=14)
    fig.tight_layout()
    _save(fig, f"fig10c_tte{p_target*100:.0f}_vs_n_eps{epsilon:g}")


# ---------------------------------------------------------------------------
# Figure 10d — energy-to-convergence vs N. Same cells/methodology as Figure
# 10c (report.tex's published TTE99 criterion: compute_validated_convergence_iter
# per seed, then tte99_from_validated's 99%-confidence conversion applied to
# energy instead of time), plotted as GPU energy (Wh) instead of wall-clock time.
#
# Metropolis/Gibbs read from the main results/tfim_1d/ archive --
# same as fig10c. They used to live in a separate results/energy_corrected/
# tree because the archive's gpu_energy_wh was measured before
# src/energy.py's active()-window fix (whole train() loop, including SR/CG,
# not just the sampler call) and disagreed with the corrected solver-only
# definition by ~2.5x at N=8/metropolis (0.274 Wh archived vs 0.110 Wh
# corrected). Per this repo's rule against hand-patching result artifacts,
# the fix was a full re-run at matched (lr, reg, ns, iterations, seeds) --
# see scripts/exper/mcmc_matched_sweep.py invocations that produced that
# dir. Verified identical (same seeds -> same exact_energy/final-energy
# trajectory, only gpu_energy_wh and wall-clock-noise sampling_time_s
# differ) to the archive's own files for all three solvers, so the
# corrected files were folded back into results/tfim_1d/ (overwriting the
# stale-energy originals) and results/energy_corrected/ was retired.
#
# Simulated Annealing reads from the main archive directly and needed no
# fix: results/tfim_1d/*/custom/simulated_annealing/ was regenerated by
# commit da60d9648 ("reproducible sa"), which has 481310c87 ("energy" --
# the active()-window fix) as an ancestor, so that regeneration already
# ran with the corrected metering. Confirmed numerically too: SA's implied
# average GPU power at N=32 (gpu_energy_wh*3600/sum(sampling_time_s), a
# proxy since raw joules aren't stored) is ~137W, matching the
# already-corrected Metropolis figure (~131W) rather than the ~1.5-2.5x
# inflated pre-fix range. (An earlier pass here wrongly concluded SA was
# NOT energy-corrected, conflating this with a separate, real issue: its
# results/energy_corrected/ copy used the untuned sa_sweeps=1 default and
# was deleted as stale -- that's about mixing quality, not energy
# metering, and doesn't apply to the main archive's sa_sweeps=40 files.)
#
# FPGA is unaffected by the energy-metering bug (it was never GPU-metered;
# its "energy" is assumed constant power x its own archived
# sampling_time_s, both untouched by the fix) so it's read from the same
# official archive fig10c uses.
#
# D-Wave QPU (pegasus/zephyr) is omitted entirely: no API exposes per-job
# QPU energy, and the only published figure (D-Wave's whitepaper, ~25kW
# whole-system draw dominated by a continuously-running dilution
# refrigerator) isn't attributable to a single job.
# ---------------------------------------------------------------------------

FPGA_ASSUMED_POWER_W = 45.0  # matches plot_ite.py's ASSUMED_POWER_W["fpga/fpga"]


def fig10d_energy_vs_n_validated_convergence(window=10, epsilon=0.1, p_target=0.99):
    def mcmc_recs(solver, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/custom/{solver}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == n and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    def fpga_recs(n):
        out = []
        for r in load(fpga_glob(n)):
            c = r["config"]
            if c["n_hidden"] == n and abs(c["learning_rate"] - 0.08) < 1e-9 \
                    and abs(c["regularization"] - 0.05) < 1e-9 and c["n_samples"] == 200:
                out.append(r)
        return out[:20]

    _sizes = [8, 12, 16, 24, 32, 64]

    groups = [
        ("(a) Classical samplers", [
            ("Metropolis", _sizes, lambda n: mcmc_recs("metropolis", n), COLOR_BLUE, "o", "-", None),
            ("Gibbs", _sizes, lambda n: mcmc_recs("gibbs", n), COLOR_GREEN, "s", "-", None),
        ]),
        ("(b) Classical, physics-inspired", [
            ("Simulated Annealing", _sizes, lambda n: mcmc_recs("simulated_annealing", n), "#6a3d9a", "v", "-.", None),
            ("FPGA", _sizes, lambda n: fpga_recs(n), "#ffa600", "X", "-", FPGA_ASSUMED_POWER_W),
        ]),
    ]

    def _energy_at_conv_wh(r, n, power_w):
        conv_iter = compute_validated_convergence_iter(
            r["history"]["energy"], r["exact_energy"], n, epsilon, window
        )
        if conv_iter is None:
            return None
        times = r["history"].get("total_sampling_time_s") or r["history"].get("sampling_time_s")
        if not times:
            return None
        cum_times = np.cumsum(times)
        time_at_conv = float(cum_times[conv_iter - 1])
        if power_w is not None:
            return power_w * time_at_conv / 3600.0
        wh = r.get("gpu_energy_wh")
        total_time = float(cum_times[-1])
        if wh is None or total_time <= 0:
            return None
        return wh * (time_at_conv / total_time)

    def _plot_energy_series(ax, label, sizes, get_recs, color, marker, linestyle, power_w):
        e_med, e_lo, e_hi, e_n = [], [], [], []
        for n in sizes:
            recs = get_recs(n)
            vals = [v for v in (_energy_at_conv_wh(r, n, power_w) for r in recs) if v is not None]
            m, l, h, _p = tte99_from_validated(vals, len(recs), p_target)
            e_med.append(m); e_lo.append(l); e_hi.append(h)
            e_n.append((len(vals), len(recs)))

        xs = [n for n, m in zip(sizes, e_med) if m is not None]
        ys = [m for m in e_med if m is not None]
        lo_v = [v for v in e_lo if v is not None]
        hi_v = [v for v in e_hi if v is not None]
        lbl = f"{label} (assumed)" if power_w is not None else label
        if xs:
            exp = fit_powerlaw_exponent(xs, ys)
            if exp is not None:
                lbl = f"{lbl} ($\\propto N^{{{exp:.2f}}}$)"
            yerr = [[y - l for y, l in zip(ys, lo_v)], [h - y for y, h in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=lbl,
                        markersize=10, linewidth=2.0, capsize=4, zorder=3, linestyle=linestyle)
        else:
            ax.plot([], [], marker=marker, color=color, linestyle=linestyle, label=lbl)
        for n, (r, total) in zip(sizes, e_n):
            if total and r < total:
                idx = sizes.index(n)
                if e_med[idx] is not None:
                    ax.annotate(f"{r}/{total}", (n, e_med[idx]), textcoords="offset points",
                                xytext=(5, 7), fontsize=9, color=color, ha="left")

    setup_style(fontsize=13)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

    # No series in either panel has data past N=32, so the N=64 corner is
    # blank in both panels -- "upper right" (b) and "upper left" (a) land
    # the (opaque, matching fig10c's default legend.frameon styling) box
    # there instead of on top of a curve. An earlier version used "lower
    # left"/frameless in panel (b) to dodge FPGA's line, but a frameless
    # legend just sits illegibly on top of the data instead of hiding it;
    # the empty N=64 corner avoids the trade-off entirely.
    legend_locs = ["upper left", "center right"]
    for ax, (panel_title, panel_series), legend_loc in zip(axes, groups, legend_locs):
        for label, sizes, get_recs, color, marker, linestyle, power_w in panel_series:
            _plot_energy_series(ax, label, sizes, get_recs, color, marker, linestyle, power_w)
        ax.set_yscale("log")
        ax.set_xscale("log")
        log_x_with_ticks(ax, _sizes)
        ax.set_xlabel("System size $N$")
        ax.set_title(panel_title, fontsize=13)
        ax.legend(loc=legend_loc, fontsize=9, handlelength=1.6, borderpad=0.4)

    axes[0].set_ylabel(f"Energy$_{{{p_target*100:.0f}}}$ to convergence [Wh]\n(median, IQR)")
    for ax in axes[1:]:
        ax.label_outer()

    fig.suptitle(
        f"Energy$_{{{p_target*100:.0f}}}$ at convergence (h=0.5, lr=0.08, reg=0.05, ns=200)\n"
        "D-Wave QPU omitted -- no per-job energy telemetry available",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, f"fig10d_energy{p_target*100:.0f}_vs_n_eps{epsilon:g}")


# ---------------------------------------------------------------------------
# Figure — TTE99 vs h at fixed N=16, using the report's published TTE99
# criterion (same criterion as fig10c: compute_validated_convergence_iter +
# tte99_from_validated), Gibbs vs Simulated Annealing, before/at/after h_c=1.
# ---------------------------------------------------------------------------

def fig_tte_vs_h_n16(window=10, epsilon=0.1, p_target=0.99,
                      h_values=(0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5), h_c=1.0):
    N = 16

    def mcmc_recs(solver, h):
        recs = load(f"results/tfim_1d/{N}/custom/{solver}/result_1d_h{h}_rbmfull_nh{N}_lr0.08_reg0.05_ns200_seed*_iter100_cem0_sigma1.0.json.gz")
        return [r for r in recs if r["config"]["n_hidden"] == N and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9 and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    series_defs = [
        ("Gibbs", lambda h: mcmc_recs("gibbs", h), COLOR_GREEN, "s", "-"),
        ("Simulated Annealing", lambda h: mcmc_recs("simulated_annealing", h), "#6a3d9a", "v", "-."),
    ]

    def _plot_tte_vs_h(ax, series_idx, label, get_recs, color, marker, linestyle):
        tte_med, tte_lo, tte_hi, tte_n, tte_budget = [], [], [], [], []
        for h in h_values:
            recs = get_recs(h)
            timed_recs = [r for r in recs if "sampling_time_s" in r["history"] or "total_sampling_time_s" in r["history"]]
            if timed_recs:
                tf = "total_sampling_time_s" if "total_sampling_time_s" in timed_recs[0]["history"] else "sampling_time_s"
                cum_times = [np.cumsum(r["history"][tf]) for r in timed_recs]
                conv_iters = [compute_validated_convergence_iter(
                    r["history"]["energy"], r["exact_energy"], N, epsilon, window
                ) for r in timed_recs]
                ttes = [float(ct[it - 1]) for ct, it in zip(cum_times, conv_iters) if it is not None]
                m, lo, hi, _p = tte99_from_validated(ttes, len(recs), p_target)
                tte_med.append(m); tte_lo.append(lo); tte_hi.append(hi)
                tte_n.append((len(ttes), len(recs)))
                tte_budget.append(max((float(ct[-1]) for ct in cum_times), default=None))
            else:
                tte_med.append(None); tte_lo.append(None); tte_hi.append(None)
                tte_n.append((0, 0)); tte_budget.append(None)

        xs = [h for h, m in zip(h_values, tte_med) if m is not None]
        ys = [m for m in tte_med if m is not None]
        lo_v = [v for v in tte_lo if v is not None]
        hi_v = [v for v in tte_hi if v is not None]
        if xs:
            yerr = [[y - lo for y, lo in zip(ys, lo_v)], [hi - y for y, hi in zip(ys, hi_v)]]
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                        markersize=10, linewidth=2.0, capsize=4, zorder=3, linestyle=linestyle)
        else:
            ax.plot([], [], marker=marker, color=color, linestyle=linestyle, label=label)
        cx = [h for h, m, b in zip(h_values, tte_med, tte_budget) if m is None and b is not None]
        cb = [b * (1.0 + 0.05 * series_idx) for m, b in zip(tte_med, tte_budget) if m is None and b is not None]
        if cx:
            ax.scatter(cx, cb, marker=marker, facecolors="none", edgecolors=color, s=100, linewidth=1.6, zorder=3)
        for idx, h in enumerate(h_values):
            r, total = tte_n[idx]
            if total and r < total:
                y_pos = tte_budget[idx] if tte_med[idx] is None else tte_med[idx]
                if y_pos is not None:
                    ax.annotate(f"{r}/{total}", (h, y_pos), textcoords="offset points",
                                xytext=(5, 7), fontsize=9, color=color, ha="left")

    setup_style(fontsize=13)
    fig, ax = plt.subplots(figsize=(7, 6))
    for series_idx, (label, get_recs, color, marker, linestyle) in enumerate(series_defs):
        _plot_tte_vs_h(ax, series_idx, label, get_recs, color, marker, linestyle)

    ax.axvline(h_c, color=MUTED, linestyle=":", linewidth=1.5, zorder=1)
    ax.text(h_c, 1.01, "$h_c$", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=10, color=MUTED)

    ax.set_yscale("log")
    ax.set_xlabel("Transverse field $h$")
    ax.set_ylabel(f"TTE$_{{{p_target*100:.0f}}}$ to $\\epsilon={epsilon:.3g}$ [s]\n(median, IQR)")
    ax.legend(loc="best", fontsize=10, handlelength=1.6, borderpad=0.4)
    ax.set_title(f"TTE$_{{{p_target*100:.0f}}}$ vs $h$, $N={N}$ (lr=0.08, reg=0.05, ns=200)", fontsize=13)
    fig.tight_layout()
    _save(fig, f"fig_tte{p_target*100:.0f}_vs_h_n{N}_eps{epsilon:g}")


def fig11_appendix_convergence_grid():
    # One row per model. Column 0 compares solvers at one parameter value;
    # columns 1-2 sweep the model's physical parameter with its richest
    # classical solver. Models without both an exact reference energy and
    # matched-hyperparameter multi-seed data are excluded.
    SOLVER_COLORS = {
        "Metropolis (CPU)": "#1f77b4",
        "Gibbs (CPU)": "#2ca02c",
        "FPGA": "#17becf",
        "VeloxQ (SA)": "#9467bd",
        "D-Wave Zephyr (QPU)": "#d62728",
        "D-Wave Pegasus (QPU)": "#ff7f0e",
    }

    def load_trace(path):
        matches = glob.glob(path)
        if len(matches) != 1:
            return None
        with gzip.open(matches[0]) as fh:
            r = json.load(fh)
        return r["exact_energy"], np.array(r["history"]["energy"])

    def plot_solver_group(ax, label, paths):
        color = SOLVER_COLORS[label]
        exact, traces = None, []
        for path in paths:
            loaded = load_trace(path)
            if loaded is None:
                continue
            exact, energies = loaded
            traces.append(energies)
        if not traces:
            return None
        if len(traces) >= 5:
            n = min(len(t) for t in traces)
            stacked = np.stack([t[:n] for t in traces])
            med = np.nanmedian(stacked, axis=0)
            lo, hi = np.nanpercentile(stacked, [25, 75], axis=0)
            x = np.arange(1, n + 1)
            ax.plot(x, med, color=color, linewidth=1.4, label=label, zorder=3)
            ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0, zorder=2)
        else:
            for i, t in enumerate(traces):
                ax.plot(np.arange(1, len(t) + 1), t, color=color, alpha=0.7, linewidth=1.2,
                         label=label if i == 0 else None)
        return exact, np.concatenate(traces)

    def clip_ylim(ax, exact, pool):
        scale = 20 * (abs(exact) + 1)
        pool = pool[np.isfinite(pool) & (np.abs(pool - exact) < scale)]
        if not pool.size:
            return
        lo, hi = np.percentile(pool, [1, 99])
        span = max(hi - lo, abs(exact) * 0.05, 1e-3)
        ax.set_ylim(min(lo, exact) - 0.5 * span, max(hi, exact) + 2 * span)

    def sweep_panel(ax, color, dirpath, tmpl, pval, seeds):
        exact, traces = None, []
        for seed in seeds:
            loaded = load_trace(f"{dirpath}/{tmpl.format(p=pval, s=seed)}")
            if loaded is None:
                continue
            exact, energies = loaded
            traces.append(energies)
            ax.plot(np.arange(1, len(energies) + 1), energies, color=color, alpha=0.6, linewidth=1.1)
        if exact is not None and traces:
            clip_ylim(ax, exact, np.concatenate(traces))
        return exact

    rows = [
        dict(
            title="TFIM 1D",
            solver_panel_title="TFIM 1D, h=0.5 -- solvers",
            solver_groups={
                "Metropolis (CPU)": [f"results/tfim_1d/16/custom/metropolis/result_1d_h0.5_rbmfull_nh16_lr0.08_reg0.05_ns200_seed{s}_iter100_cem0_sigma1.0.json.gz" for s in range(20)],
                "FPGA": [f"results/tfim_1d/16/fpga/fpga/result_1d_h0.5_rbmfull_nh16_lr0.08_reg0.05_ns200_seed{s}_iter100_cem0_sigma1.0.json.gz" for s in range(20)],
                "VeloxQ (SA)": [f"results/sweeps100/tfim_1d/16/velox/simulated_annealing/result_1d_h0.5_rbmfull_nh16_lr0.08_reg0.05_ns200_seed{s}_iter100_cem0_sigma1.0.json.gz" for s in range(20)],
                "D-Wave Zephyr (QPU)": [f"results/tfim_1d/16/dimod/zephyr/result_1d_h0.5_rbmfull_nh16_lr0.08_reg0.05_ns200_seed{s}_iter100_cem0_sigma1.0.json.gz" for s in range(1, 20)],
            },
            sweep_color=COLOR_BLUE,
            sweep_dir="results/tfim_1d/16/custom/metropolis",
            sweep_tmpl="result_1d_h{p}_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed{s}_iter300_cem0_sigma1.0.json.gz",
            pname="h", pvals=[1.0, 1.5], seeds=[1, 7, 42, 123],
        ),
        dict(
            title="TFIM 2D (L=2)",
            solver_panel_title="TFIM 2D (L=2), h=0.5 -- solvers",
            solver_groups={
                "Metropolis (CPU)": [f"results/tfim_2d/4/custom/metropolis/result_2d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed{s}_iter300.json.gz" for s in (1, 42)],
                "VeloxQ (SA)": [f"results/tfim_2d/4/velox/velox/result_2d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed{s}_iter300.json.gz" for s in (1, 42)],
                "D-Wave Pegasus (QPU)": [f"results/tfim_2d/4/dimod/pegasus/result_2d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed{s}_iter300.json.gz" for s in (1, 42)],
                "D-Wave Zephyr (QPU)": [f"results/tfim_2d/4/dimod/zephyr/result_2d_h0.5_rbmfull_nh16_lr0.1_reg0.001_ns1000_seed{s}_iter300.json.gz" for s in (1, 42)],
            },
            sweep_color=COLOR_GREEN,
            sweep_dir="results/tfim_2d/4/custom/metropolis",
            sweep_tmpl="result_2d_h{p}_rbmfull_nh16_lr0.01_reg1e-05_ns1000_seed{s}_iter300_cem0_sigma1.0.json.gz",
            pname="h", pvals=[1.0, 2.0], seeds=[1, 2, 3, 4, 5],
        ),
        dict(
            title="Long-range TFIM 1D",
            solver_panel_title="LR-TFIM 1D, alpha=1.0 -- solvers",
            solver_groups={
                "Metropolis (CPU)": [f"results/lr_tfim_1d/16/custom/metropolis/result_lr1d_h0.5_alpha1.0_rbmfull_nh16_lr0.01_reg1e-05_ns1000_seed{s}_iter300_cem0_sigma1.0.json.gz" for s in (1, 2, 3, 4, 5)],
                "Gibbs (CPU)": [f"results/lr_tfim_1d/16/custom/gibbs/result_lr1d_h0.5_alpha1.0_rbmfull_nh16_lr0.01_reg1e-05_ns1000_seed{s}_iter300_cem0_sigma1.0.json.gz" for s in (1, 2, 3, 4, 5)],
            },
            sweep_color=COLOR_MAGENTA,
            sweep_dir="results/lr_tfim_1d/16/custom/gibbs",
            sweep_tmpl="result_lr1d_h0.5_alpha{p}_rbmfull_nh16_lr0.01_reg1e-05_ns1000_seed{s}_iter300_cem0_sigma1.0.json.gz",
            pname="alpha", pvals=[0.5, 2.0], seeds=[1, 2, 3, 4, 5],
        ),
    ]
    n_cols = 3
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(14, 3.1 * len(rows)), squeeze=False)

    for row, spec in enumerate(rows):
        ax0 = axes[row][0]
        exact0, pooled = None, []
        for label, paths in spec["solver_groups"].items():
            result = plot_solver_group(ax0, label, paths)
            if result is not None:
                exact0, tr = result
                pooled.append(tr)
        if exact0 is not None:
            ax0.axhline(exact0, color=INK, linestyle="--", linewidth=1.0, zorder=4)
            clip_ylim(ax0, exact0, np.concatenate(pooled))
        ax0.set_title(spec["solver_panel_title"], fontsize=9)
        ax0.legend(fontsize=6, loc="lower right", frameon=True)
        style_axes(ax0)
        ax0.set_ylabel("Energy")

        for col, pval in enumerate(spec["pvals"], start=1):
            ax = axes[row][col]
            exact = sweep_panel(ax, spec["sweep_color"], spec["sweep_dir"], spec["sweep_tmpl"], pval, spec["seeds"])
            if exact is not None:
                ax.axhline(exact, color=INK, linestyle="--", linewidth=1.0, zorder=4)
            ax.set_title(f"{spec['title']}, {spec['pname']}={pval}", fontsize=9)
            style_axes(ax)

        for col in range(n_cols):
            if row == len(rows) - 1:
                axes[row][col].set_xlabel("SR iteration")

    fig.suptitle(
        "Appendix: convergence across models, solvers, parameters and seeds (N=16)\n"
        "col 0: multi-solver comparison (median+IQR band where seeds>=5, else individual seeds), dashed = exact energy\n"
        "cols 1-2: parameter sweep, single classical solver, individual seeds\n"
        "solver panels use one shared hyperparameter cell per model, not each solver's own tuned config -- some spread\n"
        "(e.g. VeloxQ and one Metropolis seed settling on a metastable plateau in TFIM 2D) reflects that, not a plotting artifact",
        y=1.05, fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "fig11_appendix_convergence_grid")


# ---------------------------------------------------------------------------
# Figure 12 -- appendix grid of convergence trajectories across models,
# sizes, and solvers. TFIM 1D panels use the exact same matched cell as
# fig10c/fig10d (h=0.5, lr=0.08, reg=0.05, ns=200, iterations=100, 20 seeds)
# so this is a trajectory-level view of the same archive those TTE/energy
# figures summarize into scalars -- not a separate/independent sweep.
#
# Layout: two stacked blocks, no blank cells.
#   Top block:    3 rows (N=8,12,16) x 4 cols (TFIM-classical, TFIM-physics/
#                 HW, Heisenberg J2=0.3, Heisenberg J2=0.5) -- every model has
#                 data at these 3 sizes.
#   Bottom block: 3 rows (N=24,32,64) x 2 cols (TFIM-classical, TFIM-physics/
#                 HW only -- Heisenberg has no archive past N=16). Using only
#                 2 columns lets each bottom panel take the full page width
#                 instead of leaving a blank Heisenberg-shaped hole.
# Heisenberg panels use Gibbs + Simulated Annealing only (Exchange isn't
# discussed elsewhere in the paper and was dropped to avoid introducing an
# unexplained solver here) at J2/J1=0.3 (below the Majumdar-Ghosh point) and
# 0.5 (at it), each the single dominant per-instance-tuned hyperparameter
# cell archived for that (N, method, J2) -- these aren't matched to a single
# shared (lr, reg) across N the way the TFIM cell is.
# ---------------------------------------------------------------------------

def fig12_appendix_size_solver_grid():
    def matched_cell(recs, n):
        return [r for r in recs if r["config"]["n_hidden"] == n
                and abs(r["config"]["learning_rate"] - 0.08) < 1e-9
                and abs(r["config"]["regularization"] - 0.05) < 1e-9
                and r["config"]["n_samples"] == 200
                and r["config"]["iterations"] == 100][:20]

    def tfim_mcmc_recs(method, n, cem=0):
        recs = load(f"results/tfim_1d/{n}/custom/{method}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return matched_cell(recs, n)

    def tfim_fpga_recs(n):
        # fpga_glob() doesn't pin lr/reg/ns/iterations in the filename (unlike
        # the custom/dimod solvers above) -- filter on config explicitly.
        out = []
        for r in load(fpga_glob(n)):
            c = r["config"]
            if c["n_hidden"] == n and abs(c["learning_rate"] - 0.08) < 1e-9 \
                    and abs(c["regularization"] - 0.05) < 1e-9 and c["n_samples"] == 200 \
                    and c["iterations"] == 100:
                out.append(r)
        return out[:20]

    def tfim_dwave_recs(method, n, cem=1):
        recs = load(f"results/tfim_1d/{n}/dimod/{method}/result_1d_h0.5_rbmfull_nh{n}_lr0.08_reg0.05_ns200_seed*_iter100_cem{cem}_sigma1.0.json.gz")
        return matched_cell(recs, n)

    def heis_recs(method, n, j2):
        # cem varies per archived (N, method, J2) cell (whichever the
        # per-instance hyperparameter search happened to land on) -- glob
        # across both rather than assume cem0.
        return load(f"results/heisenberg_j1j2_1d/{n}/custom/{method}/result_heisenberg_j1j2_1d_J11.0_J2{j2}_delta*_seed*_iter300_cem*_sigma1.0.json.gz")

    def plot_band(ax, recs, color, label):
        traces, exact = [], None
        for r in recs:
            e = r["history"].get("energy")
            if not e:
                continue
            traces.append(np.array(e))
            exact = r.get("exact_energy", exact)
        if not traces:
            return None
        n = min(len(t) for t in traces)
        stacked = np.stack([t[:n] for t in traces])
        med = np.nanmedian(stacked, axis=0)
        lo, hi = np.nanpercentile(stacked, [25, 75], axis=0)
        x = np.arange(1, n + 1)
        ax.plot(x, med, color=color, linewidth=1.3, label=label, zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        return exact, stacked

    def clip_ylim(ax, exact, pools):
        all_vals = np.concatenate([p.ravel() for p in pools])
        scale = 20 * (abs(exact) + 1)
        all_vals = all_vals[np.isfinite(all_vals) & (np.abs(all_vals - exact) < scale)]
        if not all_vals.size:
            return
        lo, hi = np.percentile(all_vals, [1, 99])
        span = max(hi - lo, abs(exact) * 0.05, 1e-3)
        ax.set_ylim(min(lo, exact) - 0.3 * span, max(hi, exact) + 1.2 * span)

    # Colors match fig10c/fig10d exactly so a solver reads identically
    # across the paper.
    color_sa, color_fpga, color_pegasus, color_zephyr = "#6a3d9a", "#ffa600", "#bc5090", "#ef5675"

    tfim_row1 = [
        ("Metropolis", COLOR_BLUE, lambda n: tfim_mcmc_recs("metropolis", n)),
        ("Gibbs", COLOR_GREEN, lambda n: tfim_mcmc_recs("gibbs", n)),
        ("Sim. Annealing", color_sa, lambda n: tfim_mcmc_recs("simulated_annealing", n)),
    ]
    tfim_row2 = [
        ("FPGA", color_fpga, lambda n: tfim_fpga_recs(n)),
        ("Pegasus (+CEM)", color_pegasus, lambda n: tfim_dwave_recs("pegasus", n, cem=1)),
        ("Zephyr (+CEM)", color_zephyr, lambda n: tfim_dwave_recs("zephyr", n, cem=1)),
    ]
    heis_row_easy = [
        ("Gibbs", COLOR_GREEN, lambda n: heis_recs("gibbs", n, "0.3")),
        ("Sim. Annealing", color_sa, lambda n: heis_recs("simulated_annealing", n, "0.3")),
    ]
    heis_row_mg = [
        ("Gibbs", COLOR_GREEN, lambda n: heis_recs("gibbs", n, "0.5")),
        ("Sim. Annealing", color_sa, lambda n: heis_recs("simulated_annealing", n, "0.5")),
    ]

    def draw_panel(ax, series, n, show_legend):
        exact, pools = None, []
        for label, color, get_recs in series:
            result = plot_band(ax, get_recs(n), color, label)
            if result is not None:
                exact, stacked = result
                pools.append(stacked)
        if exact is not None:
            ax.axhline(exact, color=INK, linestyle="--", linewidth=0.9, zorder=4)
            clip_ylim(ax, exact, pools)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=8)
        style_axes(ax)
        if show_legend:
            ax.legend(fontsize=6, loc="best", frameon=True, handlelength=1.5, borderpad=0.3)

    setup_style(fontsize=9)
    fig = plt.figure(figsize=(11.5, 11.3))
    top_fig, bot_fig = fig.subfigures(2, 1, height_ratios=[3, 3], hspace=0.0)

    top_sizes, bot_sizes = [8, 12, 16], [24, 32, 64]
    top_cols = [
        ("TFIM classical", tfim_row1),
        ("TFIM physics/HW", tfim_row2),
        ("Heisenberg J2=0.3", heis_row_easy),
        ("Heisenberg J2=0.5", heis_row_mg),
    ]
    bot_cols = [
        ("TFIM classical", tfim_row1),
        ("TFIM physics/HW", tfim_row2),
    ]

    def fill_block(subfig, sizes, cols, fontsize_title=9.5):
        axes = subfig.subplots(len(sizes), len(cols), squeeze=False)
        for col, (col_title, series) in enumerate(cols):
            for row, n in enumerate(sizes):
                ax = axes[row][col]
                draw_panel(ax, series, n, show_legend=(row == 0))
                if row == 0:
                    ax.set_title(col_title, fontsize=fontsize_title)
                if row == len(sizes) - 1:
                    ax.set_xlabel("SR iteration", fontsize=8)
        # Explicit left/right margins -- subfigures' subplots() otherwise
        # only fills part of the subfigure's actual width (a matplotlib
        # subfigure quirk without constrained_layout), leaving the rest
        # visibly blank regardless of column count.
        subfig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.11,
                                hspace=0.5, wspace=0.32)
        return axes

    top_axes = fill_block(top_fig, top_sizes, top_cols)
    bot_axes = fill_block(bot_fig, bot_sizes, bot_cols, fontsize_title=10.5)

    fig.suptitle(
        "Appendix: convergence trajectories across models, sizes, and solvers\n"
        "solid = median over seeds, band = IQR, dashed = exact ground-state energy",
        fontsize=12, y=1.0,
    )

    # Bold "N=" row tags in each block's own left margin, positioned from the
    # actual post-layout bbox of that block's first column -- much more
    # visible than an in-axes ylabel competing with the y-tick numbers.
    for subfig, axes, sizes in ((top_fig, top_axes, top_sizes), (bot_fig, bot_axes, bot_sizes)):
        for ax, n in zip(axes[:, 0], sizes):
            bbox = ax.get_position()  # already in subfig-local figure coords
            y_center = (bbox.y0 + bbox.y1) / 2
            subfig.text(0.01, y_center, f"N={n}", rotation=90, ha="center", va="center",
                        fontsize=12, fontweight="bold")

    _save(fig, "fig12_appendix_size_solver_grid")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_ite_vs_n_tfim1d()
    fig2_tte_vs_n_velox_fpga()
    fig3_energy_to_solution_vs_n()
    fig4_success_fraction_heisenberg()
    fig5_convergence_dwave_tfim1d()
    fig6_energy_vs_n_tfim2d_dwave()
    fig7_classical_scaling_tfim1d()
    fig8_all_solvers_n16()
    fig9_ite_tte_all_solvers_n16()
    fig10_ite_tte_vs_n_all_solvers()
    fig10c_tte_vs_n_validated_convergence()
    fig10d_energy_vs_n_validated_convergence()
    fig11_appendix_convergence_grid()
    fig12_appendix_size_solver_grid()

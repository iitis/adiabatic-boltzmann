"""
Visualization of experiment_kl_convergence.py results for PRA/PRL.

Generates four publication-quality figures:
  Fig 1 — convergence curves (median ± IQR relative energy error vs iteration)
  Fig 2 — final relative error distributions (violin + strip, sampler × h)
  Fig 3 — KL divergence vs final energy error scatter (per h, Spearman ρ)
  Fig 4 — H1 vs H2: Spearman ρ(KL, error) and ρ(grad_norm, error) per h

Only files from the five experiment samplers AND containing all required KL
fields (final_kl_exact, ess, kl_exact history) are loaded.

Usage
-----
    cd src
    python visualize_kl_convergence.py              # all figures → figures/
    python visualize_kl_convergence.py --fig 3      # only figure 3
    python visualize_kl_convergence.py --show       # display interactively
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import spearmanr

# Use non-interactive backend when no display is needed (overridden by --show)
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Experiment-defined samplers (from experiment_kl_convergence.py)
# ---------------------------------------------------------------------------

SAMPLERS = {
    "metropolis": ("custom", "metropolis"),
    "gibbs":      ("custom", "gibbs"),
    "sa_custom":  ("custom", "simulated_annealing"),
    "sa_dimod":   ("dimod", "simulated_annealing"),
    "tabu":       ("dimod", "tabu"),
}

# Display labels and colours — fixed so all figures share the same legend
SAMPLER_LABEL = {
    "metropolis": "Metropolis",
    "gibbs":      "Gibbs",
    "sa_custom":  "SA (custom)",
    "sa_dimod":   "SA (dimod)",
    "tabu":       "Tabu",
}
# Colour-blind-safe palette (Wong 2011)
SAMPLER_COLOR = {
    "metropolis": "#0072B2",
    "gibbs":      "#E69F00",
    "sa_custom":  "#009E73",
    "sa_dimod":   "#CC79A7",
    "tabu":       "#D55E00",
}

REQUIRED_TOP  = {"final_kl_exact", "final_ess", "error", "exact_energy", "final_energy"}
REQUIRED_HIST = {"kl_exact", "grad_norm", "energy", "ess"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _sampler_key(backend: str, method: str) -> str | None:
    for key, (b, m) in SAMPLERS.items():
        if backend == b and method == m:
            return key
    return None


def load_runs(results_root: Path) -> list[dict]:
    """
    Load all valid experiment runs. Each entry is a flat dict with:
        sampler_key, model, size, h, seed,
        rel_error, final_kl, final_ess, grad_norm_final,
        energy_hist, kl_hist, grad_norm_hist, n_iter
    """
    runs = []
    for f in sorted(results_root.rglob("result_*_cem0.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue

        c = d.get("config", {})
        key = _sampler_key(c.get("sampler", ""), c.get("sampling_method", ""))
        if key is None:
            continue

        if not REQUIRED_TOP.issubset(d.keys()):
            continue
        hist = d.get("history", {})
        if not REQUIRED_HIST.issubset(hist.keys()):
            continue
        if d.get("final_kl_exact") is None:
            continue

        exact = d["exact_energy"]
        final_e = d["final_energy"]
        rel_error = abs(final_e - exact) / abs(exact)

        # Relative-error history (filter out None)
        energy_hist = hist["energy"]
        rel_err_hist = [
            abs(e - exact) / abs(exact) if e is not None else None
            for e in energy_hist
        ]
        kl_hist        = hist["kl_exact"]
        grad_norm_hist = hist["grad_norm"]

        runs.append(dict(
            sampler_key    = key,
            model          = c["model"],
            size           = c["size"],
            h              = c["h"],
            seed           = c.get("seed"),
            rel_error      = rel_error,
            final_kl       = d["final_kl_exact"],
            final_ess      = d["final_ess"],
            grad_norm_final= grad_norm_hist[-1] if grad_norm_hist else None,
            rel_err_hist   = rel_err_hist,
            kl_hist        = kl_hist,
            grad_norm_hist = grad_norm_hist,
            n_iter         = len(energy_hist),
        ))

    return runs


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

def _apply_style():
    """
    PRL/PRA-grade style:
      - LaTeX rendering (Computer Modern serif, matching journal body text)
      - 8 pt base font (typical for two-column APS journals)
      - Hairline axes (0.5 pt), no top/right spines
      - 300 dpi output
    """
    plt.rcParams.update({
        # STIX Two serif fonts — bundled with matplotlib, visually equivalent
        # to Computer Modern at journal resolution; no system LaTeX required.
        "text.usetex":          False,
        "mathtext.fontset":     "stix",
        "font.family":          "STIXGeneral",
        # Font sizes — APS guideline: figure labels ≈ 8 pt
        "font.size":            8,
        "axes.labelsize":       8,
        "axes.titlesize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      7,
        # Tick geometry (APS style: ticks pointing inward)
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.minor.size":     1.5,
        "ytick.minor.size":     1.5,
        "xtick.major.width":    0.5,
        "ytick.major.width":    0.5,
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        # Axes
        "axes.linewidth":       0.5,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.titlepad":        4,
        # Lines
        "lines.linewidth":      1.0,
        "patch.linewidth":      0.5,
        # Output
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "pdf.fonttype":         42,    # embed fonts (no Type 3) for submission
        "ps.fonttype":          42,
    })


# ---------------------------------------------------------------------------
# Helper: curve aggregation
# ---------------------------------------------------------------------------

def _aggregate_curves(curves: list[list]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a list of equal-length arrays, return (median, q25, q75) at each step.
    None values in any curve at position i are excluded from the aggregate at i.
    """
    n = max(len(c) for c in curves)
    med = np.full(n, np.nan)
    q25 = np.full(n, np.nan)
    q75 = np.full(n, np.nan)
    for i in range(n):
        vals = [c[i] for c in curves if i < len(c) and c[i] is not None]
        if vals:
            med[i] = np.median(vals)
            q25[i] = np.percentile(vals, 25)
            q75[i] = np.percentile(vals, 75)
    return med, q25, q75


# ---------------------------------------------------------------------------
# Figure 1 — convergence curves
# ---------------------------------------------------------------------------

def fig_convergence_curves(runs: list[dict], out_dir: Path, show: bool):
    """
    Median relative energy error vs. iteration, one panel per h value.
    Restricted to 1D N=8 (best-populated cell, Part 1 main result).
    """
    H_VALUES = [0.5, 1.0, 1.5, 2.0]
    H_LABELS = [r"$h=0.5$", r"$h=1.0$ (critical)", r"$h=1.5$", r"$h=2.0$"]

    # PRL double-column width = 7.0 in; height chosen for readability
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=False)

    for ax, h, hlabel in zip(axes, H_VALUES, H_LABELS):
        for key in SAMPLERS:
            curves = [
                r["rel_err_hist"] for r in runs
                if r["model"] == "1d" and r["size"] == 8
                and r["h"] == h and r["sampler_key"] == key
            ]
            if not curves:
                continue
            med, q25, q75 = _aggregate_curves(curves)
            xs = np.arange(len(med))
            c  = SAMPLER_COLOR[key]
            ax.plot(xs, med, color=c, label=SAMPLER_LABEL[key])
            ax.fill_between(xs, q25, q75, color=c, alpha=0.18)

        ax.set_title(hlabel)
        ax.set_xlabel("Iteration")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.tick_params(axis="y", which="both")

    axes[0].set_ylabel(r"Relative error $|\Delta E|/|E_0|$")

    handles = [
        mlines.Line2D([0], [0], color=SAMPLER_COLOR[k], label=SAMPLER_LABEL[k])
        for k in SAMPLERS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.suptitle(r"VMC convergence — 1D TFIM, $N=8$", y=1.02)

    fig.tight_layout()
    _save(fig, out_dir, "fig1_convergence_curves", show)


# ---------------------------------------------------------------------------
# Figure 2 — final error distributions (violin + strip)
# ---------------------------------------------------------------------------

def fig_error_distributions(runs: list[dict], out_dir: Path, show: bool):
    """
    Violin plots of final relative error per sampler, faceted by h.
    Restricted to 1D N=8.
    """
    H_VALUES = [0.5, 1.0, 1.5, 2.0]
    sampler_keys = list(SAMPLERS.keys())
    xs = np.arange(len(sampler_keys))

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)

    for ax, h in zip(axes, H_VALUES):
        for xi, key in enumerate(sampler_keys):
            vals = [
                r["rel_error"] for r in runs
                if r["model"] == "1d" and r["size"] == 8
                and r["h"] == h and r["sampler_key"] == key
            ]
            if not vals:
                continue
            c = SAMPLER_COLOR[key]
            parts = ax.violinplot([vals], positions=[xi], widths=0.7,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(c)
                pc.set_alpha(0.55)
            parts["cmedians"].set_color(c)
            parts["cmedians"].set_linewidth(1.5)
            jitter = np.random.default_rng(seed=42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(xi + jitter, vals, color=c, s=8, alpha=0.7, zorder=3)

        # LaTeX-safe title: asterisk is fine in math mode
        h_label = r"$h=1.0^*$" if h == 1.0 else f"$h={h}$"
        ax.set_title(h_label)
        ax.set_xticks(xs)
        ax.set_xticklabels([SAMPLER_LABEL[k] for k in sampler_keys],
                           rotation=30, ha="right")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    axes[0].set_ylabel(r"Final relative error $|\Delta E|/|E_0|$")
    fig.suptitle(r"Final energy error — 1D TFIM, $N=8$  ($^*$\,critical point)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_error_distributions", show)


# ---------------------------------------------------------------------------
# Figure 3 — KL vs error scatter
# ---------------------------------------------------------------------------

def fig_kl_scatter(runs: list[dict], out_dir: Path, show: bool):
    """
    Scatter of final_kl_exact vs relative error, one panel per h.
    Spearman ρ annotated. Restricted to 1D N=8.
    """
    H_VALUES = [0.5, 1.0, 1.5, 2.0]
    H_LABELS = [r"$h=0.5$", r"$h=1.0$*", r"$h=1.5$", r"$h=2.0$"]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=False)

    for ax, h, hlabel in zip(axes, H_VALUES, H_LABELS):
        all_kl  = []
        all_err = []
        for key in SAMPLERS:
            pts = [
                (r["final_kl"], r["rel_error"]) for r in runs
                if r["model"] == "1d" and r["size"] == 8
                and r["h"] == h and r["sampler_key"] == key
                and r["final_kl"] is not None
            ]
            if not pts:
                continue
            kls, errs = zip(*pts)
            ax.scatter(kls, errs, color=SAMPLER_COLOR[key], s=14,
                       alpha=0.75, label=SAMPLER_LABEL[key], zorder=3)
            all_kl.extend(kls)
            all_err.extend(errs)

        if len(all_kl) >= 3:
            _sr  = spearmanr(all_kl, all_err)
            rho  = float(_sr.statistic)  # type: ignore
            pval = float(_sr.pvalue)     # type: ignore
            p_str = r"$p<0.001$" if pval < 0.001 else rf"$p={pval:.3f}$"
            # Two separate text calls to avoid \n inside LaTeX mode
            ax.text(0.97, 0.13, rf"$\rho={rho:.2f}$",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="0.75", alpha=0.9))
            ax.text(0.97, 0.02, p_str,
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color="0.4")

        ax.set_title(hlabel)
        ax.set_xlabel(r"$D_{\mathrm{KL}}(q \| |\Psi|^2)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    axes[0].set_ylabel(r"Final relative error $|\Delta E|/|E_0|$")

    handles = [
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=SAMPLER_COLOR[k], markersize=5,
                      label=SAMPLER_LABEL[k])
        for k in SAMPLERS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.suptitle(
        r"KL divergence vs. final energy error — 1D TFIM, $N=8$  ($^*$\,critical point)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig3_kl_scatter", show)


# ---------------------------------------------------------------------------
# Figure 4 — H1 vs H2: Spearman ρ per h
# ---------------------------------------------------------------------------

def fig_h1_vs_h2(runs: list[dict], out_dir: Path, show: bool):
    """
    Bar chart: Spearman ρ(KL, error) and ρ(grad_norm, error) vs h.
    Pooled over all 5 samplers × 15 seeds (1D N=8).
    Error bars = 95 % CI from bootstrap (1000 resamples).
    """
    H_VALUES = [0.5, 1.0, 1.5, 2.0]
    rng = np.random.default_rng(seed=0)

    rho_kl_med, rho_kl_lo, rho_kl_hi = [], [], []
    rho_gn_med, rho_gn_lo, rho_gn_hi = [], [], []

    for h in H_VALUES:
        pts = [
            (r["final_kl"], r["grad_norm_final"], r["rel_error"])
            for r in runs
            if r["model"] == "1d" and r["size"] == 8 and r["h"] == h
            and r["final_kl"] is not None and r["grad_norm_final"] is not None
        ]
        kls, gns, errs = map(np.array, zip(*pts))

        def _rho(x: np.ndarray, y: np.ndarray) -> float:
            return float(spearmanr(x, y).statistic)  # type: ignore

        rho_kl_med.append(_rho(kls, errs))
        rho_gn_med.append(_rho(gns, errs))

        # Bootstrap CI
        n = len(kls)
        boot_kl, boot_gn = [], []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            boot_kl.append(_rho(kls[idx], errs[idx]))
            boot_gn.append(_rho(gns[idx], errs[idx]))
        rho_kl_lo.append(rho_kl_med[-1] - np.percentile(boot_kl, 2.5))
        rho_kl_hi.append(np.percentile(boot_kl, 97.5) - rho_kl_med[-1])
        rho_gn_lo.append(rho_gn_med[-1] - np.percentile(boot_gn, 2.5))
        rho_gn_hi.append(np.percentile(boot_gn, 97.5) - rho_gn_med[-1])

    xs = np.arange(len(H_VALUES))
    w  = 0.32
    # PRL single-column width = 3.375 in
    fig, ax = plt.subplots(figsize=(3.375, 2.8))

    ax.bar(xs - w/2, rho_kl_med, w,
           yerr=[rho_kl_lo, rho_kl_hi],
           color="#0072B2", alpha=0.85,
           label=r"$\rho(D_{\mathrm{KL}},\,\epsilon)$ [H1]",
           error_kw=dict(elinewidth=0.8, capsize=2.5))
    ax.bar(xs + w/2, rho_gn_med, w,
           yerr=[rho_gn_lo, rho_gn_hi],
           color="#D55E00", alpha=0.85,
           label=r"$\rho(\|\nabla\|,\,\epsilon)$ [H2]",
           error_kw=dict(elinewidth=0.8, capsize=2.5))

    ax.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
    ax.set_xticks(xs)
    xtick_labels = [r"$h=0.5$", r"$h=1.0^*$", r"$h=1.5$", r"$h=2.0$"]
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(
        "KL (H1) vs. SR grad norm (H2)\n"
        r"as convergence predictors, $N=8$"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.97, r"$^*$ critical point", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="0.5")

    fig.tight_layout()
    _save(fig, out_dir, "fig4_h1_vs_h2", show)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, out_dir: Path, name: str, show: bool):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fig", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Generate only this figure (default: all)")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively (requires a display)")
    parser.add_argument("--results-dir", default="../results",
                        help="Path to results root (default: ../results)")
    parser.add_argument("--out-dir", default="../figures/fig_vmc_convergence",
                        help="Output directory for saved figures (default: ../figures/fig_vmc_convergence)")
    cli = parser.parse_args()

    results_root = Path(cli.results_dir)
    out_dir      = Path(cli.out_dir)

    print(f"Loading runs from {results_root} …")
    runs = load_runs(results_root)
    print(f"  {len(runs)} valid runs loaded.")

    _apply_style()

    figs = [cli.fig] if cli.fig else [1, 2, 3, 4]
    dispatch = {
        1: fig_convergence_curves,
        2: fig_error_distributions,
        3: fig_kl_scatter,
        4: fig_h1_vs_h2,
    }
    for n in figs:
        print(f"Generating figure {n} …")
        dispatch[n](runs, out_dir, cli.show)

    print("Done.")


if __name__ == "__main__":
    main()

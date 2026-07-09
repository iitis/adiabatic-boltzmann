"""
plot_cem_dynamics.py -- training-time effect of CEM beta scheduling.

For matched (h, seed) instances that differ ONLY in --cem, on TFIM 1D with
the LSB sampler at N = KL_EXACT_MAX_N (src/encoder.py) -- small enough that
the exact KL divergence to the RBM's own |Psi(v)|^2 marginal is computed
by full enumeration every iteration -- this plots the full training
trajectory (every seed as a thin trace, median as a bold line) for two
quantities:

  (1) sample quality   D_KL(P_sample || P_exact) vs iteration.
                        Lower means the LSB sampler's output more closely
                        tracks the distribution the RBM actually defines.
  (2) temperature est.  beta_x vs iteration (dashed: naive beta=1).
                        With CEM, beta_x is an EMA of a per-iteration
                        least-squares fit (src/encoder.py:estimate_beta_eff_cem).
                        Without it, beta_x is a blind random-walk heuristic
                        driven only by whether the energy improved.

One column per h value, so both effects are visible across field strengths
rather than collapsed into a single final-iteration number.

Usage (from project root):
    python scripts/viz/plot_cem_dynamics.py
"""

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from plot_style import load_json, setup_style

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "tfim_1d" / "16" / "custom" / "lsb"
PLOTS_DIR = ROOT / "plots" / "cem"

TARGET_LR = 0.01
MIN_SEEDS = 3

# Okabe-Ito colorblind-safe categorical pair.
COLOR_NOCEM = "#888888"   # neutral gray -- baseline / control
COLOR_CEM = "#0072B2"     # blue -- CEM treatment


def discover_matched_runs(results_dir, lr, min_seeds):
    """Group runs by (h, seed); keep only h values with >= min_seeds seeds
    that have both a cem=on and cem=off run at the target lr."""
    by_h_seed = defaultdict(dict)
    for path in glob.glob(str(results_dir / "*.json*")):
        try:
            d = load_json(path)
        except Exception:
            continue
        c = d["config"]
        if c.get("sampling_method") != "lsb" or c.get("learning_rate") != lr:
            continue
        cem_val = c.get("cem")
        if cem_val not in (True, False):
            continue
        by_h_seed[(c["h"], c["seed"])][cem_val] = d["history"]

    by_h = defaultdict(list)
    for (h, seed), runs in by_h_seed.items():
        if True in runs and False in runs:
            by_h[h].append({"seed": seed, "cem": runs[True], "nocem": runs[False]})

    return {h: runs for h, runs in sorted(by_h.items()) if len(runs) >= min_seeds}


def _stack(runs, key):
    """(n_seeds, n_iters) array, truncated to the shortest run."""
    arrs = [np.asarray(r[key], dtype=float) for r in runs]
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs])


def _spaghetti(ax, values, color, label):
    """Thin per-seed traces + a bold median line.

    With only a handful of seeds, mean +/- std can be misleading: it implies
    Gaussian scatter around a stable mean, but seeds can instead land on
    genuinely different outcomes (see h=0.5, no-CEM, where 5 seeds land
    between KL~0.1 and KL~11 -- a std band there both overstates a "typical"
    value and can dip negative on a log axis). Showing every trace makes
    that spread legible instead of averaging it away.
    """
    n_seeds, n_iters = values.shape
    x = np.arange(n_iters)
    for row in values:
        ax.plot(x, row, color=color, alpha=0.35, lw=0.7, zorder=2)
    median = np.median(values, axis=0)
    ax.plot(x, median, color=color, label=label, lw=1.8, zorder=3,
             path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])


def plot_dynamics(by_h, fname):
    setup_style(fontsize=10, scale=1.0)
    h_values = list(by_h.keys())
    fig, axes = plt.subplots(2, len(h_values), figsize=(3.1 * len(h_values), 5.0), sharex=True)
    if len(h_values) == 1:
        axes = axes[:, None]

    for col, h in enumerate(h_values):
        runs = by_h[h]
        n_seeds = len(runs)

        kl_nocem = _stack([{"kl": r["nocem"]["kl_exact"]} for r in runs], "kl")
        kl_cem = _stack([{"kl": r["cem"]["kl_exact"]} for r in runs], "kl")
        bx_nocem = _stack([{"bx": r["nocem"]["beta_x"]} for r in runs], "bx")
        bx_cem = _stack([{"bx": r["cem"]["beta_x"]} for r in runs], "bx")

        ax_kl = axes[0, col]
        _spaghetti(ax_kl, kl_nocem, COLOR_NOCEM, "No CEM")
        _spaghetti(ax_kl, kl_cem, COLOR_CEM, "CEM")
        ax_kl.set_yscale("log")
        ax_kl.set_title(f"h = {h:g}  (n={n_seeds} seeds)")
        if col == 0:
            ax_kl.set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{sample}} \Vert P_{\mathrm{exact}})$")

        ax_bx = axes[1, col]
        _spaghetti(ax_bx, bx_nocem, COLOR_NOCEM, "No CEM")
        _spaghetti(ax_bx, bx_cem, COLOR_CEM, "CEM")
        ax_bx.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6, zorder=0)
        ax_bx.set_xlabel("iteration")
        if col == 0:
            ax_bx.set_ylabel(r"effective $\beta_x$")

    handles = [
        plt.Line2D([0], [0], color=COLOR_NOCEM, label="No CEM (heuristic)"),
        plt.Line2D([0], [0], color=COLOR_CEM, label="CEM"),
        plt.Line2D([0], [0], color="black", lw=0.8, ls="--", label=r"naive $\beta=1$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04),
               frameon=True, edgecolor="black")
    fig.suptitle("TFIM 1D, N=16, LSB sampler -- CEM training dynamics", y=1.11, fontsize=11, fontweight="bold")
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def _summary(by_h, tail_frac=0.2):
    for h, runs in by_h.items():
        kl_nocem = _stack([{"kl": r["nocem"]["kl_exact"]} for r in runs], "kl")
        kl_cem = _stack([{"kl": r["cem"]["kl_exact"]} for r in runs], "kl")
        n = kl_nocem.shape[1]
        tail = max(1, int(n * tail_frac))
        m_nocem = kl_nocem[:, -tail:].mean()
        m_cem = kl_cem[:, -tail:].mean()
        bx_nocem_final = _stack([{"bx": r["nocem"]["beta_x"]} for r in runs], "bx")[:, -1]
        bx_cem_final = _stack([{"bx": r["cem"]["beta_x"]} for r in runs], "bx")[:, -1]
        print(
            f"  h={h:g}: tail KL  no-cem={m_nocem:.3f}  cem={m_cem:.3f}  "
            f"({'CEM better' if m_cem < m_nocem else 'no-cem better'})   "
            f"final beta_x  no-cem={bx_nocem_final.mean():.2f}+/-{bx_nocem_final.std():.2f}  "
            f"cem={bx_cem_final.mean():.2f}+/-{bx_cem_final.std():.2f}"
        )


if __name__ == "__main__":
    by_h = discover_matched_runs(RESULTS_DIR, TARGET_LR, MIN_SEEDS)
    print(f"Matched h values (>= {MIN_SEEDS} seeds, lr={TARGET_LR}): {list(by_h.keys())}")
    if not by_h:
        print("No matched runs found -- nothing to plot.")
    else:
        _summary(by_h)
        plot_dynamics(by_h, "cem_dynamics_tfim1d_n16")

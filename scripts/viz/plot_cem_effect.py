"""
plot_cem_effect.py — sample-level effect of CEM beta scheduling.

For every result pair that differs ONLY in the --cem flag (same model,
size, h, learning_rate, seed), compares:
  (1) sample quality  — exact KL divergence to the true Boltzmann
                         distribution, D_KL(P_sample || P_exact), averaged
                         over the last 20% of training iterations
  (2) temperature est. — final beta_x (the sampler's effective inverse
                          temperature, self-consistently estimated by CEM
                          when enabled, or heuristically random-walked
                          otherwise)

split into non-D-Wave (custom/lsb) and D-Wave (dimod/zephyr, dimod/pegasus)
sampler families, since D-Wave's physical effective temperature is
uncontrolled in the same way LSB's is (see cem_article.pdf, Fig. 2).

Only instances with N <= KL_EXACT_MAX_N (exact enumeration threshold in
src/encoder.py) have kl_exact populated, so the comparison is restricted
to that size range.

Usage (from project root):
    python scripts/viz/plot_cem_effect.py
"""

import glob
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import load_json

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots" / "cem_effect"

KL_EXACT_MAX_N = 16  # must match src/encoder.py — exact KL only computed below this
KL_DIVERGED_CLIP = 12.0  # display cap for inf/nan KL (diverged run), flagged separately

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_NO_CEM = "#7f7f7f"
COLOR_CEM = "#1f77b4"
MARKER_BY_METHOD = {"lsb": "o", "zephyr": "s", "pegasus": "D"}

GROUPS = {
    "Non-D-Wave (LSB)": {
        "lsb": "custom/lsb",
    },
    "D-Wave (zephyr / pegasus)": {
        "zephyr": "dimod/zephyr",
        "pegasus": "dimod/pegasus",
    },
}


def _tail_mean(values, frac=0.2):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, False
    tail = vals[-max(1, math.ceil(len(vals) * frac)):]
    finite = [v for v in tail if math.isfinite(v)]
    diverged = len(finite) < len(tail)
    if not finite:
        return KL_DIVERGED_CLIP, True
    m = float(np.mean(finite))
    if diverged or m > KL_DIVERGED_CLIP:
        return KL_DIVERGED_CLIP, True
    return m, False


def load_matched_pairs(method_globs: dict) -> list[dict]:
    """One entry per (method, N, h, lr, seed) instance with both cem=0 and cem=1 runs."""
    pairs = []
    for method, subpath in method_globs.items():
        files = glob.glob(str(RESULTS_DIR / "tfim_1d" / "*" / subpath / "*.json*"))
        by_key = defaultdict(dict)
        for p in files:
            try:
                d = load_json(p)
            except Exception:
                continue
            c = d["config"]
            n = c.get("size")
            if n is None or n > KL_EXACT_MAX_N:
                continue
            cem_val = c.get("cem")
            if cem_val not in (True, False):
                continue  # legacy/incomplete configs without a real cem flag
            key = (n, c.get("h"), c.get("learning_rate"), c.get("seed"))
            by_key[key][cem_val] = d

        for key, runs in by_key.items():
            if True not in runs or False not in runs:
                continue
            n, h, lr, seed = key
            kl_true, div_true = _tail_mean(runs[True]["history"].get("kl_exact", []))
            kl_false, div_false = _tail_mean(runs[False]["history"].get("kl_exact", []))
            bx_true = runs[True]["history"].get("beta_x", [])
            bx_false = runs[False]["history"].get("beta_x", [])
            if kl_true is None or kl_false is None or not bx_true or not bx_false:
                continue
            pairs.append({
                "method": method, "N": n, "h": h, "lr": lr, "seed": seed,
                "label": f"{method}  N={n}, h={h:g}, lr={lr:g}",
                "kl_cem": kl_true, "kl_diverged_cem": div_true,
                "kl_nocem": kl_false, "kl_diverged_nocem": div_false,
                "beta_cem": bx_true[-1], "beta_nocem": bx_false[-1],
            })
    pairs.sort(key=lambda r: (r["method"], r["N"], r["h"], r["lr"]))
    return pairs


def _dumbbell(ax, pairs, val_key_nocem, val_key_cem, log_x=False, diverged_key=None):
    ys = np.arange(len(pairs))
    for y, r in zip(ys, pairs):
        x0, x1 = r[val_key_nocem], r[val_key_cem]
        ax.plot([x0, x1], [y, y], color="#cccccc", lw=1.2, zorder=1)
        ax.scatter(x0, y, s=26, color=COLOR_NO_CEM,
                   marker=MARKER_BY_METHOD[r["method"]], zorder=3,
                   edgecolors="black", linewidths=0.3)
        ax.scatter(x1, y, s=26, color=COLOR_CEM,
                   marker=MARKER_BY_METHOD[r["method"]], zorder=3,
                   edgecolors="black", linewidths=0.3)
        if diverged_key and r[f"{diverged_key}_nocem"]:
            ax.annotate("diverged", (x0, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=6, color=COLOR_NO_CEM, style="italic")
        if diverged_key and r[f"{diverged_key}_cem"]:
            ax.annotate("diverged", (x1, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=6, color=COLOR_CEM, style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in pairs])
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    ax.invert_yaxis()
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)


def plot_group(group_title: str, pairs: list[dict], fname: str):
    if not pairs:
        print(f"  [skip] {group_title}: no matched cem pairs found")
        return

    fig_h = max(2.2, 0.32 * len(pairs) + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, fig_h))

    _dumbbell(axes[0], pairs, "kl_nocem", "kl_cem", log_x=True, diverged_key="kl_diverged")
    axes[0].set_xlabel(r"$D_{\mathrm{KL}}(P_{\mathrm{sample}} \| P_{\mathrm{exact}})$  (log scale)")
    axes[0].set_title("Sample quality")

    _dumbbell(axes[1], pairs, "beta_nocem", "beta_cem", log_x=False)
    axes[1].axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel(r"effective $\beta_x$ (final iteration)")
    axes[1].set_title(r"Temperature estimate  (dashed: naive $\beta=1$)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NO_CEM,
                   markeredgecolor="black", markeredgewidth=0.3, markersize=6, label="No CEM"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CEM,
                   markeredgecolor="black", markeredgewidth=0.3, markersize=6, label="CEM"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
               frameon=True, edgecolor="black")
    fig.suptitle(group_title, y=1.08, fontsize=11, fontweight="bold")
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def _summary(group_title, pairs):
    if not pairs:
        return
    kl_wins = sum(1 for r in pairs if r["kl_cem"] < r["kl_nocem"])
    mean_nocem = np.mean([r["kl_nocem"] for r in pairs])
    mean_cem = np.mean([r["kl_cem"] for r in pairs])
    print(f"  {group_title}: CEM improves sample quality in {kl_wins}/{len(pairs)} instances "
          f"(mean KL {mean_nocem:.2f} -> {mean_cem:.2f})")


if __name__ == "__main__":
    for group_title, method_globs in GROUPS.items():
        pairs = load_matched_pairs(method_globs)
        fname = "cem_effect_" + ("nondwave" if "Non-D-Wave" in group_title else "dwave")
        print(f"{group_title}: {len(pairs)} matched cem on/off pairs")
        _summary(group_title, pairs)
        plot_group(group_title, pairs, fname)

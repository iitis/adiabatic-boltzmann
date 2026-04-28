"""
Plot beta_x trajectories grouped by sigma (σ⁻², paper convention)
for a given (N, h) combination across all available n_hidden values.

Usage:
    python scripts/plot_beta_x_sigma.py --size 24 --h 0.5
    python scripts/plot_beta_x_sigma.py --size 24 --h 0.5 --nh 24
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "results"
PLOTS_DIR = REPO / "plots" / "beta_x"


def load_lsb(size: int, h: float, nh: int | None):
    """Return all LSB result dicts matching (size, h) and optionally n_hidden."""
    records = []
    for path in RESULTS_DIR.rglob("*.json"):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        if (
            cfg.get("sampling_method") == "lsb"
            and cfg.get("size") == size
            and abs(cfg.get("h", -1) - h) < 1e-9
            and "sigma" in cfg
            and (nh is None or cfg.get("n_hidden") == nh)
            and "beta_x" in d.get("history", {})
        ):
            records.append(d)
    return records


def _aggregate(runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(r) for r in runs)
    padded = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        padded[i, : len(r)] = r
    iters = np.arange(1, max_len + 1)
    return iters, np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def plot_one(size: int, h: float, nh: int, records: list, ax: plt.Axes):
    """Plot beta_x curves for one n_hidden panel."""
    by_sigma: dict[float, list] = defaultdict(list)
    for d in records:
        series = d["history"].get("beta_x")
        if series:
            by_sigma[d["config"]["sigma"]].append(np.array(series, dtype=float))

    if not by_sigma:
        ax.set_visible(False)
        return

    for sigma_val in sorted(by_sigma):
        runs = by_sigma[sigma_val]
        iters, mean, std = _aggregate(runs)
        sigma_noise = 1.0 / np.sqrt(sigma_val)
        label = f"σ⁻²={sigma_val:.2g}  (σ={sigma_noise:.3g})"
        if len(runs) > 1:
            label += f"  [n={len(runs)}]"
        (line,) = ax.plot(iters, mean, label=label)
        if len(runs) > 1:
            ax.fill_between(
                iters, mean - std, mean + std, alpha=0.2, color=line.get_color()
            )

    ax.set_title(f"n_hidden={nh}", fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("β_x")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot(size: int, h: float, nh: int | None):
    records = load_lsb(size, h, nh)
    if not records:
        print(f"No LSB results with beta_x found for size={size} h={h}" +
              (f" nh={nh}" if nh else ""))
        return

    # Collect distinct n_hidden values present in results
    nh_values = sorted({d["config"]["n_hidden"] for d in records}) if nh is None \
        else [nh]

    ncols = min(len(nh_values), 3)
    nrows = (len(nh_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, nh_val in enumerate(nh_values):
        ax = axes[idx // ncols][idx % ncols]
        subset = [d for d in records if d["config"]["n_hidden"] == nh_val]
        plot_one(size, h, nh_val, subset, ax)

    # Hide unused axes
    for idx in range(len(nh_values), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"β_x trajectories — N={size}, h={h}  (LSB, grouped by σ⁻²)", fontsize=11)
    plt.tight_layout()

    nh_tag = f"_nh{nh}" if nh is not None else ""
    out = PLOTS_DIR / f"beta_x_N{size}_h{h}{nh_tag}.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True, help="Number of visible spins")
    parser.add_argument("--h", type=float, required=True, help="Transverse field strength")
    parser.add_argument("--nh", type=int, default=None,
                        help="Filter to a specific n_hidden (default: all found)")
    args = parser.parse_args()
    plot(args.size, args.h, args.nh)

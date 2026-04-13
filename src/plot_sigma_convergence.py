"""
Plot LSB convergence curves grouped by sigma (σ⁻², paper convention)
for a given (N, h, n_hidden) combination.

Usage:
    python plot_sigma_convergence.py --size 24 --h 0.5 --nh 24
    python plot_sigma_convergence.py --size 24 --h 0.5 --nh 24 --metric error
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_matching(size: int, h: float, nh: int, method: str):
    """Return all result dicts for a given sampling method matching (size, h, n_hidden)."""
    records = []
    for path in RESULTS_DIR.rglob("*.json"):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        if (
            cfg.get("sampling_method") == method
            and cfg.get("size") == size
            and abs(cfg.get("h", -1) - h) < 1e-9
            and cfg.get("n_hidden") == nh
        ):
            records.append(d)
    return records


def _aggregate(runs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a list of 1-D arrays (possibly different lengths), return (iters, mean, std)."""
    max_len = max(len(r) for r in runs)
    padded = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        padded[i, : len(r)] = r
    iters = np.arange(1, max_len + 1)
    return iters, np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def plot(size: int, h: float, nh: int, metric: str, out: Path | None):
    records = load_matching(size, h, nh, method="lsb")
    if not records:
        print(f"No LSB results found for size={size} h={h} nh={nh}")
        return

    # Group by sigma value
    by_sigma: dict[float, list] = defaultdict(list)
    for d in records:
        sigma_val = d["config"]["sigma"]
        series = d["history"].get(metric)
        if series:
            by_sigma[sigma_val].append(np.array(series, dtype=float))

    exact = records[0].get("exact_energy") / size

    fig, ax = plt.subplots(figsize=(8, 5))

    for sigma_val in sorted(by_sigma):
        runs = by_sigma[sigma_val]
        iters, mean, std = _aggregate(runs)
        mean = mean / size
        std = std / size

        sigma_noise = 1.0 / np.sqrt(sigma_val)
        label = f"σ⁻²={sigma_val:.2g}  (σ={sigma_noise:.3g})"
        if len(runs) > 1:
            label += f"  [n={len(runs)}]"

        (line,) = ax.plot(iters, mean, label=label)
        if len(runs) > 1:
            ax.fill_between(
                iters, mean - std, mean + std, alpha=0.2, color=line.get_color()
            )

    # Gibbs baseline
    gibbs_records = load_matching(size, h, nh, method="gibbs")
    gibbs_series = [
        np.array(d["history"][metric], dtype=float)
        for d in gibbs_records
        if metric in d["history"]
    ]
    if gibbs_series:
        g_iters, g_mean, g_std = _aggregate(gibbs_series)
        g_mean = g_mean / size
        g_std = g_std / size
        ax.plot(g_iters, g_mean, color="black", linestyle="--", linewidth=1.5,
                label=f"Gibbs (n={len(gibbs_series)})")
        if len(gibbs_series) > 1:
            ax.fill_between(g_iters, g_mean - g_std, g_mean + g_std, alpha=0.15, color="black")

    if metric == "energy" and exact is not None:
        ax.axhline(
            exact,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"E_exact = {exact:.4f}",
        )

    ax.set_xlabel("Iteration")
    ylabel = {
        "energy": "⟨E⟩",
        "error": "|E − E_exact| / |E_exact|",
        "kl_exact": "KL divergence",
    }.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.set_title(f"LSB convergence — N={size}, h={h}, n_hidden={nh}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if metric == "error":
        ax.set_yscale("log")

    plt.tight_layout()

    if out is None:
        out = (
            RESULTS_DIR.parent
            / "plots"
            / f"sigma_convergence_N{size}_h{h}_nh{nh}_{metric}.png"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=int, required=True, help="Number of visible spins"
    )
    parser.add_argument(
        "--h", type=float, required=True, help="Transverse field strength"
    )
    parser.add_argument("--nh", type=int, required=True, help="Number of hidden units")
    parser.add_argument(
        "--metric",
        default="energy",
        choices=[
            "energy",
            "error",
            "kl_exact",
            "grad_norm",
            "ess",
            "n_unique_ratio",
            "beta_x",
        ],
        help="History key to plot (default: energy)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output path for the figure"
    )
    args = parser.parse_args()

    plot(args.size, args.h, args.nh, args.metric, args.out)

"""
#!/usr/bin/env python3
plot_cem_comparison.py — CEM vs. no-CEM convergence comparison.

Scans results/ for runs with cem=1 and pairs them with matching cem=0 runs
(same model, N, h, rbm, sampler, method, lr, n_hidden).

One figure per (model, N, h) combination with two panels:
  left:  energy per spin vs. iteration  (with exact reference line)
  right: |E - E_exact| per spin vs. iteration (semilogy)

Each paired configuration is drawn as:
  solid  line = no CEM   (heuristic β_x adaptation)
  dashed line = CEM      (CEM β scheduling)
with matching colour per (rbm, sampler, method) triple.

Usage (from src/):
  python plots/fig_cem.py
  python plots/fig_cem.py --size 8 --h 0.5
  python plots/fig_cem.py --model 1d --size 16
"""

import argparse
import json
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures" / "fig_cem"

# 2D reference energies per spin (thermodynamic limit)
EXACT_ENERGY_2D_PER_SPIN = {
    0.5: -2.0555,
    1.0: -2.1276,
    2.0: -2.4549,
    3.044: -3.0440,
}


# ── Exact energy ─────────────────────────────────────────────────────────────


def compute_exact_energy(model, N, h):
    """Return exact ground-state energy per spin."""
    if model == "2d":
        if h not in EXACT_ENERGY_2D_PER_SPIN:
            print(f"  [warn] No 2D reference energy for h={h}")
            return None
        return EXACT_ENERGY_2D_PER_SPIN[h]

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "exact_diag_ising_analytical.py"),
                "-N",
                str(N),
                "-g",
                str(h),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[-1]
            return float(line.split(": ")[-1])
        print(f"  [warn] exact_diag returned error: {result.stderr.strip()}")
        return None
    except Exception as e:
        print(f"  [warn] Failed to compute exact energy for N={N}, h={h}: {e}")
        return None


# ── Load results ──────────────────────────────────────────────────────────────


def load_results(results_dir: Path, size_filter=None, h_filter=None, model_filter=None):
    """
    Load all JSON results and group them for CEM comparison.

    Configuration key: (model, N, h, rbm, sampler, sampling_method)
    Filters applied (matching plot_results_exact.py):
      - learning_rate == 0.1
      - n_hidden == n_visible

    Returns
    -------
    paired : dict
        {config_key: {"cem": [runs], "no_cem": [runs]}}
        Only keys that have at least one run in BOTH slots are kept.
    """
    groups = defaultdict(lambda: {"cem": [], "no_cem": []})

    for json_file in results_dir.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            cfg = data["config"]
            model = cfg["model"]
            N = cfg["size"]
            h = cfg["h"]
            rbm = cfg["rbm"]
            sampler = cfg["sampler"]
            method = cfg["sampling_method"]
            lr = cfg["learning_rate"]
            n_hidden = cfg["n_hidden"]
            use_cem = bool(cfg.get("cem", False))  # old files default to False

            n_visible = N if model == "1d" else N * N

            # Same filters as plot_results_exact.py
            if lr != 0.1:
                continue
            if n_hidden != n_visible:
                continue

            # Optional CLI filters
            if model_filter and model != model_filter:
                continue
            if size_filter and N != size_filter:
                continue
            if h_filter and abs(h - h_filter) > 1e-9:
                continue

            key = (model, N, h, rbm, sampler, method)
            slot = "cem" if use_cem else "no_cem"
            groups[key][slot].append(
                {
                    "data": data,
                    "config": cfg,
                    "n_visible": n_visible,
                }
            )

        except Exception as e:
            print(f"  [warn] Could not load {json_file}: {e}")

    # Keep only keys with data in both slots
    paired = {k: v for k, v in groups.items() if v["cem"] and v["no_cem"]}
    return paired


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _mean_energy(runs):
    """Average energy curves across runs, NaN-padded to the longest run."""
    energies = [r["data"]["history"]["energy"] for r in runs]
    max_len = max(len(e) for e in energies)
    padded = np.full((len(energies), max_len), np.nan)
    for i, e in enumerate(energies):
        padded[i, : len(e)] = e
    return np.nanmean(padded, axis=0)


def _mean_beta_eff_cem(runs):
    """Average beta_eff_cem values across seeds at each iteration (ignoring None)."""
    series_list = [r["data"]["history"].get("beta_eff_cem", []) for r in runs]
    if not series_list or not series_list[0]:
        return None, None
    min_len = min(len(s) for s in series_list)
    iters, vals = [], []
    for i in range(min_len):
        row = [s[i] for s in series_list if s[i] is not None]
        if row:
            iters.append(i)
            vals.append(float(np.mean(row)))
    return iters, vals


# ── Main plot ─────────────────────────────────────────────────────────────────


def plot_cem_comparison(paired):
    """
    One figure per (model, N, h) combination.

    Layout: 3 panels per figure
      1. Energy per spin (linear)
      2. |E - E_exact| per spin (semilogy)
      3. Estimated β_eff from CEM runs (where available)
    """
    # Group config keys by (model, N, h)
    nh_groups = defaultdict(list)
    for model, N, h, rbm, sampler, method in paired:
        nh_groups[(model, N, h)].append((rbm, sampler, method))

    # Assign a colour per (rbm, sampler, method) triple
    cmap = plt.get_cmap("tab10")
    color_cache = {}

    def get_color(rbm, sampler, method):
        key = (rbm, sampler, method)
        if key not in color_cache:
            color_cache[key] = cmap(len(color_cache) % 10)
        return color_cache[key]

    saved = []

    for (model, N, h), triples in sorted(nh_groups.items()):
        print(
            f"\n(model={model}, N={N}, h={h})  — {len(triples)} paired configuration(s)"
        )

        exact = compute_exact_energy(model, N, h)
        if exact is None:
            print("  Skipping — no exact energy available.")
            continue

        n_visible = N if model == "1d" else N * N

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"CEM β scheduling comparison — model={model}, N={N}, h={h}",
            fontsize=13,
            fontweight="bold",
        )

        has_beta_eff = False

        for rbm, sampler, method in sorted(triples):
            config_key = (model, N, h, rbm, sampler, method)
            slot = paired[config_key]
            label = f"{rbm}/{sampler}/{method}"
            color = get_color(rbm, sampler, method)

            # ── Energy ───────────────────────────────────────────────────
            for runs, ls, suffix in [
                (slot["no_cem"], "solid", "no CEM"),
                (slot["cem"], "dashed", "CEM"),
            ]:
                mean_E = _mean_energy(runs) / n_visible
                iters = np.arange(len(mean_E))
                axes[0].plot(
                    iters,
                    mean_E,
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    alpha=0.85,
                    label=f"{label} [{suffix}]",
                )
                delta = np.abs(mean_E - exact)
                axes[1].semilogy(
                    iters,
                    delta,
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    alpha=0.85,
                    label=f"{label} [{suffix}]",
                )

            # ── β_eff from CEM runs ───────────────────────────────────────
            cem_iters, cem_vals = _mean_beta_eff_cem(slot["cem"])
            if cem_iters:
                axes[2].plot(
                    cem_iters,
                    cem_vals,
                    "o-",
                    color=color,
                    linewidth=1.5,
                    markersize=4,
                    label=label,
                )
                has_beta_eff = True

        # Panel 1 — energy
        axes[0].axhline(
            exact,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Exact: {exact:.5f}",
            zorder=10,
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("# iterations")
        axes[0].set_ylabel("Energy per spin")
        axes[0].set_title("Energy convergence")
        axes[0].grid(True, alpha=0.3, which="both")
        axes[0].legend(fontsize=8, loc="best")

        # Panel 2 — error
        axes[1].set_xscale("log")
        axes[1].set_xlabel("# iterations")
        axes[1].set_ylabel("|E − E_exact| per spin")
        axes[1].set_title("Error to ground state")
        axes[1].grid(True, alpha=0.3, which="both")
        axes[1].legend(fontsize=8, loc="best")

        # Panel 3 — β_eff
        if has_beta_eff:
            axes[2].axhline(
                1.0, color="crimson", linestyle="--", linewidth=1.5, label="β=1 (ideal)"
            )
            axes[2].set_xlabel("# iterations")
            axes[2].set_ylabel("β_eff (CEM estimate)")
            axes[2].set_title("Estimated β_eff (CEM runs)")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=8, loc="best")
        else:
            axes[2].set_visible(False)

        plt.tight_layout()

        out_dir = FIGURES_DIR / model
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"cem_N{N}_h{h}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved → {fname}")
        plt.show()
        saved.append(str(fname))
        plt.close(fig)

    return saved


# ── Summary overview (one page) ───────────────────────────────────────────────


def plot_cem_summary(paired):
    """
    Single summary figure with all paired (model, N, h) groups laid out in a
    grid — one subfigure per group, each with two panels (energy + error).
    CEM runs are drawn dashed, no-CEM solid, same colour per (rbm, sampler, method).

    Saved as figures/fig_cem/cem_summary.png.
    """
    cmap = plt.get_cmap("tab10")
    color_cache = {}

    def get_color(rbm, sampler, method):
        key = (rbm, sampler, method)
        if key not in color_cache:
            color_cache[key] = cmap(len(color_cache) % 10)
        return color_cache[key]

    # Collect (model, N, h) groups that have a valid exact energy
    nh_groups = defaultdict(list)
    for model, N, h, rbm, sampler, method in paired:
        nh_groups[(model, N, h)].append((rbm, sampler, method))

    valid = []
    for (model, N, h), triples in sorted(nh_groups.items()):
        exact = compute_exact_energy(model, N, h)
        if exact is None:
            print(f"  Skipping (model={model}, N={N}, h={h}) — no exact energy.")
            continue
        valid.append((model, N, h, triples, exact))

    if not valid:
        print("  No groups with exact energy — summary not generated.")
        return []

    n = len(valid)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(10 * ncols, 5 * nrows))
    fig.suptitle("CEM β scheduling — all runs overview", fontsize=15, fontweight="bold")

    subfigs = np.array(fig.subfigures(nrows, ncols, wspace=0.04, hspace=0.08)).reshape(
        nrows, ncols
    )

    for idx, (model, N, h, triples, exact) in enumerate(valid):
        row, col = divmod(idx, ncols)
        subfig = subfigs[row, col]
        subfig.patch.set_facecolor("#f5f5f5")
        subfig.patch.set_edgecolor("#aaaaaa")
        subfig.patch.set_linewidth(1.2)
        subfig.suptitle(f"model={model}  N={N}  h={h}", fontsize=11, fontweight="bold")

        n_visible = N if model == "1d" else N * N
        ax1, ax2 = subfig.subplots(1, 2)

        for rbm, sampler, method in sorted(triples):
            slot = paired[(model, N, h, rbm, sampler, method)]
            label = f"{rbm}/{sampler}/{method}"
            color = get_color(rbm, sampler, method)

            for runs, ls, suffix in [
                (slot["no_cem"], "solid", "no CEM"),
                (slot["cem"], "dashed", "CEM"),
            ]:
                mean_E = _mean_energy(runs) / n_visible
                iters = np.arange(len(mean_E))
                delta = np.abs(mean_E - exact)

                ax1.plot(
                    iters,
                    mean_E,
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"{label} [{suffix}]",
                )
                ax2.semilogy(
                    iters,
                    delta,
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"{label} [{suffix}]",
                )

        ax1.axhline(
            exact,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Exact: {exact:.4f}",
            zorder=10,
        )

        for ax, ylabel, title in [
            (ax1, "Energy per spin", "Convergence"),
            (ax2, "|E − E_exact| / spin", "Error"),
        ]:
            ax.set_xscale("log")
            ax.set_xlabel("# iterations", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=7, loc="best")

    # Hide unused cells in the last row
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        subfigs[row, col].set_visible(False)

    out_dir = FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / "cem_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {fname}")
    plt.close(fig)
    return [str(fname)]


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot CEM vs. no-CEM comparison")
    parser.add_argument(
        "--size", type=int, default=None, help="Filter by system size N"
    )
    parser.add_argument(
        "--h", type=float, default=None, help="Filter by transverse field h"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["1d", "2d"],
        help="Filter by model type",
    )
    args = parser.parse_args()

    print("Loading results...")
    paired = load_results(
        RESULTS_DIR,
        size_filter=args.size,
        h_filter=args.h,
        model_filter=args.model,
    )

    if not paired:
        print(
            "No paired CEM / no-CEM results found.\n"
            "Run training with and without --cem using the same hyperparameters first.\n"
            "Example:\n"
            "  python src/main.py --model 1d --size 8 --h 0.5\n"
            "  python src/main.py --model 1d --size 8 --h 0.5 --cem"
        )
        return

    print(
        f"\nFound {len(paired)} paired configuration(s) across "
        f"{len({(m, N, h) for m, N, h, *_ in paired})} (model, N, h) combo(s)."
    )

    saved = plot_cem_comparison(paired)
    print(f"\nGenerating summary overview...")
    saved_ov = plot_cem_summary(paired)
    print(
        f"\nDone — {len(saved)} comparison plot(s), {len(saved_ov)} summary page(s) saved."
    )


if __name__ == "__main__":
    main()

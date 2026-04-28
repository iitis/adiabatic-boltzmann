#!/usr/bin/env python3
"""
Aggregate jax_results into a per-run DataFrame and plot distance vs system size.

Convergence: rolling window of CONV_WINDOW iterations where the coefficient of
variation (std / |mean|) of energy-per-spin drops below CONV_THRESHOLD.
If never converged, the last window is used and converged=False.

Runs where exact_energy is None (large 2d systems) get distance_at_conv=NaN
and are excluded from the distance plot but included in the DataFrame.

Usage (from repo root):
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir jax_results --out aggregated_results.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONV_WINDOW = 20
CONV_THRESHOLD = 0.001  # 1% CV of energy-per-spin


def detect_convergence(energy_per_spin: np.ndarray) -> tuple[int, bool]:
    """Return (conv_iter, converged).

    First window where CV = std/|mean| < CONV_THRESHOLD counts as convergence.
    If no window qualifies, returns (total - W, False).
    """
    n = len(energy_per_spin)
    for t in range(n - CONV_WINDOW + 1):
        seg = energy_per_spin[t : t + CONV_WINDOW]
        mean = np.mean(seg)
        if mean != 0 and np.std(seg) / abs(mean) < CONV_THRESHOLD:
            return t, True
    return n - CONV_WINDOW, False


def parse_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]
    history = data["history"]

    model = cfg["model"]
    size = cfg["size"]
    N = size if model == "1d" else size**2

    energy_per_spin = np.array(history["energy"]) / N
    raw_sampling = history.get("sampling_time_s")
    sampling_time = np.array(raw_sampling) if raw_sampling is not None else None

    conv_iter, converged = detect_convergence(energy_per_spin)
    sl = slice(conv_iter, conv_iter + CONV_WINDOW)

    energy_at_conv = float(np.mean(energy_per_spin[sl]))
    avg_sampling = float(np.mean(sampling_time[sl])) if sampling_time is not None else None
    mean_sampling = float(np.mean(sampling_time)) if sampling_time is not None else None

    exact_energy = data.get("exact_energy")
    if exact_energy is not None:
        exact_eps = exact_energy / N
        distance_at_conv = abs(energy_at_conv - exact_eps)
        energy_final = float(np.mean(energy_per_spin[-CONV_WINDOW:]))
        distance_final = abs(energy_final - exact_eps)
    else:
        exact_eps = None
        distance_at_conv = None
        energy_final = None
        distance_final = None

    total_iters = cfg["iterations"]

    return {
        "model": model,
        "size": size,
        "N": N,
        "h": cfg["h"],
        "solver": cfg["sampling_method"],
        "sampler": cfg["sampler"],
        "cem": cfg.get("cem", False),
        "rbm": cfg.get("rbm", "full"),
        "n_hidden": cfg["n_hidden"],
        "lr": cfg["learning_rate"],
        "reg": cfg["regularization"],
        "seed": cfg["seed"],
        "sigma": cfg.get("sigma", 1.0),
        "n_samples": cfg["n_samples"],
        "total_iters": cfg["iterations"],
        "conv_iter": conv_iter,
        "converged": converged,
        "distance_at_conv": distance_at_conv,
        "exact_energy_per_spin": exact_eps,
        "energy_at_conv": energy_at_conv,
        "avg_sampling_time_s": avg_sampling,
        "mean_sampling_time_all_s": mean_sampling,
        "distance_final": distance_final,
        "final_kl_exact": data.get("final_kl_exact"),
        "final_ess": data.get("final_ess"),
    }


def load_dir(results_dir: Path, keep_solvers: set | None = None, skip_solvers: set | None = None) -> pd.DataFrame:
    rows = []
    skipped = 0
    for path in sorted(results_dir.rglob("*.json")):
        try:
            row = parse_file(path)
            if keep_solvers and row["solver"] not in keep_solvers:
                continue
            if skip_solvers and row["solver"] in skip_solvers:
                continue
            rows.append(row)
        except Exception as e:
            print(f"SKIP {path.name}: {e}")
            skipped += 1
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"Loaded {len(df)} runs ({skipped} skipped) from {results_dir}")
    return df


def load_all(jax_dir: Path, legacy_dir: Path) -> pd.DataFrame:
    dimod_solvers = {"pegasus", "zephyr"}
    df_jax = load_dir(jax_dir, skip_solvers=dimod_solvers)
    df_legacy = load_dir(legacy_dir, keep_solvers=dimod_solvers)
    return pd.concat([df_jax, df_legacy], ignore_index=True)


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df["solver"] == "lsb") & (~df["cem"]))]


def plot_metric(
    sub: pd.DataFrame,
    metric: str,
    ylabel: str,
    ax: plt.Axes,
    log: bool = True,
    linestyle: str = "-",
    marker: str = "o",
):
    grouped = sub.groupby(["solver", "N"])[metric].mean().reset_index()
    for solver, grp in grouped.groupby("solver"):
        grp = grp.sort_values("N")
        ax.plot(grp["N"], grp[metric], marker=marker, linestyle=linestyle, label=solver)
    ax.set_xlabel("System size N (spins)")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="jax_results", type=Path)
    parser.add_argument("--legacy-dir", default="results", type=Path)
    parser.add_argument("--out", default="aggregated_results.csv", type=Path)
    parser.add_argument("--plot", default="distance_vs_size.png", type=Path)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    results_dir = root / args.results_dir
    legacy_dir = root / args.legacy_dir
    out_csv = root / args.out
    out_plot = root / args.plot

    df = load_all(results_dir, legacy_dir)
    DIMOD_SAMPLING_TIME_PER_ITER = 0.3  # seconds, fixed override for pegasus/zephyr
    dimod_mask = df["sampler"] == "dimod"
    df.loc[dimod_mask, "mean_sampling_time_all_s"] = DIMOD_SAMPLING_TIME_PER_ITER
    df.loc[dimod_mask, "avg_sampling_time_s"] = DIMOD_SAMPLING_TIME_PER_ITER

    df["time_to_conv_s"] = df["conv_iter"] * df["mean_sampling_time_all_s"]
    df["time_total_s"] = df["total_iters"] * df["mean_sampling_time_all_s"]
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    print(f"\nRuns per solver/model:")
    print(df.groupby(["solver", "model"]).size().to_string())
    print(f"\nConvergence rate per solver:")
    print(df.groupby("solver")["converged"].mean().map("{:.1%}".format).to_string())

    base = df[(df["model"] == "1d") & df["distance_at_conv"].notna()]
    dimod_mask = base["solver"].isin(["pegasus", "zephyr"])
    df1d = filter_df(pd.concat([
        base[~dimod_mask & (base["lr"] == 0.01)],
        base[dimod_mask & (base["rbm"] == "full")],
    ], ignore_index=True))
    h_values = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(
        2, len(h_values), figsize=(5 * len(h_values), 9), sharey="row"
    )

    for col, h in enumerate(h_values):
        sub = df1d[df1d["h"] == h]

        ax_top = axes[0, col]
        plot_metric(
            sub,
            "distance_at_conv",
            r"$|e_\mathrm{VMC} - e_\mathrm{exact}|$",
            ax_top,
            log=True,
            linestyle="-",
            marker="o",
        )
        ax_top.set_title(f"Energy error  |  h = {h}")

        ax_bot = axes[1, col]
        plot_metric(
            sub[sub["conv_iter"] > 0],
            "time_to_conv_s",
            "Time to convergence (s)",
            ax_bot,
            log=True,
            linestyle="--",
            marker="s",
        )
        ax_bot.set_title(f"Sampling cost  |  h = {h}")

    for col in range(len(h_values)):
        axes[0, col].set_ylabel(
            r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at convergence"
        )
        axes[1, col].set_ylabel("Mean time to convergence (s)")
        for row in range(2):
            axes[row, col].tick_params(labelleft=True)

    fig.suptitle(
        f"1D TFIM — convergence quality and cost\n"
        f"(conv: CV < {CONV_THRESHOLD * 100:.1f}% over {CONV_WINDOW} iters, "
        f"mean over all runs per solver×N)"
    )
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"Saved → {out_plot}")

    # second plot: fixed 300 iters, no convergence measure
    out_plot2 = out_plot.parent / (out_plot.stem + "_300iters" + out_plot.suffix)
    base_final = df[(df["model"] == "1d") & df["distance_final"].notna()]
    dimod_mask_final = base_final["solver"].isin(["pegasus", "zephyr"])
    df1d_final = filter_df(pd.concat([
        base_final[~dimod_mask_final & (base_final["lr"] == 0.01)],
        base_final[dimod_mask_final & (base_final["rbm"] == "full")],
    ], ignore_index=True))
    fig2, axes2 = plt.subplots(
        2, len(h_values), figsize=(5 * len(h_values), 9), sharey="row"
    )
    for col, h in enumerate(h_values):
        sub = df1d_final[df1d_final["h"] == h]

        ax_top = axes2[0, col]
        plot_metric(sub, "distance_final",
                    r"$|e_\mathrm{VMC} - e_\mathrm{exact}|$", ax_top,
                    log=True, linestyle="-", marker="o")
        ax_top.set_title(f"Energy error  |  h = {h}")

        ax_bot = axes2[1, col]
        plot_metric(sub, "time_total_s", "Total sampling time (s)", ax_bot,
                    log=True, linestyle="--", marker="s")
        ax_bot.set_title(f"Total sampling cost  |  h = {h}")

    for col in range(len(h_values)):
        axes2[0, col].set_ylabel(r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at 300 iters")
        axes2[1, col].set_ylabel("Mean total sampling time (s)")
        for row in range(2):
            axes2[row, col].tick_params(labelleft=True)

    fig2.suptitle("1D TFIM — quality and cost at 300 iters (no convergence criterion)\n"
                  "mean over all runs per solver×N")
    fig2.tight_layout()
    fig2.savefig(out_plot2, dpi=150)
    print(f"Saved → {out_plot2}")


if __name__ == "__main__":
    main()

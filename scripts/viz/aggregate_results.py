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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import reference_energies

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
    # lr1d is a 1D chain — N = size, not size²
    N = size if model in ("1d", "lr1d") else size**2

    energy_per_spin = np.array(history["energy"]) / N
    raw_sampling = history.get("sampling_time_s")
    sampling_time = np.array(raw_sampling) if raw_sampling is not None else None

    conv_iter, converged = detect_convergence(energy_per_spin)
    sl = slice(conv_iter, conv_iter + CONV_WINDOW)

    energy_at_conv = float(np.mean(energy_per_spin[sl]))
    avg_sampling = float(np.mean(sampling_time[sl])) if sampling_time is not None else None
    mean_sampling = float(np.mean(sampling_time)) if sampling_time is not None else None

    # Reference energy: always read from master cache so LR-TFIM lookups use the
    # correct key (JSON-embedded values may be stale or absent).
    alpha = cfg.get("alpha")
    J = cfg.get("J", 1.0)
    if model == "lr1d" and alpha is not None:
        lr_key = f"lr_tfim_1d_alpha{float(alpha):.10g}_J{float(J):.10g}"
        exact_energy_total = reference_energies.lookup(lr_key, int(size), float(cfg["h"]))
    else:
        raw = data.get("exact_energy")
        exact_energy_total = float(raw) if raw is not None else None

    if exact_energy_total is not None:
        exact_eps = exact_energy_total / N
        distance_at_conv = abs(energy_at_conv - exact_eps)
        energy_final = float(np.mean(energy_per_spin[-CONV_WINDOW:]))
        distance_final = abs(energy_final - exact_eps)
    else:
        exact_eps = None
        distance_at_conv = None
        energy_final = None
        distance_final = None

    return {
        "model": model,
        "size": size,
        "N": N,
        "h": cfg["h"],
        "alpha": alpha,
        "J": J,
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


def plot_lr_tfim(df_lr: pd.DataFrame, out_path: Path) -> None:
    """2×4 grid: rows = (error, time), cols = α values, x-axis = h, lines = solver."""
    base = filter_df(df_lr[df_lr["distance_final"].notna()])
    alpha_values = sorted(base["alpha"].dropna().unique())
    if not alpha_values:
        print("No LR-TFIM data with exact reference energies — skipping lr_tfim plot.")
        return

    fig, axes = plt.subplots(
        2, len(alpha_values),
        figsize=(4.5 * len(alpha_values), 8),
        sharey="row", sharex="col",
    )
    if len(alpha_values) == 1:
        axes = axes.reshape(2, 1)

    for col, alpha in enumerate(alpha_values):
        sub = base[base["alpha"] == alpha].copy()

        grouped_err = sub.groupby(["solver", "h"])["distance_final"].mean().reset_index()
        for solver, grp in grouped_err.groupby("solver"):
            grp = grp.sort_values("h")
            axes[0, col].plot(grp["h"], grp["distance_final"], marker="o", label=solver)
        axes[0, col].set_title(f"α = {alpha:g}")
        axes[0, col].set_yscale("log")
        axes[0, col].grid(True, which="both", alpha=0.3)

        grouped_t = sub.groupby(["solver", "h"])["time_total_s"].mean().reset_index()
        for solver, grp in grouped_t.groupby("solver"):
            grp = grp.sort_values("h")
            axes[1, col].plot(grp["h"], grp["time_total_s"], marker="s", linestyle="--", label=solver)
        axes[1, col].set_yscale("log")
        axes[1, col].grid(True, which="both", alpha=0.3)
        axes[1, col].set_xlabel("Transverse field h")

    axes[0, 0].set_ylabel(r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at 300 iters")
    axes[1, 0].set_ylabel("Total sampling time (s)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", title="Solver", framealpha=0.9)

    fig.suptitle(
        f"LR-TFIM (N=16) — quality and cost by α  (last {CONV_WINDOW}-iter window, 300 iters)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


def plot_lr_vs_tfim(df_lr: pd.DataFrame, df_1d: pd.DataFrame, out_path: Path) -> None:
    """1×2 panels (error | time) comparing 1D TFIM and LR-TFIM at N=16, metropolis only."""
    solver = "metropolis"

    lr_base = filter_df(
        df_lr[(df_lr["distance_final"].notna()) & (df_lr["solver"] == solver)]
    )
    tfim_base = filter_df(
        df_1d[
            (df_1d["distance_final"].notna())
            & (df_1d["solver"] == solver)
            & (df_1d["N"] == 16)
            & (df_1d["lr"] == 0.01)
        ]
    )

    if lr_base.empty or tfim_base.empty:
        print("Insufficient data for LR vs TFIM comparison — skipping.")
        return

    fig, (ax_err, ax_t) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1D TFIM reference — bold black line
    tfim_grp = tfim_base.groupby("h")["distance_final"].mean().reset_index().sort_values("h")
    ax_err.plot(tfim_grp["h"], tfim_grp["distance_final"],
                color="black", linewidth=2.5, marker="o", label="TFIM  (α → ∞)", zorder=5)

    tfim_t = tfim_base.groupby("h")["time_total_s"].mean().reset_index().sort_values("h")
    ax_t.plot(tfim_t["h"], tfim_t["time_total_s"],
              color="black", linewidth=2.5, marker="o", linestyle="--", label="TFIM  (α → ∞)", zorder=5)

    # LR-TFIM — one coloured line per α, thinner
    cmap = plt.get_cmap("plasma")
    alpha_values = sorted(lr_base["alpha"].dropna().unique())
    colors = [cmap(i / max(len(alpha_values) - 1, 1)) for i in range(len(alpha_values))]

    for alpha, color in zip(alpha_values, colors):
        sub = lr_base[lr_base["alpha"] == alpha]
        err_grp = sub.groupby("h")["distance_final"].mean().reset_index().sort_values("h")
        ax_err.plot(err_grp["h"], err_grp["distance_final"],
                    color=color, marker="^", linewidth=1.5, label=f"LR-TFIM  α={alpha:g}")

        t_grp = sub.groupby("h")["time_total_s"].mean().reset_index().sort_values("h")
        ax_t.plot(t_grp["h"], t_grp["time_total_s"],
                  color=color, marker="^", linewidth=1.5, linestyle="--", label=f"LR-TFIM  α={alpha:g}")

    for ax, ylabel in [
        (ax_err, r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at 300 iters"),
        (ax_t,   "Total sampling time (s)"),
    ]:
        ax.set_yscale("log")
        ax.set_xlabel("Transverse field h")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    ax_err.set_title("Energy error at 300 iters")
    ax_t.set_title("Total sampling cost (300 iters)")

    handles, labels = ax_err.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Model", framealpha=0.9,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(
        f"LR-TFIM vs 1D TFIM — N=16, {solver}  (last {CONV_WINDOW}-iter window, 300 iters)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def plot_lr_tfim_vs_size(df_lr: pd.DataFrame, out_path: Path) -> None:
    """2×4 grid: rows = (error, time), cols = α values, x-axis = N, averaged over h."""
    base = filter_df(df_lr[df_lr["distance_final"].notna()])
    alpha_values = sorted(base["alpha"].dropna().unique())
    if not alpha_values:
        print("No LR-TFIM distance data — skipping lr_tfim_vs_size plot.")
        return

    fig, axes = plt.subplots(
        2, len(alpha_values),
        figsize=(4.5 * len(alpha_values), 8),
        sharey="row", sharex="col",
    )
    if len(alpha_values) == 1:
        axes = axes.reshape(2, 1)

    for col, alpha in enumerate(alpha_values):
        sub = base[base["alpha"] == alpha]

        grouped_err = sub.groupby(["solver", "N"])["distance_final"].mean().reset_index()
        for solver, grp in grouped_err.groupby("solver"):
            grp = grp.sort_values("N")
            axes[0, col].plot(grp["N"], grp["distance_final"], marker="o", label=solver)
        axes[0, col].set_title(f"α = {alpha:g}")
        axes[0, col].set_yscale("log")
        axes[0, col].set_xscale("log")
        axes[0, col].grid(True, which="both", alpha=0.3)

        grouped_t = sub.groupby(["solver", "N"])["time_total_s"].mean().reset_index()
        for solver, grp in grouped_t.groupby("solver"):
            grp = grp.sort_values("N")
            axes[1, col].plot(grp["N"], grp["time_total_s"], marker="s", linestyle="--", label=solver)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xscale("log")
        axes[1, col].grid(True, which="both", alpha=0.3)
        axes[1, col].set_xlabel("System size N (spins)")

    axes[0, 0].set_ylabel(r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at 300 iters")
    axes[1, 0].set_ylabel("Total sampling time (s)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", title="Solver", framealpha=0.9)

    fig.suptitle(
        f"LR-TFIM — quality and cost vs system size\n"
        f"(averaged over h;  only sizes with exact ED reference shown;\n"
        f"last {CONV_WINDOW}-iter window, 300 iters)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="jax_results", type=Path)
    parser.add_argument("--legacy-dir", default="jax_results", type=Path)
    parser.add_argument("--out", default="aggregated_results.csv", type=Path)
    parser.add_argument("--plot", default="distance_vs_size.png", type=Path)
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    results_dir = root / args.results_dir
    legacy_dir = root / args.legacy_dir
    out_csv = root / args.out
    out_plot = root / args.plot

    df = load_all(results_dir, legacy_dir)

    times_path = root / "scripts" / "dwave_sampling_times.json"
    if not times_path.exists():
        raise FileNotFoundError(
            f"D-Wave timing file not found: {times_path}\n"
            "Run:  python scripts/measure_dwave_times.py"
        )
    with times_path.open() as _f:
        _dwave_times: dict[str, dict[str, float]] = json.load(_f)

    def _lookup_dwave_time(row) -> float:
        t = _dwave_times.get(row["solver"], {}).get(str(int(row["N"])))
        if t is None:
            raise KeyError(
                f"No measured QPU time for solver={row['solver']} N={int(row['N'])}. "
                "Re-run scripts/measure_dwave_times.py."
            )
        return float(t)

    dimod_mask = df["sampler"] == "dimod"
    if dimod_mask.any():
        df.loc[dimod_mask, "mean_sampling_time_all_s"] = (
            df[dimod_mask].apply(_lookup_dwave_time, axis=1)
        )
        df.loc[dimod_mask, "avg_sampling_time_s"] = (
            df[dimod_mask].apply(_lookup_dwave_time, axis=1)
        )

    df["time_to_conv_s"] = df["conv_iter"] * df["mean_sampling_time_all_s"]
    df["time_total_s"] = df["total_iters"] * df["mean_sampling_time_all_s"]
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    print(f"\nRuns per solver/model:")
    print(df.groupby(["solver", "model"]).size().to_string())
    print(f"\nConvergence rate per solver:")
    print(df.groupby("solver")["converged"].mean().map("{:.1%}".format).to_string())

    # ── LR-TFIM plots ─────────────────────────────────────────────────────────
    df_lr = df[df["model"] == "lr1d"].copy()
    if not df_lr.empty:
        plot_lr_tfim(df_lr, out_plot.parent / "lr_tfim_distance_vs_h.png")
        plot_lr_tfim_vs_size(df_lr, out_plot.parent / "lr_tfim_distance_vs_size.png")
        base_1d_all = df[(df["model"] == "1d") & df["distance_final"].notna()]
        plot_lr_vs_tfim(df_lr, base_1d_all, out_plot.parent / "lr_vs_tfim_comparison.png")

    base = df[(df["model"] == "1d") & df["distance_final"].notna()]
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
            "distance_final",
            r"$|e_\mathrm{VMC} - e_\mathrm{exact}|$",
            ax_top,
            log=True,
            linestyle="-",
            marker="o",
        )
        ax_top.set_title(f"Energy error  |  h = {h}")

        ax_bot = axes[1, col]
        plot_metric(
            sub,
            "time_total_s",
            "Total sampling time (s)",
            ax_bot,
            log=True,
            linestyle="--",
            marker="s",
        )
        ax_bot.set_title(f"Sampling cost  |  h = {h}")

    for col in range(len(h_values)):
        axes[0, col].set_ylabel(
            r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at 300 iters"
        )
        axes[1, col].set_ylabel("Total sampling time (s)")
        for row in range(2):
            axes[row, col].tick_params(labelleft=True)

    fig.suptitle(
        f"1D TFIM — quality and cost  (last {CONV_WINDOW}-iter window, 300 iters,\n"
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

    # third plot: error and cost at the first converged window
    out_plot3 = out_plot.parent / (out_plot.stem + "_conv_window" + out_plot.suffix)
    base_conv = df[(df["model"] == "1d") & df["distance_at_conv"].notna()]
    dimod_mask_conv = base_conv["solver"].isin(["pegasus", "zephyr"])
    df1d_conv = filter_df(pd.concat([
        base_conv[~dimod_mask_conv & (base_conv["lr"] == 0.01)],
        base_conv[dimod_mask_conv & (base_conv["rbm"] == "full")],
    ], ignore_index=True))
    fig3, axes3 = plt.subplots(
        2, len(h_values), figsize=(5 * len(h_values), 9), sharey="row"
    )
    for col, h in enumerate(h_values):
        sub = df1d_conv[df1d_conv["h"] == h]

        ax_top = axes3[0, col]
        plot_metric(sub, "distance_at_conv",
                    r"$|e_\mathrm{VMC} - e_\mathrm{exact}|$", ax_top,
                    log=True, linestyle="-", marker="o")
        ax_top.set_title(f"Energy error at convergence  |  h = {h}")

        ax_bot = axes3[1, col]
        plot_metric(sub, "time_to_conv_s", "Time to convergence (s)", ax_bot,
                    log=True, linestyle="--", marker="s")
        ax_bot.set_title(f"Cost to convergence  |  h = {h}")

    for col in range(len(h_values)):
        axes3[0, col].set_ylabel(r"Mean $|e_\mathrm{VMC} - e_\mathrm{exact}|$ at conv. window")
        axes3[1, col].set_ylabel("Mean time to convergence (s)")
        for row in range(2):
            axes3[row, col].tick_params(labelleft=True)

    fig3.suptitle(
        f"1D TFIM — quality and cost at first converged window\n"
        f"(CV < {CONV_THRESHOLD:.1%} over {CONV_WINDOW} iters; mean over all runs per solver×N)"
    )
    fig3.tight_layout()
    fig3.savefig(out_plot3, dpi=150)
    print(f"Saved → {out_plot3}")


if __name__ == "__main__":
    main()

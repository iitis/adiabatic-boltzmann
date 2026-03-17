import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("results")


# ---------------------------
# 1. Load all result files
# ---------------------------
def load_results(root):
    records = []

    for file in root.rglob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)

        config = data["config"]
        history = data["history"]

        record = {
            # --- metadata ---
            "file": str(file),
            "size": config["size"],
            "h": config["h"],
            "rbm": config["rbm"],
            "n_hidden": config["n_hidden"],
            "sampler": config["sampler"],
            "sampling_method": config["sampling_method"],
            "lr": config["learning_rate"],
            "reg": config["regularization"],
            "n_samples": config["n_samples"],
            "seed": config["seed"],
            # --- results ---
            "final_energy": data["final_energy"],
            "exact_energy": data["exact_energy"],
            "error": data["error"],
            # --- full history ---
            "energy_curve": history["energy"],
            "error_curve": history["error"],
            "energy_error_curve": history.get("energy_error", []),
            "grad_norm_curve": history.get("grad_norm", []),
            "cond_curve": history.get("s_condition_number", []),
            "weight_norm_curve": history.get("weight_norm", []),
        }

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------
# 2. Aggregate statistics
# ---------------------------
def summarize(df):
    grouped = df.groupby(
        [
            "size",
            "h",
            "rbm",
            "n_hidden",
            "sampler",
            "sampling_method",
            "lr",
            "reg",
            "n_samples",
        ]
    )

    summary = grouped["error"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean")

    return summary


# ---------------------------
# 3. Plot: convergence curves
# ---------------------------
def plot_convergence(df, n=5):
    best = df.nsmallest(n, "error")

    plt.figure(figsize=(10, 5))

    for _, row in best.iterrows():
        label = f"h={row.h}, nh={row.n_hidden}, ns={row.n_samples}"
        plt.plot(row["energy_curve"], label=label)

        # exact energy line
        plt.axhline(row["exact_energy"], linestyle="--")

    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Top {n} runs (lowest error)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# 4. Plot: error vs hyperparams
# ---------------------------
def plot_error_vs_samples(df):
    plt.figure()

    for h in sorted(df["h"].unique()):
        subset = df[df["h"] == h]
        means = subset.groupby("n_samples")["error"].mean()

        plt.plot(means.index, means.values, marker="o", label=f"h={h}")

    plt.xlabel("n_samples")
    plt.ylabel("Mean Error")
    plt.title("Error vs Number of Samples")
    plt.legend()
    plt.show()


def plot_error_vs_regularization(df):
    plt.figure()

    means = df.groupby("reg")["error"].mean()

    plt.plot(means.index, means.values, marker="o")
    plt.xscale("log")

    plt.xlabel("regularization")
    plt.ylabel("Mean Error")
    plt.title("Error vs Regularization")
    plt.show()


# ---------------------------
# 5. Diagnostics (very useful)
# ---------------------------
def plot_diagnostics(df, idx=0):
    row = df.iloc[idx]

    plt.figure(figsize=(12, 6))

    # Gradient norm
    if row["grad_norm_curve"]:
        plt.subplot(2, 2, 1)
        plt.plot(row["grad_norm_curve"])
        plt.title("Gradient Norm")

    # Condition number
    if row["cond_curve"]:
        plt.subplot(2, 2, 2)
        plt.plot(row["cond_curve"])
        plt.yscale("log")
        plt.title("S Condition Number")

    # Weight norm
    if row["weight_norm_curve"]:
        plt.subplot(2, 2, 3)
        plt.plot(row["weight_norm_curve"])
        plt.title("Weight Norm")

    # Energy error
    if row["energy_error_curve"]:
        plt.subplot(2, 2, 4)
        plt.plot(row["energy_error_curve"])
        plt.title("Energy Std / sqrt(N)")

    plt.suptitle(f"Diagnostics: {row.file}")
    plt.tight_layout()
    plt.show()


# ---------------------------
# 6. Main
# ---------------------------
if __name__ == "__main__":
    df = load_results(ROOT)

    print(f"Loaded {len(df)} runs")

    summary = summarize(df)
    print("\nTop configurations:")
    print(summary.head(10))

    # Plots
    plot_convergence(df, n=5)
    plot_error_vs_samples(df)
    plot_error_vs_regularization(df)

    # Diagnostics for best run
    best_idx = df["error"].idxmin()
    plot_diagnostics(df, best_idx)

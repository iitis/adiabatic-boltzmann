#!/usr/bin/env python3
"""
Train a real-valued RBM with Gibbs sampling across J2/J1 values on the
J1-J2 Heisenberg chain, then plot the energy error vs J2/J1 alongside
the sign impurity from exact diagonalization.

The plot directly connects the fundamental representational limitation
(sign_imp = 0 for J2/J1 <= 0.5, nonzero above) to the practical training
error of a real RBM.

Usage:
    python scripts/gibbs_j1j2_sweep.py
    python scripts/gibbs_j1j2_sweep.py --size 12 --seeds 5 --n-samples 800
"""

import argparse
import gzip
import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC  = _ROOT / "src"
_VIZ  = _ROOT / "scripts" / "viz"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_VIZ))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from model   import FullyConnectedRBM
from ising   import J1J2HeisenbergXXZ1D
from sampler import ClassicalSampler
from encoder import Trainer
from plot_style import setup_style

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",      type=int,   default=8)
    p.add_argument("--nh-alpha",  type=int,   default=3,
                   help="n_hidden = nh_alpha * n_visible")
    p.add_argument("--iterations",type=int,   default=300)
    p.add_argument("--n-samples", type=int,   default=800)
    p.add_argument("--sampler",   type=str,   default="gibbs",
                   choices=["metropolis", "gibbs", "exchange"],
                   help="Sampling method: gibbs (default), metropolis, or exchange")
    p.add_argument("--n-sweeps",  type=int,   default=10,
                   help="Sweeps per training iteration (metropolis/gibbs/exchange)")
    p.add_argument("--lr",        type=float, default=0.015)
    p.add_argument("--reg",       type=float, default=1.7e-4)
    p.add_argument("--seeds",     type=int,   default=5)
    p.add_argument("--j2-steps",  type=int,   default=11,
                   help="Number of J2/J1 values (linearly spaced 0..j2-max)")
    p.add_argument("--j2-max",   type=float, default=1.0,
                   help="Maximum J2/J1 value (default 1.0)")
    p.add_argument("--retrain",   action="store_true",
                   help="Ignore cached results and retrain")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_run(N: int, n_hidden: int, J2_J1: float, seed: int, cfg: dict) -> dict:
    """Train one RBM, return dict with energy history and exact energy."""
    J1, J2 = 1.0, J2_J1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ising = J1J2HeisenbergXXZ1D(N, J1=J1, J2=J2, delta=1.0)

    key = jax.random.PRNGKey(seed)
    rbm = FullyConnectedRBM(N, n_hidden, key)
    sampler = ClassicalSampler(cfg["sampler"], n_sweeps=cfg["n_sweeps"])

    trainer = Trainer(rbm, ising, sampler, config={
        "learning_rate":  cfg["lr"],
        "n_iterations":   cfg["iterations"],
        "n_samples":      cfg["n_samples"],
        "regularization": cfg["reg"],
        "seed":           seed,
    }, args=None)

    history = trainer.train()

    exact = ising.exact_ground_energy()
    return {
        "J2_J1":        J2_J1,
        "seed":         seed,
        "final_energy": float(history["energy"][-1]),
        "exact_energy": float(exact) if exact is not None else None,
        "energies":     [float(e) for e in history["energy"]],
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args) -> dict:
    """Run the full J2/J1 sweep, returning collected results."""
    N        = args.size
    n_hidden = args.nh_alpha * N
    ratios   = np.linspace(0.0, args.j2_max, args.j2_steps)
    seeds    = list(range(args.seeds))
    cfg      = dict(lr=args.lr, reg=args.reg, iterations=args.iterations,
                    n_samples=args.n_samples, n_sweeps=args.n_sweeps,
                    sampler=args.sampler)

    out_dir = _ROOT / "scripts" / "output" / "gibbs_j1j2_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    j2max_tag  = f"_j2max{args.j2_max:.1f}".replace(".", "p")
    cache_path = out_dir / f"j1j2_{args.sampler}_sweep_N{N}_nh{n_hidden}_iter{args.iterations}_ns{args.n_samples}{j2max_tag}.json"

    if cache_path.exists() and not args.retrain:
        print(f"Loading cached results from {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)

    all_runs = []
    n_total  = len(ratios) * len(seeds)
    done     = 0
    for r in ratios:
        for s in seeds:
            done += 1
            print(f"\n[{done}/{n_total}] J2/J1={r:.2f}  seed={s}", flush=True)
            run = train_run(N, n_hidden, float(r), s, cfg)
            all_runs.append(run)

    results = {
        "N":          N,
        "n_hidden":   n_hidden,
        "iterations": args.iterations,
        "n_samples":  args.n_samples,
        "lr":         args.lr,
        "reg":        args.reg,
        "sampler":    args.sampler,
        "n_sweeps":   args.n_sweeps,
        "runs":       all_runs,
    }
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {cache_path}")
    return results


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(results: dict):
    """Return (ratios, mean_err, std_err) where err = (E_RBM-E_exact)/N."""
    N    = results["N"]
    runs = results["runs"]

    by_ratio = {}
    for r in runs:
        key = round(r["J2_J1"], 6)
        by_ratio.setdefault(key, []).append(r)

    ratios, means, stds = [], [], []
    for ratio in sorted(by_ratio):
        group = by_ratio[ratio]
        errs  = [(g["final_energy"] - g["exact_energy"]) / N
                 for g in group
                 if g["exact_energy"] is not None
                 and g["final_energy"] is not None
                 and np.isfinite(g["final_energy"])]
        if errs:
            ratios.append(ratio)
            means.append(float(np.mean(errs)))
            stds.append(float(np.std(errs)))

    return np.array(ratios), np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Sign impurity from existing ED data
# ---------------------------------------------------------------------------

def load_sign_imp(N: int):
    """Load sign_imp vs J2/J1 for the given N from the cached ED sweep."""
    path = _ROOT / "plots" / "j1j2" / "j1j2_sign_problem.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    key = f"N{N}"
    if key not in data:
        # Fall back to closest available N
        available = [int(k[1:]) for k in data if k.startswith("N")]
        closest   = min(available, key=lambda n: abs(n - N))
        key       = f"N{closest}"
        print(f"  [sign_imp] N={N} not in ED cache; using N={closest}")
    d = data[key]
    return np.array(d["J2_J1"]), np.array(d["sign_imp"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results: dict, out_dir: Path) -> None:
    setup_style()

    N           = results["N"]
    ratios, means, stds = aggregate(results)
    si_x, si_y         = load_sign_imp(N)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    # --- top: energy error ---
    ax = axes[0]
    sampler_name = results.get("sampler", "metropolis")
    ax.plot(ratios, means, color="#2563eb", lw=1.8, marker="o",
            markersize=4, markerfacecolor="white", markeredgewidth=1.2,
            label=f"$N={N}$, $M={results['n_hidden']}$, {results['iterations']} iter ({sampler_name})")
    ax.fill_between(ratios, np.array(means) - np.array(stds),
                                  np.array(means) + np.array(stds),
                    alpha=0.20, color="#2563eb")
    ax.axvline(0.5, color="#222", ls=":", lw=1.2, zorder=0)
    ax.set_ylabel(
        r"Energy error / site" + "\n"
        r"$(E_{\mathrm{RBM}}-E_{\mathrm{exact}})/N$")
    ax.set_ylim(bottom=-0.002)
    ax.legend(fontsize=9, loc="upper left")
    # Annotate at the first data point clearly above the pre-MG baseline
    ratios_arr = np.array(ratios)
    means_arr  = np.array(means)
    above_mg   = ratios_arr > 0.5
    y_tip = float(means_arr[above_mg][0]) if above_mg.any() else float(max(means_arr))
    x_tip = float(ratios_arr[above_mg][0]) if above_mg.any() else 0.5
    y_max = float(max(means_arr))
    ax.annotate(
        "Majumdar--Ghosh\n" + r"($J_2/J_1=0.5$)",
        xy=(x_tip, y_tip * 0.5),
        xytext=(max(x_tip + 0.08, 0.65), y_max * 0.65),
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.9),
        fontsize=9, color="#333",
    )

    # --- bottom: sign impurity from ED ---
    ax = axes[1]
    if si_x is not None:
        ax.plot(si_x, si_y, color="#dc2626", lw=1.8,
                label=r"Sign impurity (exact, ED)")
    ax.axvline(0.5, color="#222", ls=":", lw=1.2, zorder=0)
    ax.set_ylabel(
        r"Sign impurity $\displaystyle\sum_{\phi<0}\phi(v)^2$")
    ax.set_ylim(-0.01, 0.55)
    ax.set_xlabel(r"$J_2/J_1$")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    sampler_name = results.get("sampler", "metropolis")
    for ext in ("pdf", "png"):
        path = out_dir / f"j1j2_{sampler_name}_energy_error.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Saved → {path}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    results = run_sweep(args)

    out_dir = _ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    plot(results, out_dir)


if __name__ == "__main__":
    main()

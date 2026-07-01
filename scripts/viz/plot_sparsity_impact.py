#!/usr/bin/env python3
"""
plot_sparsity_impact.py

Assesses the impact of Pegasus / Zephyr QPU topology sparsity on RBM-VMC
ground-state accuracy for the 1D Transverse Field Ising Model.

Five plots
----------
1. Energy error vs. hidden-unit ratio α  (line + error bars, fixed h=h_c)
2. Learning curves  (line + shaded ±σ bands, Dense / Pegasus / Zephyr)
3. Error heatmap in (α, h) space  (3-panel: Dense | Pegasus | Zephyr)
4. Scaling with system size N  (line + error bars, fixed α and h=h_c)
5. Parameter efficiency  (scatter: n_params vs. final energy error)

Usage (from repo root)
----------------------
    python scripts/viz/plot_sparsity_impact.py              # quick (CPU-friendly)
    python scripts/viz/plot_sparsity_impact.py --full       # GPU-scale grid
    python scripts/viz/plot_sparsity_impact.py --plot-only  # replot from cache
    python scripts/viz/plot_sparsity_impact.py --full --plot-only
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax

from plot_style import setup_style
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler
from encoder import Trainer

# ── Topology styling ──────────────────────────────────────────────────────────

TOPOLOGIES  = ["dense", "pegasus", "zephyr"]
TOPO_LABEL  = {"dense": "Dense",   "pegasus": "Pegasus",  "zephyr": "Zephyr"}
TOPO_COLOR  = {"dense": "#2166ac", "pegasus": "#d62728",  "zephyr": "#2ca02c"}
TOPO_LS     = {"dense": "-",       "pegasus": "--",       "zephyr": "-."}
TOPO_MARKER = {"dense": "o",       "pegasus": "s",        "zephyr": "^"}

# ── Experiment grids ──────────────────────────────────────────────────────────

QUICK = dict(
    sizes       = [8, 16],
    alphas      = [1, 2, 3],
    h_sweep     = [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0],
    h_crit      = 1.0,
    alpha_scale = 2,
    n_heatmap   = 8,
    lc_n        = 8,
    lc_alpha    = 2,
    seeds       = [42, 123],
    n_iters     = 100,
    n_samples   = 200,
    lr          = 0.05,
    reg         = 1e-3,
)

FULL = dict(
    sizes       = [8, 12, 16, 24, 32, 48, 64],
    alphas      = [1, 2, 3, 4, 5],
    # 14 linspace values + h_c=1.0 explicitly → 15 distinct points
    h_sweep     = sorted(set([round(h, 6) for h in np.linspace(0.3, 2.0, 14)] + [1.0])),
    h_crit      = 1.0,
    alpha_scale = 2,
    n_heatmap   = 16,
    lc_n        = 32,
    lc_alpha    = 3,
    seeds       = [42, 123, 456, 789, 1234],
    n_iters     = 300,
    n_samples   = 500,
    lr          = 0.05,
    reg         = 1e-3,
)

# ── Cache helpers ─────────────────────────────────────────────────────────────


def _key(N, alpha, h, topology, seed):
    return f"{N}_{alpha}_{h:.8g}_{topology}_{seed}"


def _build_experiment_list(cfg):
    exps = set()
    # Heatmap: fixed n_heatmap, all (alpha, h_sweep)
    for alpha in cfg["alphas"]:
        for h in cfg["h_sweep"]:
            for topo in TOPOLOGIES:
                for seed in cfg["seeds"]:
                    exps.add((cfg["n_heatmap"], alpha, h, topo, seed))
    # Scaling: all sizes, fixed (alpha_scale, h_crit)
    for N in cfg["sizes"]:
        for topo in TOPOLOGIES:
            for seed in cfg["seeds"]:
                exps.add((N, cfg["alpha_scale"], cfg["h_crit"], topo, seed))
    # α-sweep: n_heatmap, all alpha, h_crit (usually already in heatmap set)
    for alpha in cfg["alphas"]:
        for topo in TOPOLOGIES:
            for seed in cfg["seeds"]:
                exps.add((cfg["n_heatmap"], alpha, cfg["h_crit"], topo, seed))
    # Learning curves: lc_n and lc_alpha may not be covered by the other sets
    for topo in TOPOLOGIES:
        for seed in cfg["seeds"]:
            exps.add((cfg["lc_n"], cfg["lc_alpha"], cfg["h_crit"], topo, seed))
    return sorted(exps)


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_cache(cache: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


# ── Single experiment ─────────────────────────────────────────────────────────


def _make_rbm(topology, N, alpha, seed):
    key = jax.random.PRNGKey(seed)
    n_hidden = N * alpha
    if topology == "dense":
        return FullyConnectedRBM(N, n_hidden, key)
    elif topology in ("pegasus", "zephyr"):
        return DWaveTopologyRBM(N, n_hidden, key, solver=topology, seed=42)
    else:
        raise ValueError(f"Unknown topology: {topology}")


def _run_one(N, alpha, h, topology, seed, cfg):
    import numpy as np
    np.random.seed(seed)

    ising   = TransverseFieldIsing1D(size=N, h=h)
    rbm     = _make_rbm(topology, N, alpha, seed)
    sampler = ClassicalSampler(method="metropolis")
    config  = {
        "n_samples":          cfg["n_samples"],
        "n_iterations":       cfg["n_iters"],
        "learning_rate":      cfg["lr"],
        "regularization":     cfg["reg"],
        "stop_at_convergence": False,
        "save_checkpoints":   False,
    }
    trainer = Trainer(rbm=rbm, ising_model=ising, sampler=sampler, config=config)
    history = trainer.train()

    energies = np.array(history["energy"])
    E_exact  = ising.exact_ground_energy()
    tail     = max(1, len(energies) // 5)
    E_final  = float(np.nanmean(energies[-tail:]))
    rel_err  = abs(E_final - E_exact) / abs(E_exact)

    return {
        "energy_history": [float(e) for e in energies],
        "E_exact":        float(E_exact),
        "E_final":        E_final,
        "rel_error":      rel_err,
        "n_params":       rbm.n_parameters(),
        "sparsity":       rbm.sparsity(),
    }


def run_experiments(cfg, cache_path: Path) -> dict:
    cache = load_cache(cache_path)
    exps  = _build_experiment_list(cfg)
    total = len(exps)
    done  = sum(1 for e in exps if _key(*e) in cache)
    print(f"  {done}/{total} experiments already in cache — running {total-done} new ones")

    for i, (N, alpha, h, topology, seed) in enumerate(exps):
        k = _key(N, alpha, h, topology, seed)
        if k in cache:
            continue
        print(f"  [{i+1}/{total}] N={N} α={alpha} h={h:.3f} {topology} seed={seed}")
        cache[k] = _run_one(N, alpha, h, topology, seed, cfg)
        save_cache(cache, cache_path)

    return cache


# ── Data access helpers ───────────────────────────────────────────────────────


def _collect(cache, N, alpha, h, topology, seeds, field="rel_error"):
    vals = []
    for seed in seeds:
        rec = cache.get(_key(N, alpha, h, topology, seed))
        if rec is not None:
            vals.append(rec[field])
    return np.array(vals)


def _ms(arr):
    """Return (mean, std) of a 1-D array, ignoring NaN."""
    a = np.array(arr, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


# ── Plot 1 — Energy error vs. α ───────────────────────────────────────────────


def plot_alpha(cache, cfg, out: Path):
    N    = cfg["n_heatmap"]
    h    = cfg["h_crit"]
    seeds = cfg["seeds"]

    setup_style(fontsize=10, scale=1.8)
    fig, ax = plt.subplots()

    for topo in TOPOLOGIES:
        means, stds = [], []
        for alpha in cfg["alphas"]:
            errs = _collect(cache, N, alpha, h, topo, seeds)
            m, s = _ms(errs)
            means.append(m)
            stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.errorbar(
            cfg["alphas"], means, yerr=stds,
            label=TOPO_LABEL[topo],
            color=TOPO_COLOR[topo],
            linestyle=TOPO_LS[topo],
            marker=TOPO_MARKER[topo],
            markersize=5, capsize=3, linewidth=1.4,
        )

    ax.set_xlabel(r"Hidden-unit ratio $\alpha = n_h / N$")
    ax.set_ylabel(r"Relative energy error $|\varepsilon|$")
    ax.set_yscale("log")
    ax.set_xticks(cfg["alphas"])
    ax.set_title(rf"$N = {N}$,  $h = {h}$ (critical)")
    ax.legend()

    _savefig(fig, out, "sparsity_alpha")


# ── Plot 2 — Learning curves ──────────────────────────────────────────────────


def plot_learning_curves(cache, cfg, out: Path):
    N     = cfg["lc_n"]
    alpha = cfg["lc_alpha"]
    h     = cfg["h_crit"]
    seeds = cfg["seeds"]

    setup_style(fontsize=10, scale=1.8)
    fig, ax = plt.subplots()

    for topo in TOPOLOGIES:
        histories = []
        E_exact = None
        for seed in seeds:
            rec = cache.get(_key(N, alpha, h, topo, seed))
            if rec is None:
                continue
            histories.append(rec["energy_history"])
            E_exact = rec["E_exact"]

        if not histories or E_exact is None:
            continue

        n_iters = min(len(h_) for h_ in histories)
        arr = np.array([h_[:n_iters] for h_ in histories])
        rel = np.abs((arr - E_exact) / abs(E_exact))

        iters = np.arange(1, n_iters + 1)
        mean  = np.nanmean(rel, axis=0)
        std   = np.nanstd(rel, axis=0)

        ax.plot(iters, mean,
                label=TOPO_LABEL[topo],
                color=TOPO_COLOR[topo],
                linestyle=TOPO_LS[topo],
                linewidth=1.4)
        ax.fill_between(iters, mean - std, mean + std,
                        color=TOPO_COLOR[topo], alpha=0.18)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Relative energy error $|\varepsilon|$")
    ax.set_yscale("log")
    ax.set_title(rf"$N = {N}$,  $\alpha = {alpha}$,  $h = {h}$")
    ax.legend()

    _savefig(fig, out, "sparsity_learning_curves")


# ── Plot 3 — Heatmap (α, h) for each topology ────────────────────────────────


def plot_heatmap(cache, cfg, out: Path):
    N       = cfg["n_heatmap"]
    alphas  = cfg["alphas"]
    h_sweep = cfg["h_sweep"]
    seeds   = cfg["seeds"]

    # Build error matrices (n_alphas × n_h) for each topology
    grids = {}
    for topo in TOPOLOGIES:
        grid = np.full((len(alphas), len(h_sweep)), np.nan)
        for ia, alpha in enumerate(alphas):
            for ih, h in enumerate(h_sweep):
                errs = _collect(cache, N, alpha, h, topo, seeds)
                if len(errs):
                    grid[ia, ih] = np.nanmean(errs)
        grids[topo] = grid

    # Shared colour scale (log)
    all_vals = np.concatenate([g[~np.isnan(g)].ravel() for g in grids.values()])
    if len(all_vals) == 0:
        print("  [heatmap] no data — skipping")
        return
    vmin = max(1e-5, float(np.nanmin(all_vals)))
    vmax = float(np.nanmax(all_vals))

    setup_style(fontsize=9, scale=1.0)
    fig = plt.figure(figsize=(12, 3.5))
    gs  = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.07], wspace=0.12)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    cax  = fig.add_subplot(gs[0, 3])

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = "RdYlGn_r"

    h_ticks = [f"{v:.2f}" for v in h_sweep]
    a_ticks = [str(a) for a in alphas]

    for col, topo in enumerate(TOPOLOGIES):
        ax = axes[col]
        im = ax.imshow(
            grids[topo], origin="lower", aspect="auto",
            norm=norm, cmap=cmap,
            extent=[-0.5, len(h_sweep) - 0.5, -0.5, len(alphas) - 0.5],
        )
        ax.set_title(TOPO_LABEL[topo], fontsize=10)
        ax.set_xlabel(r"Transverse field $h$")
        ax.set_xticks(np.arange(len(h_sweep)))
        ax.set_xticklabels(h_ticks, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(alphas)))
        if col == 0:
            ax.set_yticklabels(a_ticks)
            ax.set_ylabel(r"$\alpha = n_h / N$")
        else:
            ax.set_yticklabels([])
        ax.grid(False)
        # Mark critical point
        h_c_idx = min(range(len(h_sweep)), key=lambda i: abs(h_sweep[i] - 1.0))
        ax.axvline(h_c_idx, color="white", linewidth=0.8, linestyle=":")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Mean rel. error $|\varepsilon|$", fontsize=9)

    fig.suptitle(rf"Energy error vs. $(\alpha, h)$ — $N={N}$", y=1.02, fontsize=10)
    _savefig(fig, out, "sparsity_heatmap")


# ── Plot 4 — Scaling with N ───────────────────────────────────────────────────


def plot_scaling(cache, cfg, out: Path):
    alpha = cfg["alpha_scale"]
    h     = cfg["h_crit"]
    seeds = cfg["seeds"]
    sizes = cfg["sizes"]

    setup_style(fontsize=10, scale=1.8)
    fig, ax = plt.subplots()

    for topo in TOPOLOGIES:
        means, stds, ns_ok = [], [], []
        for N in sizes:
            errs = _collect(cache, N, alpha, h, topo, seeds)
            if len(errs) == 0:
                continue
            m, s = _ms(errs)
            means.append(m)
            stds.append(s)
            ns_ok.append(N)
        if not ns_ok:
            continue
        means, stds = np.array(means), np.array(stds)
        ax.errorbar(
            ns_ok, means, yerr=stds,
            label=TOPO_LABEL[topo],
            color=TOPO_COLOR[topo],
            linestyle=TOPO_LS[topo],
            marker=TOPO_MARKER[topo],
            markersize=5, capsize=3, linewidth=1.4,
        )

    ax.set_xlabel(r"System size $N$")
    ax.set_ylabel(r"Relative energy error $|\varepsilon|$")
    ax.set_yscale("log")
    ax.set_title(rf"$\alpha = {alpha}$,  $h = {h}$ (critical)")
    ax.legend()

    _savefig(fig, out, "sparsity_scaling")


# ── Plot 5 — Parameter efficiency scatter ─────────────────────────────────────


def plot_param_efficiency(cache, cfg, out: Path):
    h     = cfg["h_crit"]
    seeds = cfg["seeds"]

    setup_style(fontsize=10, scale=1.8)
    fig, ax = plt.subplots()

    for topo in TOPOLOGIES:
        n_params_list, err_means, err_stds = [], [], []
        for N in cfg["sizes"]:
            for alpha in cfg["alphas"]:
                errs = _collect(cache, N, alpha, h, topo, seeds)
                if len(errs) == 0:
                    continue
                # n_params: take from first available seed
                n_params = None
                for seed in seeds:
                    rec = cache.get(_key(N, alpha, h, topo, seed))
                    if rec is not None:
                        n_params = rec["n_params"]
                        break
                if n_params is None:
                    continue
                m, s = _ms(errs)
                n_params_list.append(n_params)
                err_means.append(m)
                err_stds.append(s)

        if not n_params_list:
            continue
        order = np.argsort(n_params_list)
        x = np.array(n_params_list)[order]
        y = np.array(err_means)[order]
        ye = np.array(err_stds)[order]

        ax.errorbar(
            x, y, yerr=ye,
            label=TOPO_LABEL[topo],
            color=TOPO_COLOR[topo],
            linestyle="none",
            marker=TOPO_MARKER[topo],
            markersize=5, capsize=2, alpha=0.85,
        )
        # Trend line through same topology points
        ax.plot(x, y, color=TOPO_COLOR[topo], linestyle=TOPO_LS[topo],
                linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Free parameters")
    ax.set_ylabel(r"Relative energy error $|\varepsilon|$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(rf"$h = {h}$ (critical)")
    ax.legend()

    _savefig(fig, out, "sparsity_param_efficiency")


# ── Save helper ───────────────────────────────────────────────────────────────


def _savefig(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved  {p}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--full", action="store_true",
        help="Use GPU-scale grid (many N, α, seeds, fine h-sweep). "
             "Default is a quick CPU-friendly grid."
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="Skip experiments — load from cache and regenerate plots only."
    )
    p.add_argument(
        "--cache", type=str, default=None,
        help="Path to JSON cache file. Defaults to plots/sparsity/cache_{quick,full}.json"
    )
    p.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for plots. Defaults to plots/sparsity/"
    )
    p.add_argument(
        "--iters", type=int, default=None,
        help="Override number of training iterations."
    )
    p.add_argument(
        "--samples", type=int, default=None,
        help="Override number of samples per iteration."
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate."
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = dict(FULL if args.full else QUICK)

    if args.iters   is not None: cfg["n_iters"]   = args.iters
    if args.samples is not None: cfg["n_samples"]  = args.samples
    if args.lr      is not None: cfg["lr"]         = args.lr

    label      = "full" if args.full else "quick"
    out        = Path(args.out_dir)   if args.out_dir else _REPO / "plots" / "sparsity"
    cache_path = Path(args.cache)     if args.cache   else out / f"cache_{label}.json"

    if args.plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache at {cache_path} — run without --plot-only first."
            )
        print(f"  loading cache  {cache_path}")
        cache = load_cache(cache_path)
    else:
        cache = run_experiments(cfg, cache_path)

    print("\n── Generating plots ──────────────────────────────────")
    plot_alpha(cache, cfg, out)
    plot_learning_curves(cache, cfg, out)
    plot_heatmap(cache, cfg, out)
    plot_scaling(cache, cfg, out)
    plot_param_efficiency(cache, cfg, out)
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Marshall trick comparison — train and plot.

Trains RBM with and without the Marshall sign correction across J2/J1 in [0,1],
caches results to scripts/output/marshall_comparison/, then produces a two-panel
plot in the style of gibbs_j1j2_sweep:

  Top    — Energy error / site (E_RBM - E_exact) / N, mean ± std over seeds.
  Bottom — Sign impurity of the exact GS in two gauges:
             Marshall basis  φ = s·Ψ   (what "with Marshall" sees)
             Raw basis       φ = Ψ     (what "without Marshall" sees)

Usage:
    python scripts/viz/plot_marshall_comparison.py
    python scripts/viz/plot_marshall_comparison.py --retrain
    python scripts/viz/plot_marshall_comparison.py --size 8 --seeds 3 --iters 200
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC  = _ROOT / "src"
_VIZ  = Path(__file__).resolve().parent
_OUT  = _ROOT / "scripts" / "output" / "marshall_comparison"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(str(_VIZ)))

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

from model   import FullyConnectedRBM
from ising   import J1J2HeisenbergXXZ1D
from sampler import ClassicalSampler
from encoder import Trainer
from plot_style import setup_style


# ---------------------------------------------------------------------------
# Sign impurity from exact diagonalisation
# ---------------------------------------------------------------------------

def _coupling_matrix(N, bond_len):
    dim = 2 ** N
    k   = np.arange(dim)
    rows, cols, vals, diag = [], [], [], np.zeros(dim)
    for i in range(N):
        j  = (i + bond_len) % N
        si = 1 - 2 * ((k >> i) & 1)
        sj = 1 - 2 * ((k >> j) & 1)
        diag += 0.25 * si * sj
        opp   = si != sj
        k_src = k[opp]; k_dst = k_src ^ (1 << i) ^ (1 << j)
        rows.append(k_dst); cols.append(k_src)
        vals.append(np.full(opp.sum(), 0.5))
    rows = np.concatenate(rows + [k])
    cols = np.concatenate(cols + [k])
    vals = np.concatenate(vals + [diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def _marshall_signs(N):
    k = np.arange(2 ** N)
    s = np.ones(2 ** N)
    for i in range(1, N, 2):
        s *= 1 - 2 * ((k >> i) & 1)
    return s


def _average_sign(phi):
    """⟨s⟩ = Σ_v φ(v) / Σ_v |φ(v)|  (Troyer & Wiese, cond-mat/0408370)."""
    return float(np.sum(phi)) / float(np.sum(np.abs(phi)))


def compute_average_signs(N, ratios):
    H_nn, H_nnn = _coupling_matrix(N, 1), _coupling_matrix(N, 2)
    s           = _marshall_signs(N)
    as_raw, as_marshall = [], []
    for r in ratios:
        H = 1.0 * H_nn + float(r) * H_nnn
        evals, evecs = eigsh(H, k=2, which="SA", tol=1e-12, maxiter=200_000)
        psi0 = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
        psi1 = evecs[:, 1] / np.linalg.norm(evecs[:, 1])
        # Near-degenerate (MG point): search over all linear combinations to
        # find the one with maximum ⟨s⟩_marshall (= minimum sign impurity).
        # For a non-degenerate GS, all theta give the same |⟨s⟩| up to global sign.
        degenerate = abs(evals[1] - evals[0]) < 1e-6 * abs(evals[0]) + 1e-10
        if degenerate:
            best_as, best_psi = -2.0, psi0
            for theta in np.linspace(0, np.pi, 600):
                p = np.cos(theta) * psi0 + np.sin(theta) * psi1
                p /= np.linalg.norm(p)
                phi_m = s * p
                cur_as = _average_sign(phi_m)
                if cur_as > best_as:
                    best_as, best_psi = cur_as, p.copy()
            psi = best_psi
        else:
            psi = psi0
        # Canonical sign: maximise ⟨s⟩ in Marshall basis (positive for J2<0.5)
        phi_m = s * psi
        if _average_sign(phi_m) < 0:
            psi   = -psi
            phi_m = -phi_m
        as_raw.append(_average_sign(psi))
        as_marshall.append(_average_sign(phi_m))
    return np.array(as_raw), np.array(as_marshall)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_one(N, n_hidden, J2, seed, lr, reg, iters, n_samples, use_marshall):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ising = J1J2HeisenbergXXZ1D(N, J1=1.0, J2=J2, delta=1.0)
    if not use_marshall:
        ising.off_sign = 1

    key = jax.random.PRNGKey(seed)
    rbm = FullyConnectedRBM(N, n_hidden, key)
    sampler = ClassicalSampler("exchange", n_sweeps=8)
    history = Trainer(rbm, ising, sampler, config={
        "learning_rate":  lr,
        "n_iterations":   iters,
        "n_samples":      n_samples,
        "regularization": reg,
        "seed":           seed,
    }, args=None).train()
    return float(history["energy"][-1])


def run_sweep(args):
    N        = args.size
    n_hidden = args.alpha * N
    ratios   = np.linspace(0.0, 1.0, args.steps)
    seeds    = list(range(args.seeds))

    cache_path = _OUT / (
        f"marshall_comparison_N{N}_nh{n_hidden}"
        f"_iter{args.iters}_ns{args.samples}_seeds{args.seeds}.json"
    )
    _OUT.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not args.retrain:
        print(f"Loading cached results from {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)

    runs = []
    for use_marshall in [True, False]:
        mode = "marshall" if use_marshall else "no_marshall"
        print(f"\nTraining {mode} (N={N}, nh={n_hidden}, iters={args.iters}) …")
        for r in ratios:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ising_ref = J1J2HeisenbergXXZ1D(N, J1=1.0, J2=float(r), delta=1.0)
            E_exact = float(ising_ref.exact_ground_energy())
            for seed in seeds:
                E = _train_one(N, n_hidden, float(r), seed,
                               args.lr, args.reg, args.iters, args.samples, use_marshall)
                err = abs(E - E_exact) / abs(E_exact)
                print(f"  [{mode}] J2/J1={r:.2f} seed={seed}  E={E:.5f}  err={err:.4e}")
                runs.append({"mode": mode, "J2_J1": float(r), "seed": seed,
                             "E_RBM": E, "E_exact": E_exact})

    results = {"N": N, "n_hidden": n_hidden, "iters": args.iters,
               "n_samples": args.samples, "seeds": args.seeds,
               "lr": args.lr, "reg": args.reg, "runs": runs}
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {cache_path}")
    return results


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(results):
    N    = results["N"]
    runs = results["runs"]
    by   = {"marshall": {}, "no_marshall": {}}
    for r in runs:
        by[r["mode"]].setdefault(r["J2_J1"], []).append(
            (r["E_RBM"] - r["E_exact"]) / N
        )
    out = {}
    for mode in by:
        ratios, means, stds = [], [], []
        for j2 in sorted(by[mode]):
            errs = by[mode][j2]
            ratios.append(j2); means.append(np.mean(errs)); stds.append(np.std(errs))
        out[mode] = (np.array(ratios), np.array(means), np.array(stds))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results, out_dir):
    """Two independent single-panel figures (energy error, average sign),
    meant to be placed side by side in the report rather than stacked."""
    setup_style()

    N   = results["N"]
    agg = aggregate(results)

    # Use the same J2/J1 grid as the training data — not a dense sweep
    ratios_m, _, _ = agg["marshall"]
    print(f"Computing average signs at {len(ratios_m)} training points (N={N}) …")
    as_raw, as_marshall = compute_average_signs(N, ratios_m)

    kw_m  = dict(color="#2563eb", lw=1.8, marker="o", markersize=4.5,
                 markerfacecolor="white", markeredgewidth=1.3)
    kw_nm = dict(color="#dc2626", lw=1.8, marker="s", markersize=4.5,
                 markerfacecolor="white", markeredgewidth=1.3)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- panel (a): energy error / site (log scale) ---
    fig_e, ax = plt.subplots(figsize=(6.5, 3.8))
    for (mode, kw, label) in [
        ("marshall",    kw_m,  r"With Marshall ($\Psi = s \cdot A$)"),
        ("no_marshall", kw_nm, r"Without Marshall ($\Psi = A$)"),
    ]:
        ratios, means, stds = agg[mode]
        means_abs = np.abs(means)
        ax.plot(ratios, means_abs, label=label, **kw)
        ax.fill_between(ratios, np.clip(means_abs - stds, 1e-8, None), means_abs + stds,
                        alpha=0.18, color=kw["color"])

    ax.set_yscale("log")
    ax.set_ylabel(r"Energy error / site" + "\n"
                  r"$|E_{\mathrm{RBM}}-E_{\mathrm{exact}}|/N$")
    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")

    fig_e.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"marshall_comparison_energy.{ext}"
        fig_e.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")

    # --- panel (b): average sign <s> = 1 - 2*sign_impurity ---
    fig_s, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.plot(ratios_m, as_marshall, label=r"Marshall basis $\phi = s\!\cdot\!\Psi$",
            color="#2563eb", lw=1.8, marker="o", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.3)
    ax.plot(ratios_m, as_raw,     label=r"Raw basis $\phi = \Psi$",
            color="#dc2626", lw=1.8, marker="s", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.3)
    ax.axhline(0, color="#aaa", lw=0.7, ls="--", zorder=0)
    ax.axvline(0.5, color="#222", ls=":", lw=1.2, zorder=0)
    ax.set_ylabel(r"Average sign $\langle s \rangle$")
    ax.set_ylim(-0.08, 1.06)
    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=9, loc="upper left")
    fig_s.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"marshall_comparison_sign.{ext}"
        fig_s.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",    type=int,   default=8)
    p.add_argument("--alpha",   type=int,   default=3)
    p.add_argument("--seeds",   type=int,   default=3)
    p.add_argument("--iters",   type=int,   default=200)
    p.add_argument("--samples", type=int,   default=600)
    p.add_argument("--lr",      type=float, default=0.03)
    p.add_argument("--reg",     type=float, default=5e-4)
    p.add_argument("--steps",   type=int,   default=11)
    p.add_argument("--retrain", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    results = run_sweep(args)
    out_dir = _ROOT / "plots" / "j1j2"
    plot(results, out_dir)


if __name__ == "__main__":
    main()

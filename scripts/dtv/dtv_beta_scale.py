"""
D_TV and β_eff vs LSB energy scale (beta_x): TFIM and Heisenberg XXZ.

Analogue of Nelson et al. (PRApp 17, 044046, 2022) Figs. 1–2 for classical LSB.

LSB operates on the full (visible + hidden) RBM-to-Ising graph.  The coupling
matrix and fields are rescaled by beta_x:

    M ← W / beta_x,    f ← [a, b] / beta_x

For a *perfect* Gibbs sampler at β=1 over the joint (v, h) Ising model, the
visible marginal would be p(v) ∝ |Ψ(v)|^{2/beta_x}, i.e. β_eff = 1/beta_x.
LSB is a heuristic bifurcation algorithm — not an exact Gibbs sampler — so the
actual β_eff of its visible samples deviates from this ideal in a non-trivial
way that depends on lsb_steps, sigma, and the problem landscape.

Experiment: sweep beta_x over a range (analogous to α_in in the paper) for
several lsb_steps values (analogous to annealing time).  Measure:

  1. D_TV(visible samples, exact |Ψ(v)|²)        — quality metric
  2. β_eff via argmin_β D_KL(p_S ∥ |Ψ|^{2β})   — effective temperature

The ideal β_eff = 1/beta_x reference is plotted for comparison.

Usage:
    python scripts/dtv_beta_scale.py
    python scripts/dtv_beta_scale.py --model heisenberg
    python scripts/dtv_beta_scale.py --model both --retrain
    python scripts/dtv_beta_scale.py --beta-x-min 0.3 --beta-x-max 0.8 --beta-x-n 21
    python scripts/dtv_beta_scale.py --beta-x-values 0.1 0.5 1.0 2.0
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "scripts" / "viz"))

from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D, HeisenbergXXZ1D
from sampler import ClassicalSampler
from encoder import SRLinearSystem, conjugate_gradient
from plot_style import setup_style
from kl_utils import (
    all_configs_jax,
    exact_psi_sq,
    empirical_dist_jax,
    d_tv,
    finite_sampling_floor,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_ckpt(rbm, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({
            "a": np.array(rbm.a).tolist(),
            "b": np.array(rbm.b).tolist(),
            "W": np.array(rbm.W).tolist(),
        }, f)


def _load_ckpt(rbm, path: Path) -> None:
    with open(path, "rb") as f:
        d = pickle.load(f)
    rbm.a = jnp.array(d["a"])
    rbm.b = jnp.array(d["b"])
    rbm.W = jnp.array(d["W"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(rbm, ising, train_sampler_method, n_samples, n_iter, lr, reg,
           ckpt_path: Path, label: str) -> None:
    if ckpt_path.exists():
        print(f"[{label}] checkpoint found — loading.")
        _load_ckpt(rbm, ckpt_path)
        return

    print(f"[{label}] training  N={rbm.n_visible}  M={rbm.n_hidden}"
          f"  sampler={train_sampler_method}  {n_iter} iters ...")
    sampler = ClassicalSampler(train_sampler_method, n_warmup=100, n_sweeps=20)

    for it in range(1, n_iter + 1):
        V = jnp.asarray(sampler.sample(rbm, n_samples), dtype=jnp.float64)
        E = ising.local_energy_batch(V, rbm)

        if not bool(jnp.all(jnp.isfinite(E))):
            raise RuntimeError(f"[{label}] NaN/inf at iteration {it}.")

        Theta = V @ rbm.W + rbm.b[None, :]
        TanH = jnp.tanh(Theta)
        sr = SRLinearSystem(V, TanH, E, reg)
        x, _ = conjugate_gradient(sr.matvec, sr.force, tol=1e-8, maxiter=200)
        xa, xb, xW = sr.unpack(x)
        update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
        rbm.set_weights(rbm.get_weights() - lr * update)

        if it % 20 == 0 or it == n_iter:
            print(f"  iter {it:3d}  E={float(jnp.mean(E)):.6f}")

    _save_ckpt(rbm, ckpt_path)
    print(f"  saved → {ckpt_path.name}")


# ---------------------------------------------------------------------------
# β_eff estimation
# ---------------------------------------------------------------------------

def _estimate_beta_eff(energies_np: np.ndarray, p_emp_np: np.ndarray,
                       beta_bounds=(0.01, 200.0)) -> float:
    """
    β_eff = argmin_β D_KL(p_S ∥ p_β)  where p_β(v) ∝ exp(-β * E(v)).

    E(v) = -2 * log|Ψ(v)|, so β=1 recovers the target |Ψ|² distribution.

    For a perfect Gibbs sampler at energy scale beta_x, we expect β_eff = 1/beta_x.
    Deviations from this ideal line show where LSB behaves non-Gibbsian.
    """
    def _logsumexp(a):
        c = float(np.max(a))
        return c + float(np.log(np.sum(np.exp(a - c))))

    def objective(beta):
        log_unnorm = -beta * energies_np
        log_Z = _logsumexp(log_unnorm)
        log_b = log_unnorm - log_Z
        mask = p_emp_np > 0
        return float(np.sum(p_emp_np[mask] * (np.log(p_emp_np[mask]) - log_b[mask])))

    result = minimize_scalar(objective, bounds=beta_bounds, method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# beta_x sweep
# ---------------------------------------------------------------------------

def _sweep_beta_x(rbm, beta_x_values, steps_list, n_samples, n_seeds, N,
                  lsb_delta=0.1, lsb_gamma=0.1, lsb_sigma=1.0):
    """
    For each (beta_x, lsb_steps) pair draw n_seeds sample sets; compute D_TV + β_eff.

    Returns
    -------
    dtv_results  : dict[steps → dict[beta_x → list[float]]]  (fraction, not %)
    beta_results : dict[steps → dict[beta_x → list[float]]]  (β_eff values)
    p_exact      : (2^N,) exact |Ψ(v)|² probabilities
    """
    configs = all_configs_jax(N)
    p_exact = exact_psi_sq(rbm, N)
    energies_np = np.asarray(-2.0 * jax.vmap(rbm.log_psi)(configs))

    dtv_results = {s: {} for s in steps_list}
    beta_results = {s: {} for s in steps_list}

    sampler = ClassicalSampler("lsb")

    for s_idx, steps in enumerate(steps_list):
        print(f"  lsb_steps={steps}:")
        for bx_idx, beta_x in enumerate(beta_x_values):
            dtv_vals, beta_vals = [], []
            cfg = {
                "beta_x": beta_x,
                "lsb_steps": steps,
                "lsb_delta": lsb_delta,
                "lsb_gamma": lsb_gamma,
                "lsb_sigma": lsb_sigma,
            }
            for seed in range(n_seeds):
                sampler._key = jax.random.PRNGKey(s_idx * 10000 + bx_idx * 100 + seed)
                v, _ = sampler.sample(rbm, n_samples, config=cfg, return_hidden=True,
                                      return_jax=True)
                p_emp = empirical_dist_jax(v, N)
                dtv_vals.append(float(d_tv(p_exact, p_emp)))
                beta_vals.append(_estimate_beta_eff(energies_np, np.asarray(p_emp)))

            dtv_results[steps][beta_x] = dtv_vals
            beta_results[steps][beta_x] = beta_vals
            ideal = 1.0 / beta_x
            print(f"    beta_x={beta_x:.3f}  "
                  f"D_TV={np.mean(dtv_vals)*100:.1f}%  "
                  f"β_eff={np.mean(beta_vals):.2f}  "
                  f"(ideal={ideal:.2f})")

    return dtv_results, beta_results, p_exact


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STEPS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_IDEAL_COLOR = "#888888"
_FLOOR_COLOR = "#333333"
_SWEET_COLOR = "#222222"
_FIG_W, _FIG_H = 5.0, 4.2


def _sweet_spot(beta_x_values, dtv_results, steps_list):
    """beta_x that minimises grand-mean D_TV (over all steps and seeds)."""
    grand = [np.mean([v for s in steps_list for v in dtv_results[s][bx]])
             for bx in beta_x_values]
    return beta_x_values[int(np.argmin(grand))]


def _plot_dtv_panel(ax, beta_x_values, dtv_results, beta_results, steps_list,
                    floor_pct, panel_label, show_ylabel, show_xlabel):
    for i, steps in enumerate(steps_list):
        color = _STEPS_COLORS[i % len(_STEPS_COLORS)]
        means = np.array([np.mean(dtv_results[steps][bx]) for bx in beta_x_values]) * 100
        stds = np.array([np.std(dtv_results[steps][bx]) for bx in beta_x_values]) * 100
        ax.plot(beta_x_values, means, "o-", color=color,
                label=f"{steps} steps", linewidth=2, markersize=5)
        ax.fill_between(beta_x_values, means - stds, means + stds,
                        alpha=0.18, color=color)
    ax.axhline(floor_pct, color=_FLOOR_COLOR, linestyle="--", linewidth=1.5,
               label="sampling floor")

    sweet_bx = _sweet_spot(beta_x_values, dtv_results, steps_list)
    ax.axvline(sweet_bx, color=_SWEET_COLOR, linestyle=":", linewidth=1.5)
    ax.text(sweet_bx, 1.02,
            rf"$\beta_x\approx{sweet_bx:.3f}$",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=15, clip_on=False)

    if show_xlabel:
        ax.set_xlabel(r"$\beta_x$")
    if show_ylabel:
        ax.set_ylabel(r"$D_\mathrm{TV}$ (\%)")
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
            va="top", ha="left", fontsize=15)
    ax.legend(fontsize=13, loc="upper right")


def _plot_beta_panel(ax, beta_x_values, dtv_results, beta_results, steps_list,
                     floor_pct, panel_label, show_ylabel, show_xlabel):
    ideal = np.array([1.0 / bx for bx in beta_x_values])
    ax.plot(beta_x_values, ideal, "--", color=_IDEAL_COLOR, linewidth=1.5,
            label=r"ideal $1/\beta_x$")

    for i, steps in enumerate(steps_list):
        color = _STEPS_COLORS[i % len(_STEPS_COLORS)]
        means = np.array([np.mean(beta_results[steps][bx]) for bx in beta_x_values])
        stds = np.array([np.std(beta_results[steps][bx]) for bx in beta_x_values])
        ax.plot(beta_x_values, means, "o-", color=color,
                label=f"{steps} steps", linewidth=2, markersize=5)
        ax.fill_between(beta_x_values, means - stds, means + stds,
                        alpha=0.18, color=color)

    ax.axhline(1.0, color=_FLOOR_COLOR, linestyle=":", linewidth=1.2,
               label=r"$\beta_{\mathrm{eff}}=1$")

    sweet_bx = _sweet_spot(beta_x_values, dtv_results, steps_list)
    ax.axvline(sweet_bx, color=_SWEET_COLOR, linestyle=":", linewidth=1.5)
    ax.text(sweet_bx, 1.02,
            rf"$\beta_x\approx{sweet_bx:.3f}$",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=15, clip_on=False)

    if show_xlabel:
        ax.set_xlabel(r"$\beta_x$")
    if show_ylabel:
        ax.set_ylabel(r"$\beta_{\mathrm{eff}}$")
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
            va="top", ha="left", fontsize=15)
    ax.legend(fontsize=13, loc="upper right")


def _make_plots(rows_data, beta_x_values, steps_list, n_samples, out_dir):
    """
    rows_data : list of (N, results_list) — one entry per row.
    Produces one PDF per (N, fig_key) pair.
    """
    n_cols = max(len(rl) for _, rl in rows_data)
    setup_style(fontsize=16, scale=2.5 * n_cols)

    for fig_key, ylabel_str, panel_fn in [
        ("dtv",  r"$D_{\mathrm{TV}}$ (\%)",  _plot_dtv_panel),
        ("beta", r"$\beta_{\mathrm{eff}}$",  _plot_beta_panel),
    ]:
        for N, results_list in rows_data:
            n = len(results_list)
            fig, axes = plt.subplots(
                1, n,
                figsize=(_FIG_W * n, _FIG_H),
                squeeze=False,
            )
            for col_idx, entry in enumerate(results_list):
                ax = axes[0, col_idx]
                show_ylabel = (col_idx == 0)
                panel_fn(
                    ax,
                    beta_x_values,
                    entry["dtv"], entry["beta"],
                    steps_list,
                    entry["floor"],
                    entry["label"],
                    show_ylabel=show_ylabel,
                    show_xlabel=True,
                )
                if show_ylabel:
                    ax.set_ylabel(ylabel_str)

            fig.tight_layout()
            out_path = out_dir / f"dtv_beta_scale_{fig_key}_N{N}.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            print(f"  {fig_key} N={N} → {out_path}")
            plt.close(fig)

# ---------------------------------------------------------------------------
# JSON serialisation  (multi-N aware)
# ---------------------------------------------------------------------------

def _save_json(rows_data, beta_x_values, steps_list, n_samples, path: Path):
    def _ser_entry(e):
        return {
            "label": e["label"],
            "floor": e["floor"],
            "dtv": {str(s): {str(bx): v for bx, v in e["dtv"][s].items()}
                    for s in steps_list},
            "beta": {str(s): {str(bx): v for bx, v in e["beta"][s].items()}
                     for s in steps_list},
        }
    payload = {
        "n_samples": n_samples,
        "beta_x_values": beta_x_values,
        "steps_list": steps_list,
        "rows": [{"N": N, "entries": [_ser_entry(e) for e in rl]}
                 for N, rl in rows_data],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results → {path}")


def _load_json(path: Path):
    with open(path) as f:
        raw = json.load(f)
    beta_x_values = raw["beta_x_values"]
    steps_list = raw["steps_list"]

    def _deser_entry(e):
        return {
            "label": e["label"],
            "floor": e["floor"],
            "dtv": {int(s): {float(bx): v for bx, v in e["dtv"][str(s)].items()}
                    for s in steps_list},
            "beta": {int(s): {float(bx): v for bx, v in e["beta"][str(s)].items()}
                     for s in steps_list},
        }

    rows_data = [(row["N"], [_deser_entry(e) for e in row["entries"]])
                 for row in raw["rows"]]
    return rows_data, beta_x_values, steps_list, raw["n_samples"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["tfim", "heisenberg", "both"], default="tfim")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12],
                   help="System sizes to run (one row per size, default: 8 12)")
    p.add_argument("--n-hidden", type=int, default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--J-values", type=float, nargs="+", default=[1.0])
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument(
        "--beta-x-values", type=float, nargs="+",
        default=[0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60],
    )
    p.add_argument(
        "--steps-list", type=int, nargs="+",
        default=[200, 1000, 5000],
        help="LSB step counts to compare (analogous to annealing times)",
    )
    p.add_argument("--lsb-delta", type=float, default=0.1)
    p.add_argument("--lsb-gamma", type=float, default=0.1)
    p.add_argument("--lsb-sigma", type=float, default=1.0)
    p.add_argument("--floor-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _run_one_size(N, M, args, repo_root):
    """Train + sweep for a single system size. Returns results_list."""
    ckpt_dir = repo_root / "checkpoints" / "dtv_beta_scale" / f"N{N}_M{M}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_tfim = args.model in ("tfim", "both")
    run_heis = args.model in ("heisenberg", "both")
    results_list = []

    # ── TFIM ──────────────────────────────────────────────────────────────
    if run_tfim:
        for h in args.h_values:
            label = f"TFIM h={h}"
            ckpt_label = f"tfim_N{N}_h{h}_M{M}"
            ckpt_path = ckpt_dir / f"{ckpt_label}_trained.pkl"
            if args.retrain and ckpt_path.exists():
                ckpt_path.unlink()

            print(f"\n{'='*60}\nTFIM  N={N}  h={h}\n{'='*60}")
            rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(args.seed))
            ising = TransverseFieldIsing1D(N, h)
            _train(rbm, ising, "simulated_annealing",
                   args.train_samples, args.n_iter, args.lr, args.reg,
                   ckpt_path, ckpt_label)

            p_exact = exact_psi_sq(rbm, N)
            floor = finite_sampling_floor(p_exact, args.n_samples, args.floor_trials)
            print(f"  finite-sampling floor: {floor*100:.2f}%")

            dtv_r, beta_r, _ = _sweep_beta_x(
                rbm, args.beta_x_values, args.steps_list,
                args.n_samples, args.n_seeds, N,
                lsb_delta=args.lsb_delta,
                lsb_gamma=args.lsb_gamma,
                lsb_sigma=args.lsb_sigma,
            )
            results_list.append({"label": label, "floor": floor,
                                  "dtv": dtv_r, "beta": beta_r})

    # ── Heisenberg ────────────────────────────────────────────────────────
    if run_heis:
        for J in args.J_values:
            label = f"Heisenberg J={J}"
            ckpt_label = f"heis_N{N}_J{J}_M{M}"
            ckpt_path = ckpt_dir / f"{ckpt_label}_trained.pkl"
            if args.retrain and ckpt_path.exists():
                ckpt_path.unlink()

            print(f"\n{'='*60}\nHeisenberg XXZ  N={N}  J={J}\n{'='*60}")
            rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(args.seed + 1000))
            heis = HeisenbergXXZ1D(N, J=J)
            _train(rbm, heis, "exchange",
                   args.train_samples, args.n_iter, args.lr, args.reg,
                   ckpt_path, ckpt_label)

            p_exact = exact_psi_sq(rbm, N)
            floor = finite_sampling_floor(p_exact, args.n_samples, args.floor_trials)
            print(f"  finite-sampling floor: {floor*100:.2f}%")

            dtv_r, beta_r, _ = _sweep_beta_x(
                rbm, args.beta_x_values, args.steps_list,
                args.n_samples, args.n_seeds, N,
                lsb_delta=args.lsb_delta,
                lsb_gamma=args.lsb_gamma,
                lsb_sigma=args.lsb_sigma,
            )
            results_list.append({"label": label, "floor": floor,
                                  "dtv": dtv_r, "beta": beta_r})

    return results_list


def main():
    args = parse_args()

    for N in args.sizes:
        if N > 16:
            raise SystemExit(f"--sizes includes {N} > 16: exact enumeration requires N ≤ 16.")

    repo_root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "plots" / "dtv_beta_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes_tag = "_".join(str(N) for N in args.sizes)
    json_path = out_dir / f"dtv_beta_scale_N{sizes_tag}.json"

    if args.plot_only:
        if not json_path.exists():
            raise SystemExit(f"No results at {json_path}. Run without --plot-only first.")
        rows_data, beta_x_values, steps_list, n_samples = _load_json(json_path)
        _make_plots(rows_data, beta_x_values, steps_list, n_samples, out_dir)
        return

    rows_data = []
    for N in args.sizes:
        M = args.n_hidden if args.n_hidden is not None else N
        results_list = _run_one_size(N, M, args, repo_root)
        rows_data.append((N, results_list))

    _save_json(rows_data, args.beta_x_values, args.steps_list, args.n_samples, json_path)
    _make_plots(rows_data, args.beta_x_values, args.steps_list, args.n_samples, out_dir)


if __name__ == "__main__":
    main()

"""
D_TV vs SA final temperature: TFIM and Heisenberg XXZ RBMs.

Analogue of Nelson et al. (PRApp 17, 044046, 2022) Figs. 1–2, using classical
SA instead of D-Wave QA.  For a fixed trained RBM, sweeps SA final temperature
T_final while holding T_initial fixed, measuring:

  1. D_TV(empirical histogram, exact |Ψ(v)|²)   — sampling quality
  2. β_eff via argmin_β D_KL(p_S ∥ |Ψ|^{2β})  — effective inverse temperature

Lower T_final (cold SA): sampler over-concentrates at modes → high D_TV, high β_eff.
Higher T_final (hot SA): sampler approaches uniform → high D_TV, low β_eff.
Sweet spot at intermediate T_final where SA tracks |Ψ|² most faithfully (β_eff ≈ 1).

Three n_sweeps values (SA effort) are compared as separate lines, analogous to the
four annealing times in the Nelson et al. experiment.

Usage:
    python scripts/dtv_temperature.py
    python scripts/dtv_temperature.py --model heisenberg
    python scripts/dtv_temperature.py --model both --retrain
    python scripts/dtv_temperature.py --size 8 --h-values 0.5 1.0 --n-samples 20000
"""

import argparse
import json
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
                       beta_bounds=(0.01, 100.0)) -> float:
    """
    β_eff = argmin_β D_KL(p_S ∥ p_β)  where p_β(v) ∝ exp(-β * E(v)).

    E(v) = -2 * log|Ψ(v)|, so β=1 recovers the target distribution |Ψ|².
    β_eff > 1 → sampler produced an over-concentrated (colder) distribution.
    β_eff < 1 → sampler produced an under-concentrated (hotter) distribution.

    energies_np : (2^N,)  precomputed E values for all configs
    p_emp_np    : (2^N,)  empirical probabilities from sampler
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
# Temperature sweep
# ---------------------------------------------------------------------------

def _sweep_temperature(rbm, T_final_values, n_sweeps_list, n_samples, n_seeds, N,
                       T_initial=5.0):
    """
    For each (T_final, n_sweeps) pair draw n_seeds sample sets and compute D_TV + β_eff.

    Returns
    -------
    dtv_results  : dict[n_sweeps → dict[T_final → list[float]]]   (fraction, not %)
    beta_results : dict[n_sweeps → dict[T_final → list[float]]]   (β_eff values)
    p_exact      : (2^N,) exact |Ψ(v)|² probabilities
    """
    configs = all_configs_jax(N)
    p_exact = exact_psi_sq(rbm, N)
    # Precompute E(v) = -2*log|Ψ(v)| for all 2^N configs in one vmap call.
    energies_np = np.asarray(-2.0 * jax.vmap(rbm.log_psi)(configs))

    dtv_results = {ns: {} for ns in n_sweeps_list}
    beta_results = {ns: {} for ns in n_sweeps_list}

    # One sampler object reused across all (T_final, n_sweeps, seed) calls.
    sampler = ClassicalSampler("simulated_annealing", n_warmup=50, n_sweeps=1)

    for ns_idx, n_sweeps in enumerate(n_sweeps_list):
        print(f"  n_sweeps={n_sweeps}:")
        for t_idx, T_final in enumerate(T_final_values):
            dtv_vals, beta_vals = [], []
            cfg = {"n_sweeps": n_sweeps, "T_initial": T_initial, "T_final": T_final}

            for seed in range(n_seeds):
                # Deterministic, unique key per (n_sweeps_idx, t_idx, seed)
                sampler._key = jax.random.PRNGKey(ns_idx * 10000 + t_idx * 100 + seed)
                v = sampler.sample(rbm, n_samples, config=cfg, return_jax=True)
                p_emp = empirical_dist_jax(v, N)
                dtv_vals.append(float(d_tv(p_exact, p_emp)))
                beta_vals.append(_estimate_beta_eff(energies_np, np.asarray(p_emp)))

            dtv_results[n_sweeps][T_final] = dtv_vals
            beta_results[n_sweeps][T_final] = beta_vals
            print(f"    T_final={T_final:.3f}  "
                  f"D_TV={np.mean(dtv_vals)*100:.1f}%  "
                  f"β_eff={np.mean(beta_vals):.2f}")

    return dtv_results, beta_results, p_exact


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_SWEEPS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_FLOOR_COLOR = "#555555"
_FIG_W, _FIG_H = 5.0, 4.2


def _plot_dtv_panel(ax, T_final_values, dtv_results, n_sweeps_list, floor_pct, title):
    for i, ns in enumerate(n_sweeps_list):
        color = _SWEEPS_COLORS[i % len(_SWEEPS_COLORS)]
        means = np.array([np.mean(dtv_results[ns][T]) for T in T_final_values]) * 100
        stds = np.array([np.std(dtv_results[ns][T]) for T in T_final_values]) * 100
        ax.plot(T_final_values, means, "o-", color=color,
                label=f"{ns} sweeps", linewidth=2, markersize=5)
        ax.fill_between(T_final_values, means - stds, means + stds,
                        alpha=0.18, color=color)
    ax.axhline(floor_pct, color=_FLOOR_COLOR, linestyle="--", linewidth=1.5,
               label="sampling floor")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_\mathrm{final}$")
    ax.set_ylabel(r"$D_\mathrm{TV}$ (\%)")
    ax.set_title(title)
    ax.legend(fontsize=9)


def _plot_beta_panel(ax, T_final_values, beta_results, n_sweeps_list, title):
    for i, ns in enumerate(n_sweeps_list):
        color = _SWEEPS_COLORS[i % len(_SWEEPS_COLORS)]
        means = np.array([np.mean(beta_results[ns][T]) for T in T_final_values])
        stds = np.array([np.std(beta_results[ns][T]) for T in T_final_values])
        ax.plot(T_final_values, means, "o-", color=color,
                label=f"{ns} sweeps", linewidth=2, markersize=5)
        ax.fill_between(T_final_values, means - stds, means + stds,
                        alpha=0.18, color=color)
    ax.axhline(1.0, color=_FLOOR_COLOR, linestyle="--", linewidth=1.5,
               label=r"$\beta_\mathrm{eff}=1$ (target)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_\mathrm{final}$")
    ax.set_ylabel(r"$\beta_\mathrm{eff}$")
    ax.set_title(title)
    ax.legend(fontsize=9)


def _make_plots(results_list, T_final_values, n_sweeps_list, N, n_samples, out_dir):
    """
    results_list: list of (label, dtv_results, beta_results, floor) dicts
    Produces one combined figure (all labels in columns) for D_TV and β_eff.
    """
    n = len(results_list)
    setup_style(fontsize=12, scale=2.5 * n)

    # Figure 1: D_TV
    fig1, axes1 = plt.subplots(1, n, figsize=(_FIG_W * n, _FIG_H), sharey=True)
    if n == 1:
        axes1 = [axes1]
    for ax, entry in zip(axes1, results_list):
        _plot_dtv_panel(ax, T_final_values, entry["dtv"], n_sweeps_list,
                        entry["floor"] * 100, entry["label"])
        if ax is not axes1[0]:
            ax.set_ylabel("")
    fig1.suptitle(
        rf"$D_\mathrm{{TV}}$ vs SA final temperature $\mid$ N={N}"
        rf"  $n_\mathrm{{samples}}={n_samples}$  $T_\mathrm{{initial}}=5$"
    )
    fig1.tight_layout()
    out1 = out_dir / f"dtv_temperature_dtv_N{N}.pdf"
    fig1.savefig(out1)
    print(f"  D_TV plot → {out1}")
    plt.close(fig1)

    # Figure 2: β_eff
    fig2, axes2 = plt.subplots(1, n, figsize=(_FIG_W * n, _FIG_H), sharey=True)
    if n == 1:
        axes2 = [axes2]
    for ax, entry in zip(axes2, results_list):
        _plot_beta_panel(ax, T_final_values, entry["beta"], n_sweeps_list, entry["label"])
        if ax is not axes2[0]:
            ax.set_ylabel("")
    fig2.suptitle(
        rf"Effective temperature $\beta_\mathrm{{eff}}$ vs SA final temperature $\mid$ N={N}"
    )
    fig2.tight_layout()
    out2 = out_dir / f"dtv_temperature_beta_N{N}.pdf"
    fig2.savefig(out2)
    print(f"  β_eff plot → {out2}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _save_json(results_list, T_final_values, n_sweeps_list, N, n_samples, path: Path):
    payload = {
        "N": N, "n_samples": n_samples,
        "T_final_values": T_final_values,
        "n_sweeps_list": n_sweeps_list,
        "entries": [],
    }
    for entry in results_list:
        serialised = {
            "label": entry["label"],
            "floor": entry["floor"],
            "dtv": {
                str(ns): {str(T): v for T, v in entry["dtv"][ns].items()}
                for ns in n_sweeps_list
            },
            "beta": {
                str(ns): {str(T): v for T, v in entry["beta"][ns].items()}
                for ns in n_sweeps_list
            },
        }
        payload["entries"].append(serialised)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results → {path}")


def _load_json(path: Path):
    with open(path) as f:
        raw = json.load(f)
    T_final_values = raw["T_final_values"]
    n_sweeps_list = raw["n_sweeps_list"]
    results_list = []
    for entry in raw["entries"]:
        dtv = {int(ns): {float(T): v for T, v in entry["dtv"][str(ns)].items()}
               for ns in n_sweeps_list}
        beta = {int(ns): {float(T): v for T, v in entry["beta"][str(ns)].items()}
                for ns in n_sweeps_list}
        results_list.append({
            "label": entry["label"],
            "floor": entry["floor"],
            "dtv": dtv,
            "beta": beta,
        })
    return results_list, T_final_values, n_sweeps_list, raw["N"], raw["n_samples"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["tfim", "heisenberg", "both"], default="tfim",
                   help="Which model(s) to run (default: tfim)")
    p.add_argument("--size", type=int, default=8,
                   help="Number of spins N (must be ≤ 16 for exact enumeration)")
    p.add_argument("--n-hidden", type=int, default=None,
                   help="RBM hidden units (default: N)")
    p.add_argument("--h-values", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="TFIM transverse field values (default: 0.5 1.0 2.0)")
    p.add_argument("--J-values", type=float, nargs="+", default=[1.0],
                   help="Heisenberg exchange coupling values (default: 1.0)")
    p.add_argument("--n-iter", type=int, default=200,
                   help="SR training iterations (default: 200)")
    p.add_argument("--train-samples", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=10_000,
                   help="Samples per (T_final, n_sweeps, seed) evaluation")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Independent seeds per (T_final, n_sweeps) point")
    p.add_argument("--T-initial", type=float, default=5.0)
    p.add_argument(
        "--T-final-values", type=float, nargs="+",
        default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="SA final temperatures to sweep (default: 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0)",
    )
    p.add_argument(
        "--n-sweeps-list", type=int, nargs="+", default=[10, 50, 200],
        help="SA annealing efforts to compare (default: 10 50 200)",
    )
    p.add_argument("--floor-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain", action="store_true",
                   help="Delete existing checkpoints and retrain")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip sampling; reload existing JSON and regenerate plots")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    N = args.size
    M = args.n_hidden if args.n_hidden is not None else N

    if N > 16:
        raise SystemExit(f"--size {N} > 16: exact enumeration requires N ≤ 16.")

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / "dtv_temperature" / f"N{N}_M{M}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "output" / "dtv_temperature"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"dtv_temperature_N{N}_M{M}.json"

    if args.plot_only:
        if not json_path.exists():
            raise SystemExit(f"No results file at {json_path}. Run without --plot-only first.")
        results_list, T_final_values, n_sweeps_list, N, n_samples = _load_json(json_path)
        _make_plots(results_list, T_final_values, n_sweeps_list, N, n_samples, out_dir)
        return

    # Determine which models to run
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

            print(f"\n{'='*60}")
            print(f"TFIM  N={N}  h={h}")
            print("=" * 60)

            key = jax.random.PRNGKey(args.seed)
            rbm = FullyConnectedRBM(N, M, key)
            ising = TransverseFieldIsing1D(N, h)

            _train(rbm, ising, "simulated_annealing",
                   args.train_samples, args.n_iter, args.lr, args.reg,
                   ckpt_path, ckpt_label)

            p_exact = exact_psi_sq(rbm, N)
            floor = finite_sampling_floor(p_exact, args.n_samples, args.floor_trials)
            print(f"  finite-sampling floor: {floor*100:.2f}%")

            dtv_r, beta_r, _ = _sweep_temperature(
                rbm, args.T_final_values, args.n_sweeps_list,
                args.n_samples, args.n_seeds, N, T_initial=args.T_initial,
            )
            results_list.append({
                "label": label,
                "floor": floor,
                "dtv": dtv_r,
                "beta": beta_r,
            })

    # ── Heisenberg ────────────────────────────────────────────────────────
    if run_heis:
        for J in args.J_values:
            label = f"Heisenberg J={J}"
            ckpt_label = f"heis_N{N}_J{J}_M{M}"
            ckpt_path = ckpt_dir / f"{ckpt_label}_trained.pkl"

            if args.retrain and ckpt_path.exists():
                ckpt_path.unlink()

            print(f"\n{'='*60}")
            print(f"Heisenberg XXZ  N={N}  J={J}")
            print("=" * 60)

            key = jax.random.PRNGKey(args.seed + 1000)
            rbm = FullyConnectedRBM(N, M, key)
            # Heisenberg needs spin-exchange sampler for correct S_z-conserving training.
            heis = HeisenbergXXZ1D(N, J=J)

            _train(rbm, heis, "exchange",
                   args.train_samples, args.n_iter, args.lr, args.reg,
                   ckpt_path, ckpt_label)

            p_exact = exact_psi_sq(rbm, N)
            floor = finite_sampling_floor(p_exact, args.n_samples, args.floor_trials)
            print(f"  finite-sampling floor: {floor*100:.2f}%")

            # Temperature sweep uses SA (single-spin-flip) for both TFIM and Heisenberg —
            # this is intentional: we're characterising SA's behaviour on each problem,
            # not optimising the sampler for Heisenberg specifically.
            dtv_r, beta_r, _ = _sweep_temperature(
                rbm, args.T_final_values, args.n_sweeps_list,
                args.n_samples, args.n_seeds, N, T_initial=args.T_initial,
            )
            results_list.append({
                "label": label,
                "floor": floor,
                "dtv": dtv_r,
                "beta": beta_r,
            })

    _save_json(results_list, args.T_final_values, args.n_sweeps_list,
               N, args.n_samples, json_path)
    _make_plots(results_list, args.T_final_values, args.n_sweeps_list,
                N, args.n_samples, out_dir)


if __name__ == "__main__":
    main()

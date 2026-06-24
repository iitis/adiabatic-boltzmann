"""
D_TV vs sampler effort: SA vs LSB on a trained RBM.

Trains small RBMs (N=8) on TFIM at h ∈ {0.5, 1.0, 2.0}, then for each
frozen RBM sweeps the effort parameter of both classical samplers and
measures D_TV(empirical histogram, exact |Ψ(v)|²).

Entire pipeline (sample → histogram → D_TV) stays on GPU via JAX.

Usage (from repo root):
    python scripts/dtv_classical_samplers.py
    python scripts/dtv_classical_samplers.py --size 8 --n-samples 5000
    python scripts/dtv_classical_samplers.py --retrain
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
import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "scripts" / "viz"))

from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
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
# Training (inline SR, same pattern as dtv_vs_annealing.py)
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


def _train(rbm, ising, n_samples, n_iter, lr, reg, ckpt_path: Path, label: str) -> None:
    if ckpt_path.exists():
        print(f"[{label}] checkpoint found — skipping training.")
        _load_ckpt(rbm, ckpt_path)
        return

    print(f"[{label}] training  N={rbm.n_visible}  M={rbm.n_hidden}  {n_iter} iters ...")
    sampler = ClassicalSampler("simulated_annealing", n_warmup=100, n_sweeps=20)

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
# D_TV sweep (pure JAX hot loop)
# ---------------------------------------------------------------------------

def _sweep(rbm, sampler_method, effort_values, effort_config_key,
           n_samples, n_seeds, N, extra_config=None):
    """
    For each effort value, draw n_seeds independent sample sets and compute D_TV.

    Returns {effort_value: list[float]} (D_TV as fraction, not percent).
    """
    p_exact = exact_psi_sq(rbm, N)
    results = {}

    for effort in effort_values:
        vals = []
        sampler = ClassicalSampler(sampler_method, n_warmup=100, n_sweeps=1)
        for seed in range(n_seeds):
            sampler._key = jax.random.PRNGKey(seed * 1000 + effort)
            cfg = {effort_config_key: effort, **(extra_config or {})}
            v = sampler.sample(rbm, n_samples, config=cfg, return_jax=True)
            p_emp = empirical_dist_jax(v, N)
            vals.append(float(d_tv(p_exact, p_emp)))
        results[effort] = vals
        mean = np.mean(vals)
        print(f"    effort={effort:5d}  D_TV={mean*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_SAMPLER_STYLES = {
    "sa":         ("#1f77b4", "o-", "SA"),
    "metropolis": ("#ff7f0e", "s-", "Metropolis"),
    "gibbs":      ("#2ca02c", "^-", "Gibbs"),
    "lsb":        ("#d62728", "D-", "LSB"),
}
_FLOOR_COLOR = "#555555"


_FIG_W = 4.5  # per-panel width (shared by both plots)
_FIG_H = 4.5  # height


def _make_plot(all_data, h_values, N, n_samples, out_path: Path) -> None:
    n_h = len(h_values)
    setup_style(fontsize=12, scale=2.5 * n_h)
    fig, axes = plt.subplots(1, n_h, figsize=(_FIG_W * n_h, _FIG_H), sharey=True)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, h_values):
        data = all_data[h]
        floor_pct = data["floor"] * 100

        for key, (color, marker, label) in _SAMPLER_STYLES.items():
            if key not in data:
                continue
            res = data[key]
            xs = sorted(res.keys())
            means = np.array([np.mean(res[x]) for x in xs]) * 100
            stds = np.array([np.std(res[x]) for x in xs]) * 100
            ax.plot(xs, means, marker, color=color, label=label, linewidth=2, markersize=5)
            ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color=color)

        ax.axhline(floor_pct, color=_FLOOR_COLOR, linestyle="--", linewidth=1.5,
                   label="sampling floor")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xlabel("effort (sweeps / steps)")
        ax.set_title(f"$h = {h}$")
        if ax is axes[-1]:
            ax.legend(loc="upper right")

    axes[0].set_ylabel(r"$D_\mathrm{TV}$ (\%)")
    fig.suptitle(
        rf"Sampling quality: SA / MH / Gibbs / LSB $\mid$ TFIM 1D $N={N}$ $n_\mathrm{{samples}}={n_samples}$"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Plot saved → {out_path}")


def _make_dist_plot(all_data, h_values, N, out_path: Path) -> None:
    """Sorted |Ψ(v)|² rank plot — one panel per h value."""
    n_h = len(h_values)
    setup_style(fontsize=12, scale=2.5 * n_h)
    fig, axes = plt.subplots(1, n_h, figsize=(_FIG_W * n_h, _FIG_H), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, h_values):
        p_exact = all_data[h].get("p_exact")
        if p_exact is None:
            ax.set_visible(False)
            continue
        p = np.sort(np.asarray(p_exact))[::-1]
        n_states = len(p)
        ax.bar(np.arange(n_states), p, width=1.0, color="#4c72b0", alpha=0.8)
        ax.axhline(1.0 / n_states, color=_FLOOR_COLOR, linestyle="--",
                   linewidth=1.5, label="uniform")
        ax.set_xlabel("configuration rank")
        ax.set_title(f"$h = {h}$")
        #ax.set_xlim(0, n_states)
        ax.set_ylim(bottom=0)
        if ax is axes[-1]:
            ax.legend(loc="upper right")
    axes[0].set_xlim(0,50)
    axes[1].set_xlim(0,100)
    axes[0].set_ylabel(r"$|\Psi(v)|^2$")
    fig.suptitle(rf"Exact $|\Psi(v)|^2$ distribution $\mid$ TFIM 1D $N={N}$")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--n-hidden", type=int, default=None)
    p.add_argument("--h-values", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--train-samples", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument(
        "--sa-sweeps", type=int, nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    )
    p.add_argument(
        "--lsb-steps", type=int, nargs="+",
        default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
    )
    p.add_argument("--floor-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no-lsb-damping", action="store_true",
                   help="Disable LSB damping (set gamma=0, replicates undamped behaviour)")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip sampling; reload existing JSON and regenerate the plot only")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _load_json(json_path: Path):
    """Reconstruct all_data from a saved JSON, converting string keys back to int."""
    with open(json_path) as f:
        raw = json.load(f)
    h_values = raw["h_values"]
    all_data = {}
    for h in h_values:
        entry = raw["results"][str(h)]
        all_data[h] = {
            "floor": entry["floor"],
            "p_exact": entry.get("p_exact"),
            **{
                key: {int(k): v for k, v in entry[key].items()}
                for key in ("sa", "metropolis", "gibbs", "lsb")
                if key in entry
            },
        }
    return all_data, h_values, raw["size"], raw["n_samples"]


def main():
    args = parse_args()
    N = args.size
    M = args.n_hidden if args.n_hidden is not None else N

    if N > 16:
        raise SystemExit(f"--size {N} > 16: exact enumeration requires N ≤ 16.")

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / "dtv_classical" / f"N{N}_M{M}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "output" / "dtv_classical"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        json_path = out_dir / f"dtv_classical_N{N}_M{M}.json"
        if not json_path.exists():
            raise SystemExit(f"No results file found at {json_path}. Run without --plot-only first.")
        all_data, h_values, N, n_samples = _load_json(json_path)
        _make_plot(all_data, h_values, N, n_samples,
                   out_dir / f"dtv_classical_N{N}_M{M}.pdf")
        _make_dist_plot(all_data, h_values, N,
                        out_dir / f"dtv_classical_N{N}_M{M}_dist.pdf")
        return

    all_data = {}

    for h in args.h_values:
        label = f"tfim_N{N}_h{h}_M{M}"
        ckpt_path = ckpt_dir / f"{label}_trained.pkl"

        if args.retrain and ckpt_path.exists():
            ckpt_path.unlink()

        print(f"\n{'='*60}")
        print(f"h = {h}")
        print("=" * 60)

        key = jax.random.PRNGKey(args.seed)
        rbm = FullyConnectedRBM(N, M, key)
        ising = TransverseFieldIsing1D(N, h)

        _train(rbm, ising, args.train_samples, args.n_iter,
               args.lr, args.reg, ckpt_path, label)

        p_exact = exact_psi_sq(rbm, N)
        floor = finite_sampling_floor(p_exact, args.n_samples, args.floor_trials)
        print(f"  finite-sampling floor: {floor*100:.2f}%")

        print("  SA sweep:")
        sa_results = _sweep(rbm, "simulated_annealing", args.sa_sweeps,
                            "n_sweeps", args.n_samples, args.n_seeds, N)

        print("  Metropolis sweep:")
        mh_results = _sweep(rbm, "metropolis", args.sa_sweeps,
                            "n_sweeps", args.n_samples, args.n_seeds, N)

        print("  Gibbs sweep:")
        gibbs_results = _sweep(rbm, "gibbs", args.sa_sweeps,
                               "n_sweeps", args.n_samples, args.n_seeds, N)

        print("  LSB sweep:")
        lsb_extra = {"lsb_gamma": 0.0} if args.no_lsb_damping else {}
        lsb_results = _sweep(rbm, "lsb", args.lsb_steps,
                             "lsb_steps", args.n_samples, args.n_seeds, N,
                             extra_config=lsb_extra)

        all_data[h] = {
            "floor": floor,
            "p_exact": p_exact,
            "sa": sa_results,
            "metropolis": mh_results,
            "gibbs": gibbs_results,
            "lsb": lsb_results,
        }

    # Save JSON
    json_path = out_dir / f"dtv_classical_N{N}_M{M}.json"
    with open(json_path, "w") as f:
        json.dump({
            "size": N, "n_hidden": M, "h_values": args.h_values,
            "n_samples": args.n_samples, "n_seeds": args.n_seeds,
            "sa_sweeps": args.sa_sweeps, "lsb_steps": args.lsb_steps,
            "results": {
                str(h): {
                    "floor": v["floor"],
                    "p_exact": np.asarray(v["p_exact"]).tolist(),
                    **{
                        key: {str(k): vs for k, vs in v[key].items()}
                        for key in ("sa", "metropolis", "gibbs", "lsb")
                        if key in v
                    },
                }
                for h, v in all_data.items()
            },
        }, f, indent=2)
    print(f"\nResults saved → {json_path}")

    _make_plot(all_data, args.h_values, N, args.n_samples,
               out_dir / f"dtv_classical_N{N}_M{M}.pdf")
    _make_dist_plot(all_data, args.h_values, N,
                    out_dir / f"dtv_classical_N{N}_M{M}_dist.pdf")


if __name__ == "__main__":
    main()

"""
DTV vs D-Wave annealing time.

Trains small RBMs (N ≤ 16) on TFIM and Heisenberg using classical MH,
saves checkpoints at beginning / mid / trained stages, then sweeps D-Wave
annealing time and measures DTV against the exact |Ψ(v)|² distribution.

Usage (run from repo root):
    python scripts/dtv_vs_annealing.py --stage trained
    python scripts/dtv_vs_annealing.py --stage beginning --size 8
    python scripts/dtv_vs_annealing.py --stage mid --retrain

Note: this script consumes D-Wave QPU budget (tracked in src/time.json).
Each (annealing_time, rep) pair costs one D-Wave call.
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

# ── src imports ──────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D, HeisenbergXXZ1D
from sampler import ClassicalSampler, DimodSampler
from encoder import SRLinearSystem, conjugate_gradient


# ── checkpoint helpers ───────────────────────────────────────────────────────

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


# ── training ─────────────────────────────────────────────────────────────────

def _train(rbm, ising_model, n_samples, n_iter, lr, reg, ckpt_dir, label):
    """
    Train RBM with classical MH + SR.  Save checkpoints after 0, n_iter//2,
    and n_iter parameter updates (labelled beginning / mid / trained).

    Returns dict {stage: Path}.  If all three already exist, skips training.
    """
    mid_iter = max(1, n_iter // 2)
    stage_saves = {0: "beginning", mid_iter: "mid", n_iter: "trained"}

    ckpt_paths = {}
    missing = False
    for _, stage in stage_saves.items():
        p = ckpt_dir / f"{label}_{stage}.pkl"
        if p.exists():
            ckpt_paths[stage] = p
        else:
            missing = True

    if not missing:
        print(f"[{label}] All checkpoints found — skipping training.")
        return ckpt_paths

    print(f"[{label}] Training  N={rbm.n_visible}  M={rbm.n_hidden}  {n_iter} iters ...")
    sampler = ClassicalSampler("gibbs", n_warmup=200, n_sweeps=1)

    def _maybe_save(n_updates):
        if n_updates in stage_saves:
            stage = stage_saves[n_updates]
            p = ckpt_dir / f"{label}_{stage}.pkl"
            _save_ckpt(rbm, p)
            ckpt_paths[stage] = p
            return stage
        return None

    _maybe_save(0)

    for it in range(1, n_iter + 1):
        V_np = sampler.sample(rbm, n_samples, config={})
        V = jnp.asarray(V_np, dtype=jnp.float64)
        E = ising_model.local_energy_batch(V, rbm)

        if not bool(jnp.all(jnp.isfinite(E))):
            raise RuntimeError(f"[{label}] NaN/inf energy at iteration {it}.")

        Theta = V @ rbm.W + rbm.b[None, :]
        TanH = jnp.tanh(Theta)
        sr = SRLinearSystem(V, TanH, E, reg)
        x, cg_info = conjugate_gradient(sr.matvec, sr.force, tol=1e-8, maxiter=200)
        xa, xb, xW = sr.unpack(x)
        update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
        rbm.set_weights(rbm.get_weights() - lr * update)

        saved_stage = _maybe_save(it)
        if saved_stage:
            print(f"  iter {it:3d}  E={float(jnp.mean(E)):.6f}  → saved {saved_stage}")
        elif it % 10 == 0:
            print(
                f"  iter {it:3d}  E={float(jnp.mean(E)):.6f}"
                f"  CG {cg_info['iterations']}it"
            )

    return ckpt_paths


# ── exact distribution ────────────────────────────────────────────────────────

def _compute_p_true(rbm):
    """
    Exact |Ψ(v)|² / Z for all 2^N visible configurations.
    Returns (p_true: np.ndarray shape (2^N,), all_v: np.ndarray, config_idx: dict).
    """
    N = rbm.n_visible
    indices = np.arange(2 ** N, dtype=np.int32)
    all_v = ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.float64) * 2 - 1
    all_v_jax = jnp.asarray(all_v)
    Theta = all_v_jax @ rbm.W + rbm.b[None, :]
    log_psi2 = -(all_v_jax @ rbm.a) + jnp.sum(jnp.logaddexp(Theta, -Theta), axis=1)
    lw = log_psi2 - jnp.max(log_psi2)
    p = jnp.exp(lw)
    p = p / jnp.sum(p)
    p_np = np.asarray(p)
    config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
    return p_np, all_v, config_idx


def _compute_dtv(v_samples, p_true, all_v, config_idx):
    """DTV = ½ Σ |p_true(v) − q_emp(v)|."""
    ns = len(v_samples)
    counts = np.zeros(len(all_v))
    for row in np.asarray(v_samples).astype(int).tolist():
        idx = config_idx.get(tuple(row))
        if idx is not None:
            counts[idx] += 1
    q_emp = counts / ns
    return 0.5 * float(np.sum(np.abs(p_true - q_emp)))


# ── D-Wave sweep ──────────────────────────────────────────────────────────────

def sweep_effort(rbm, effort_values, n_samples, n_reps, beta_x, backend, solver):
    """
    For each effort value, draw samples n_reps times and compute DTV.

    backend="sa"    : DimodSampler("simulated_annealing"), effort = num_sweeps
    backend="dwave" : DimodSampler(solver),                effort = annealing_time (µs)

    Returns {effort_value (int) -> list[float] of length n_reps}.
    """
    p_true, all_v, config_idx = _compute_p_true(rbm)

    if backend == "sa":
        sampler = DimodSampler(method="simulated_annealing")
        effort_key = "num_sweeps"
        effort_label = "sweeps"
    else:
        sampler = DimodSampler(method=solver)
        effort_key = "annealing_time"
        effort_label = "µs"

    results = {}
    for val in effort_values:
        dtvs = []
        print(f"  {effort_key}={val:5d}{effort_label}  ", end="", flush=True)
        for _ in range(n_reps):
            v = sampler.sample(
                rbm, n_samples,
                config={effort_key: val, "beta_x": beta_x},
            )
            dtv = _compute_dtv(v, p_true, all_v, config_idx)
            dtvs.append(dtv)
            print(f"  {dtv:.4f}", end="", flush=True)
        results[val] = dtvs
        print()

    return results


# ── plot ──────────────────────────────────────────────────────────────────────

_COLORS = {"tfim": "#1f77b4", "heisenberg": "#d62728"}
_LABELS = {"tfim": "TFIM  h=0.5", "heisenberg": "Heisenberg  J=1, Δ=1"}


_XLABEL = {
    "sa": "num_sweeps (SA)",
    "dwave": "Annealing time (µs)",
}
_BACKEND_LABEL = {
    "sa": "dimod SA",
    "dwave": "D-Wave QPU",
}


def make_plot(all_results, stage, n_samples, beta_x, backend, output_path: Path):
    """
    all_results: {model_key: {effort_value: [dtv, ...]}}
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_key, results in all_results.items():
        xs = sorted(results.keys())
        means = np.array([np.mean(results[x]) for x in xs])
        stds = np.array([np.std(results[x]) for x in xs])
        color = _COLORS[model_key]
        ax.plot(xs, means, "o-", color=color, label=_LABELS[model_key],
                linewidth=2, markersize=5)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color=color)

    ax.set_xscale("log")
    ax.set_xlabel(_XLABEL[backend], fontsize=12)
    ax.set_ylabel("DTV", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"{_BACKEND_LABEL[backend]} sampling quality  |  stage: {stage}  |  "
        f"β={beta_x}  n_samples={n_samples}",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved → {output_path}")


# ── annealing time generation ─────────────────────────────────────────────────

def generate_annealing_times(n: int, at_min: int, at_max: int) -> list[int]:
    """
    N log-spaced integer annealing times in [at_min, at_max] µs.

    Log-spacing is appropriate because thermalization quality improves on a
    log scale: the gain from 1→2 µs is comparable to 100→200 µs.  Duplicates
    from rounding (possible when the range is small) are removed, so the
    returned list may have fewer than N entries.

    D-Wave Advantage supports 0.5–2000 µs; default range is 1–2000 µs.
    """
    times = np.unique(
        np.round(np.logspace(np.log10(at_min), np.log10(at_max), n)).astype(int)
    )
    return times.tolist()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DTV vs D-Wave annealing time")
    p.add_argument(
        "--stage",
        choices=["beginning", "mid", "trained"],
        required=True,
        help="Which RBM training stage to evaluate",
    )
    p.add_argument("--size", type=int, default=8, help="n_visible (must be ≤ 16)")
    p.add_argument(
        "--n-hidden", type=int, default=None,
        help="RBM hidden units (default: same as --size)",
    )
    p.add_argument("--n-iter", type=int, default=30, help="Training iterations")
    p.add_argument(
        "--train-samples", type=int, default=500,
        help="Samples per iteration during training",
    )
    p.add_argument("--lr", type=float, default=0.05, help="Training learning rate")
    p.add_argument("--reg", type=float, default=1e-3, help="SR regularization")
    p.add_argument(
        "--annealing-times", type=int, nargs="+", default=None,
        help=(
            "Explicit D-Wave annealing times in µs (overrides --n-annealing-times). "
            "If omitted, --n-annealing-times log-spaced values are used."
        ),
    )
    p.add_argument(
        "--n-annealing-times", type=int, default=8,
        help="Number of log-spaced annealing times between --at-min and --at-max (default 8)",
    )
    p.add_argument(
        "--at-min", type=int, default=None,
        help="Min effort value (num_sweeps for SA, µs for D-Wave). Defaults: SA=10, D-Wave=1",
    )
    p.add_argument(
        "--at-max", type=int, default=None,
        help="Max effort value (num_sweeps for SA, µs for D-Wave). Defaults: SA=10000, D-Wave=2000",
    )
    p.add_argument(
        "--n-samples", type=int, default=1000,
        help="D-Wave samples per (annealing_time, rep) call",
    )
    p.add_argument("--n-reps", type=int, default=3, help="Repetitions per effort value")
    p.add_argument("--beta-x", type=float, default=1.0, help="QUBO inverse temperature scale")
    p.add_argument(
        "--backend", default="sa", choices=["sa", "dwave"],
        help="Sampler backend: 'sa' = dimod neal (default), 'dwave' = D-Wave QPU",
    )
    p.add_argument(
        "--solver", default="zephyr", choices=["zephyr", "pegasus"],
        help="D-Wave solver (only used when --backend dwave)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Directory for plot and JSON results (default: scripts/output/)",
    )
    p.add_argument(
        "--retrain", action="store_true",
        help="Delete existing checkpoints and retrain from scratch",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    N = args.size
    if N > 16:
        raise SystemExit(f"--size {N} > 16: exact enumeration requires ≤ 16 visible spins.")

    M = args.n_hidden if args.n_hidden is not None else N

    _defaults = {"sa": (10, 10_000), "dwave": (1, 2000)}
    at_min = args.at_min if args.at_min is not None else _defaults[args.backend][0]
    at_max = args.at_max if args.at_max is not None else _defaults[args.backend][1]

    if args.annealing_times is not None:
        effort_values = args.annealing_times
    else:
        effort_values = generate_annealing_times(args.n_annealing_times, at_min, at_max)

    effort_label = "num_sweeps" if args.backend == "sa" else "annealing_time µs"
    print(f"Backend: {args.backend}  |  {effort_label}: {effort_values}")

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints" / "dtv_experiment" / f"N{N}_M{M}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "output" / "dtv_vs_annealing"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retrain:
        for p in ckpt_dir.glob("*.pkl"):
            p.unlink()
        print("Deleted existing checkpoints.")

    model_configs = {
        "tfim": {
            "ising": TransverseFieldIsing1D(N, h=0.5),
            "label": f"tfim_N{N}_h0.5_M{M}",
        },
        "heisenberg": {
            "ising": HeisenbergXXZ1D(N, J=1.0, delta=1.0),
            "label": f"heisenberg_N{N}_J1.0_M{M}",
        },
    }

    # ── Train both models ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Phase 1: training")
    print("=" * 60)
    ckpt_map = {}
    for model_key, cfg in model_configs.items():
        key = jax.random.PRNGKey(args.seed)
        rbm = FullyConnectedRBM(N, M, key)
        paths = _train(
            rbm, cfg["ising"],
            n_samples=args.train_samples,
            n_iter=args.n_iter,
            lr=args.lr,
            reg=args.reg,
            ckpt_dir=ckpt_dir,
            label=cfg["label"],
        )
        ckpt_map[model_key] = paths

    # ── D-Wave sweep ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Phase 2: D-Wave sweep  (stage={args.stage})")
    print("=" * 60)
    all_results = {}
    for model_key, cfg in model_configs.items():
        stage_path = ckpt_map[model_key].get(args.stage)
        if stage_path is None:
            raise RuntimeError(
                f"Checkpoint for stage '{args.stage}' not found for {model_key}. "
                "Run without --retrain or check training completed successfully."
            )

        key = jax.random.PRNGKey(args.seed)
        rbm = FullyConnectedRBM(N, M, key)
        _load_ckpt(rbm, stage_path)
        print(f"\n[{model_key}]  loaded {stage_path.name}")

        results = sweep_effort(
            rbm,
            effort_values=effort_values,
            n_samples=args.n_samples,
            n_reps=args.n_reps,
            beta_x=args.beta_x,
            backend=args.backend,
            solver=args.solver,
        )
        all_results[model_key] = results

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / f"dtv_{args.backend}_{args.stage}_N{N}_M{M}.json"
    with open(json_path, "w") as f:
        json.dump({
            "stage": args.stage,
            "backend": args.backend,
            "solver": args.solver if args.backend == "dwave" else None,
            "size": N,
            "n_hidden": M,
            "n_iter": args.n_iter,
            "n_samples": args.n_samples,
            "n_reps": args.n_reps,
            "beta_x": args.beta_x,
            "effort_values": effort_values,
            "results": {
                model_key: {str(v): dtvs for v, dtvs in res.items()}
                for model_key, res in all_results.items()
            },
        }, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = out_dir / f"dtv_{args.backend}_{args.stage}_N{N}_M{M}.png"
    make_plot(all_results, args.stage, args.n_samples, args.beta_x, args.backend, plot_path)


if __name__ == "__main__":
    main()

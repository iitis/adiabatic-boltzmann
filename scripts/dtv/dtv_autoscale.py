"""
D_TV and β_eff vs beta_x: does D-Wave's auto_scale nullify the beta_x
effective-temperature knob?

A reviewer pointed out that beta_x (src/sampler.py's rbm_to_ising /
dbm_to_ising) only rescales h and J *uniformly* — and D-Wave's SAPI
auto_scale=True (the default in every dwave()/dwave_parallel() call in
src/sampler.py) renormalizes the submitted h/J to make full use of the
solver's coefficient range regardless of that uniform scale. Per D-Wave's
own docs (dwave/system/temperatures.py, freezeout_effective_temperature):

    P(x) = exp(-B(s*) R E_Ising(x) / 2 kT)
    R = 1        if auto_scale=False
    R != 1       if auto_scale=True (default) — "additional scaling factors
                 must be accounted for"

So beta_x's effect on the physical energy scale seen by the annealer should
be absorbed by R whenever auto_scale=True, i.e. β_eff should be roughly flat
across beta_x. With auto_scale=False, R=1 and β_eff should track the ideal
1/beta_x line (up to a fixed offset set by the QPU's own physical
temperature) — same ideal reference already used for the classical LSB
sampler in dtv_beta_scale.py.

Uses a DWaveTopologyRBM (chain-free identity embedding on Pegasus) so any
effect observed is attributable to beta_x/auto_scale alone, not chain
breakage.

Cost: measured live QPU cost is ~0.04-0.065s per call (see
scripts/exper/dwave_matched_sweep.py); this sweep issues
len(beta_x_values) * len(auto_scale_values) * n_repeats calls, i.e. a few
seconds of QPU time total for the defaults below. A budget guard (same
pattern as scripts/exper/dwave_matched_sweep.py) still checks
src/../time.json before every call and aborts rather than silently
overspending.

OBSOLETE auto_scale=True branch: the hypothesis above has already been
confirmed (see plots/dtv_autoscale/dtv_autoscale_N8_h1.0.json,
gathered before the fix) and src/sampler.py now hardcodes
auto_scale=False in every D-Wave call, with no config path left to turn
it back on. --auto-scale-values defaults to [False] only; passing True
is accepted by argparse purely to document the pre-fix regime and raises
at execution time (see _sweep) rather than silently running as False
while being recorded as True.

Usage:
    python scripts/dtv/dtv_autoscale.py
    python scripts/dtv/dtv_autoscale.py --beta-x-values 0.2 0.5 1.0 2.0 5.0
    python scripts/dtv/dtv_autoscale.py --retrain
    python scripts/dtv/dtv_autoscale.py --plot-only
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

from model import DWaveTopologyRBM
from ising import TransverseFieldIsing1D
from sampler import ClassicalSampler, DimodSampler
from encoder import SRLinearSystem, conjugate_gradient
from helpers import read_qpu_time_ms
from plot_style import setup_style
from kl_utils import (
    all_configs_jax,
    exact_psi_sq,
    empirical_dist_jax,
    d_tv,
    finite_sampling_floor,
)


# ---------------------------------------------------------------------------
# QPU budget guard
# ---------------------------------------------------------------------------

DWAVE_TIME_FILE = Path("time.json")
DWAVE_BUDGET_MS = 60 * 60 * 1000    # absolute cumulative ceiling across all sessions ever
SESSION_BUDGET_MS = 5 * 60 * 1000   # this invocation may spend at most 5 min of NEW QPU time


def _require_qpu_time_ms() -> float:
    if not DWAVE_TIME_FILE.exists():
        raise FileNotFoundError(
            f"{DWAVE_TIME_FILE} not found — D-Wave budget tracking file missing. "
            'Create it with {"time_ms": 0} or run a D-Wave experiment first.'
        )
    return read_qpu_time_ms(DWAVE_TIME_FILE)


def _qpu_budget_exceeded(session_baseline_ms: float) -> bool:
    used = _require_qpu_time_ms()
    if used >= DWAVE_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] {used / 60_000:.2f} min used >= "
            f"{DWAVE_BUDGET_MS / 60_000:.0f} min absolute limit. Aborting."
        )
        return True
    session_spent = used - session_baseline_ms
    if session_spent >= SESSION_BUDGET_MS:
        print(
            f"\n[QPU BUDGET] {session_spent / 60_000:.2f} min spent this session >= "
            f"{SESSION_BUDGET_MS / 60_000:.0f} min session cap. Aborting."
        )
        return True
    return False


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

def _train(rbm, ising, n_samples, n_iter, lr, reg, ckpt_path: Path, label: str) -> None:
    if ckpt_path.exists():
        print(f"[{label}] checkpoint found — loading.")
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
# β_eff estimation
# ---------------------------------------------------------------------------

def _estimate_beta_eff(energies_np: np.ndarray, p_emp_np: np.ndarray,
                        beta_bounds=(0.01, 200.0)) -> float:
    """
    β_eff = argmin_β D_KL(p_S ∥ p_β)  where p_β(v) ∝ exp(-β * E(v)).

    E(v) = -2 * log|Ψ(v)|, so β=1 recovers the target |Ψ|² distribution.
    For a perfect Gibbs sampler at energy scale beta_x, we expect
    β_eff = 1/beta_x (only when auto_scale=False — that's the hypothesis
    under test here).
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
# beta_x × auto_scale sweep
# ---------------------------------------------------------------------------

def _sweep(rbm, dwave_sampler, beta_x_values, auto_scale_values, num_reads,
           annealing_time, n_repeats, N, session_baseline_ms):
    """
    For each (auto_scale, beta_x) pair issue n_repeats independent QPU calls;
    compute D_TV + β_eff per call.

    Returns
    -------
    dtv_results  : dict[auto_scale → dict[beta_x → list[float]]]  (fraction, not %)
    beta_results : dict[auto_scale → dict[beta_x → list[float]]]  (β_eff values)
    p_exact      : (2^N,) exact |Ψ(v)|² probabilities
    """
    configs = all_configs_jax(N)
    p_exact = exact_psi_sq(rbm, N)
    energies_np = np.asarray(-2.0 * jax.vmap(rbm.log_psi)(configs))

    dtv_results = {a: {} for a in auto_scale_values}
    beta_results = {a: {} for a in auto_scale_values}

    for auto_scale in auto_scale_values:
        if auto_scale:
            raise SystemExit(
                "OBSOLETE: auto_scale=True can no longer be exercised through "
                "DimodSampler — src/sampler.py now hardcodes auto_scale=False "
                "in every D-Wave call (dwave(), dwave_parallel()), with no "
                "config path to turn it back on. "
                "Requesting it here would silently run as auto_scale=False while "
                "being recorded as True, corrupting the comparison this script "
                "exists to make. The historical auto_scale=True vs False evidence "
                "is preserved in plots/dtv_autoscale/dtv_autoscale_N8_h1.0.json "
                "(gathered before the fix) — use --auto-scale-values False to "
                "calibrate beta_x going forward."
            )
        print(f"  auto_scale={auto_scale}:")
        for beta_x in beta_x_values:
            dtv_vals, beta_vals = [], []
            cfg = {
                "beta_x": beta_x,
                "auto_scale": auto_scale,
                "annealing_time": annealing_time,
            }
            for rep in range(n_repeats):
                if _qpu_budget_exceeded(session_baseline_ms):
                    raise RuntimeError(
                        "QPU budget exceeded mid-sweep — aborting rather than "
                        "silently truncating results."
                    )
                try:
                    v = dwave_sampler.sample(rbm, num_reads, config=cfg, return_hidden=False)
                except RuntimeError as e:
                    # expected hardware range violation; skip, don't fake a value
                    print(f"    [SKIP] auto_scale={auto_scale} beta_x={beta_x} "
                          f"repeat={rep}: {e}")
                    continue
                v = jnp.asarray(v, dtype=jnp.float64)
                p_emp = empirical_dist_jax(v, N)
                dtv_vals.append(float(d_tv(p_exact, p_emp)))
                beta_vals.append(_estimate_beta_eff(energies_np, np.asarray(p_emp)))

            if not dtv_vals:
                print(f"    beta_x={beta_x:.3f}  auto_scale={auto_scale}: "
                      f"all {n_repeats} repeats failed (hardware range exceeded) — "
                      f"DROPPED from results, not filled with a placeholder.")
                continue

            dtv_results[auto_scale][beta_x] = dtv_vals
            beta_results[auto_scale][beta_x] = beta_vals
            ideal = 1.0 / beta_x
            print(f"    beta_x={beta_x:.3f}  "
                  f"D_TV={np.mean(dtv_vals)*100:.1f}%  "
                  f"β_eff={np.mean(beta_vals):.2f}  "
                  f"(ideal if auto_scale=False: {ideal:.2f})"
                  + (f"  [{len(dtv_vals)}/{n_repeats} repeats]" if len(dtv_vals) < n_repeats else ""))

    return dtv_results, beta_results, p_exact


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_AUTOSCALE_COLORS = {True: "#d62728", False: "#1f77b4"}
_IDEAL_COLOR = "#888888"
_FLOOR_COLOR = "#333333"
_FIG_W, _FIG_H = 5.0, 3.1


def _robust_mean_std(vals, label=""):
    """
    Mean/std after dropping draws that saturate the bounded-scalar KL-argmin
    fit in _estimate_beta_eff (same failure mode documented for the CEM
    validation figure: the fit occasionally lands on a spurious optimum for
    a near-degenerate empirical distribution, unrelated to the sampler's
    actual physical behaviour). Flagged explicitly, never silently averaged in.
    """
    vals = np.asarray(vals, dtype=float)
    med = np.median(vals)
    keep = np.abs(vals - med) <= 10 * max(med, 1e-9)
    if not np.all(keep):
        print(f"    [OUTLIER] {label}: dropping {vals[~keep].tolist()} "
              f"(median of remaining: {med:.3f}) — KL-argmin fit saturated.")
    kept = vals[keep]
    return float(np.mean(kept)), float(np.std(kept))


def _plot_dtv_panel(ax, beta_x_values, dtv_results, auto_scale_values, floor_pct):
    for auto_scale in auto_scale_values:
        color = _AUTOSCALE_COLORS[auto_scale]
        present = [bx for bx in beta_x_values if bx in dtv_results[auto_scale]]
        if not present:
            continue
        means = np.array([np.mean(dtv_results[auto_scale][bx]) for bx in present]) * 100
        stds = np.array([np.std(dtv_results[auto_scale][bx]) for bx in present]) * 100
        ax.plot(present, means, "o-", color=color,
                label=f"auto_scale={auto_scale}", linewidth=2, markersize=5)
        ax.fill_between(present, means - stds, means + stds,
                         alpha=0.18, color=color)
    ax.axhline(floor_pct, color=_FLOOR_COLOR, linestyle="--", linewidth=1.5,
               label="sampling floor")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta_x$")
    ax.set_ylabel(r"$D_\mathrm{TV}$ (\%)")
    ax.legend(fontsize=10)


def _plot_beta_panel(ax, beta_x_values, beta_results, auto_scale_values):
    ideal = np.array([1.0 / bx for bx in beta_x_values])
    ax.plot(beta_x_values, ideal, "--", color=_IDEAL_COLOR, linewidth=1.5,
            label=r"ideal $1/\beta_x$ (auto\_scale=False)")

    for auto_scale in auto_scale_values:
        color = _AUTOSCALE_COLORS[auto_scale]
        present = [bx for bx in beta_x_values if bx in beta_results[auto_scale]]
        if not present:
            continue
        stats = [_robust_mean_std(beta_results[auto_scale][bx],
                                   label=f"auto_scale={auto_scale} beta_x={bx}")
                 for bx in present]
        means = np.array([m for m, _ in stats])
        stds = np.array([s for _, s in stats])
        ax.plot(present, means, "o-", color=color,
                label=f"auto_scale={auto_scale}", linewidth=2, markersize=5)
        ax.fill_between(present, means - stds, means + stds,
                        alpha=0.18, color=color)

    ax.axhline(1.0, color=_FLOOR_COLOR, linestyle=":", linewidth=1.2,
               label=r"$\beta_{\mathrm{eff}}=1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_x$")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}$")
    ax.legend(fontsize=10)


def _make_plots(beta_x_values, dtv_results, beta_results, auto_scale_values, floor, N, h, n_reads, out_dir):
    setup_style(fontsize=14, scale=2.5)

    fig1, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _plot_dtv_panel(ax1, beta_x_values, dtv_results, auto_scale_values, floor * 100)
    fig1.tight_layout()
    out1 = out_dir / f"dtv_autoscale_dtv_N{N}.pdf"
    fig1.savefig(out1, bbox_inches="tight")
    print(f"  plot → {out1}")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _plot_beta_panel(ax2, beta_x_values, beta_results, auto_scale_values)
    fig2.tight_layout()
    out2 = out_dir / f"dtv_autoscale_beta_N{N}.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"  plot → {out2}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _save_json(beta_x_values, auto_scale_values, dtv_results, beta_results,
               floor, N, h, num_reads, n_repeats, annealing_time, path: Path):
    payload = {
        "N": N, "h": h, "num_reads": num_reads, "n_repeats": n_repeats,
        "annealing_time": annealing_time,
        "beta_x_values": beta_x_values,
        "auto_scale_values": auto_scale_values,
        "floor": floor,
        "dtv": {str(a): {str(bx): v for bx, v in dtv_results[a].items()}
                for a in auto_scale_values},
        "beta": {str(a): {str(bx): v for bx, v in beta_results[a].items()}
                 for a in auto_scale_values},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  results → {path}")


def _load_json(path: Path):
    with open(path) as f:
        raw = json.load(f)
    beta_x_values = raw["beta_x_values"]
    auto_scale_values = raw["auto_scale_values"]
    dtv_results = {a: {float(bx): v for bx, v in raw["dtv"][str(a)].items()}
                   for a in auto_scale_values}
    beta_results = {a: {float(bx): v for bx, v in raw["beta"][str(a)].items()}
                    for a in auto_scale_values}
    return (beta_x_values, auto_scale_values, dtv_results, beta_results,
            raw["floor"], raw["N"], raw["h"], raw["num_reads"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=8,
                   help="N (visible qubits); must fit a chain-free Pegasus biclique.")
    p.add_argument("--n-hidden", type=int, default=None, help="default: N")
    p.add_argument("--h", type=float, default=1.0, help="TFIM transverse field")
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument(
        "--beta-x-values", type=float, nargs="+",
        default=[0.2, 0.5, 1.0, 2.0, 5.0],
    )
    p.add_argument(
        "--auto-scale-values", type=lambda s: s.lower() == "true", nargs="+",
        default=[False],
        help="OBSOLETE: 'True' is kept only to document the pre-fix regime and "
             "will raise at execution time -- src/sampler.py now hardcodes "
             "auto_scale=False with no config path to re-enable it. Default: False.",
    )
    p.add_argument("--num-reads", type=int, default=500,
                   help="QPU reads per (auto_scale, beta_x, repeat) call")
    p.add_argument("--n-repeats", type=int, default=3,
                   help="independent QPU calls per (auto_scale, beta_x) point")
    p.add_argument("--annealing-time", type=float, default=20.0, help="microseconds")
    p.add_argument("--topology-seed", type=int, default=42,
                   help="seed for DWaveTopologyRBM's chain-free subgraph selection")
    p.add_argument("--seed", type=int, default=42, help="RBM init / training seed")
    p.add_argument("--floor-trials", type=int, default=20)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    N = args.size
    M = args.n_hidden if args.n_hidden is not None else N

    if N > 16:
        raise SystemExit(f"--size {N} > 16: exact enumeration requires N ≤ 16.")

    repo_root = _ROOT
    out_dir = Path(args.output_dir) if args.output_dir else repo_root / "plots" / "dtv_autoscale"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"dtv_autoscale_N{N}_h{args.h}.json"

    if args.plot_only:
        if not json_path.exists():
            raise SystemExit(f"No results at {json_path}. Run without --plot-only first.")
        (beta_x_values, auto_scale_values, dtv_results, beta_results,
         floor, N, h, num_reads) = _load_json(json_path)
        _make_plots(beta_x_values, dtv_results, beta_results, auto_scale_values,
                    floor, N, h, num_reads, out_dir)
        return

    ckpt_dir = repo_root / "checkpoints" / "dtv_autoscale" / f"N{N}_M{M}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_label = f"tfim_N{N}_h{args.h}_M{M}"
    ckpt_path = ckpt_dir / f"{ckpt_label}_trained.pkl"
    if args.retrain and ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\n{'='*60}\nDWaveTopologyRBM (pegasus)  N={N}  M={M}  h={args.h}\n{'='*60}")
    rbm = DWaveTopologyRBM(
        N, M, jax.random.PRNGKey(args.seed),
        solver="pegasus", seed=args.topology_seed, live=True,
    )
    print(f"  sparsity={rbm.sparsity():.3f}  n_parameters={rbm.n_parameters()}")
    ising = TransverseFieldIsing1D(N, args.h)
    _train(rbm, ising, args.train_samples, args.n_iter, args.lr, args.reg,
           ckpt_path, ckpt_label)

    p_exact = exact_psi_sq(rbm, N)
    floor = finite_sampling_floor(p_exact, args.num_reads, args.floor_trials)
    print(f"  finite-sampling floor: {floor*100:.2f}%")

    session_baseline_ms = _require_qpu_time_ms()
    print(f"[QPU BUDGET] baseline: {session_baseline_ms/60_000:.2f} min already used "
          f"(absolute cap {DWAVE_BUDGET_MS/60_000:.0f} min, session cap "
          f"+{SESSION_BUDGET_MS/60_000:.0f} min for this run)")

    dwave_sampler = DimodSampler("pegasus")
    auto_scale_values = args.auto_scale_values
    dtv_results, beta_results, _ = _sweep(
        rbm, dwave_sampler, args.beta_x_values, auto_scale_values,
        args.num_reads, args.annealing_time, args.n_repeats, N,
        session_baseline_ms,
    )

    used_after = read_qpu_time_ms(DWAVE_TIME_FILE)
    print(f"[QPU BUDGET] spent this run: {(used_after - session_baseline_ms)/1000:.3f} s")

    _save_json(args.beta_x_values, auto_scale_values, dtv_results, beta_results,
               floor, N, args.h, args.num_reads, args.n_repeats,
               args.annealing_time, json_path)
    _make_plots(args.beta_x_values, dtv_results, beta_results, auto_scale_values,
                floor, N, args.h, args.num_reads, out_dir)


if __name__ == "__main__":
    main()

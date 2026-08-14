#!/usr/bin/env python3
"""
Validate CEM's beta_eff estimate against an independent, exact ground-truth
estimate, across system size, field, LSB energy scale (beta_x), and training
checkpoint (early/mid/late).

For each (N, h) TFIM instance we train an RBM with SR, saving checkpoints at
three points during training. At each checkpoint we sweep beta_x, draw LSB
joint (v, h) samples, and on each sample batch compute two independent
estimates of the sampler's effective inverse temperature:

  - beta_ground_truth : argmin_beta KL(p_empirical(v) || |Psi(v)|^{2*beta}),
                         i.e. exact KL-minimisation against the *exact*
                         visible-marginal Boltzmann distribution (same
                         definition already used for the dtv_beta_scale
                         reference line in the report).
  - beta_cem          : encoder.estimate_beta_eff_cem(V, H, rbm), the CEM
                         tanh-matching estimate actually used during training.

Results are dumped to JSON; see plot_cem_validation.py for the figure.

Usage:
    python scripts/exper/cem_validation_sweep.py
    python scripts/exper/cem_validation_sweep.py --retrain
    python scripts/exper/cem_validation_sweep.py --smoke-test   # tiny config, for a quick check
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

from model import FullyConnectedRBM  # noqa: E402
from ising import TransverseFieldIsing1D  # noqa: E402
from sampler import ClassicalSampler  # noqa: E402
from encoder import SRLinearSystem, conjugate_gradient, estimate_beta_eff_cem  # noqa: E402
from kl_utils import all_configs_jax, empirical_dist_jax  # noqa: E402

_CKPT_DIR = _ROOT / "checkpoints" / "cem_validation"
_OUT_DIR = _ROOT / "plots" / "cem_validation"

CKPT_FRACTIONS = {"early": 0.1, "mid": 0.5, "late": 1.0}
LR, REG = 0.05, 1e-3
LSB_CFG_BASE = {"lsb_steps": 1000, "lsb_delta": 0.1, "lsb_gamma": 0.1, "lsb_sigma": 1.0}


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
# Training with early/mid/late checkpoints
# ---------------------------------------------------------------------------

def _train_with_checkpoints(rbm, ising, n_iter, train_samples, ckpt_dir, label,
                             retrain=False) -> dict:
    paths = {tag: ckpt_dir / f"{label}_{tag}.pkl" for tag in CKPT_FRACTIONS}
    if not retrain and all(p.exists() for p in paths.values()):
        print(f"[{label}] checkpoints found — loading.")
        return paths

    print(f"[{label}] training N={rbm.n_visible} M={rbm.n_hidden}  {n_iter} iters ...")
    sampler = ClassicalSampler("simulated_annealing", n_warmup=100, n_sweeps=20)
    milestone_iters = {tag: max(1, round(frac * n_iter)) for tag, frac in CKPT_FRACTIONS.items()}
    saved = set()

    for it in range(1, n_iter + 1):
        V = jnp.asarray(sampler.sample(rbm, train_samples), dtype=jnp.float64)
        E = ising.local_energy_batch(V, rbm)
        if not bool(jnp.all(jnp.isfinite(E))):
            raise RuntimeError(f"[{label}] NaN/inf at iteration {it}.")

        Theta = V @ rbm.W + rbm.b[None, :]
        TanH = jnp.tanh(Theta)
        sr = SRLinearSystem(V, TanH, E, REG)
        x, _ = conjugate_gradient(sr.matvec, sr.force, tol=1e-8, maxiter=200)
        xa, xb, xW = sr.unpack(x)
        update = jnp.concatenate([xa.ravel(), xb.ravel(), xW.T.ravel()])
        rbm.set_weights(rbm.get_weights() - LR * update)

        for tag, mit in milestone_iters.items():
            if it == mit and tag not in saved:
                _save_ckpt(rbm, paths[tag])
                saved.add(tag)
                print(f"  [{label}] checkpoint '{tag}' saved at iter {it}")

    return paths


# ---------------------------------------------------------------------------
# Ground-truth beta_eff: KL-argmin against the exact visible marginal
# ---------------------------------------------------------------------------

def _ground_truth_beta(rbm, N, samples_v, beta_bounds=(0.01, 200.0)) -> float:
    """
    beta_eff = argmin_beta D_KL(p_empirical(v) || p_beta(v)),
    p_beta(v) ~ exp(-beta*a.v) * prod_j 2cosh(beta*Theta_j(v)) -- the visible
    marginal of the JOINTLY beta-rescaled Boltzmann distribution, i.e. the
    family the sampler actually realises when a, b, W are uniformly rescaled
    by beta before sampling (report.tex L317-320: E_in = alpha*E_theta).

    NOT |Psi(v)|^{2*beta} (raising the beta=1 marginal to a power beta) --
    that family coincides with this one only at beta=1, since
    prod_j (2cosh Theta_j)^beta != prod_j 2cosh(beta*Theta_j) in general.
    Audit finding F1: this mismatch, not CEM's own error, was driving most of
    the reported RMSE and the "unbiased only near beta_eff=1" pattern.
    """
    configs = all_configs_jax(N)
    a_v = np.asarray(configs @ rbm.a)                     # (2^N,)
    theta = np.asarray(configs @ rbm.W + rbm.b[None, :])  # (2^N, M)
    p_emp = np.asarray(empirical_dist_jax(samples_v, N))

    def _logsumexp(a):
        c = float(np.max(a))
        return c + float(np.log(np.sum(np.exp(a - c))))

    def objective(beta):
        log_unnorm = -beta * a_v + np.sum(np.log(2.0 * np.cosh(beta * theta)), axis=1)
        log_b = log_unnorm - _logsumexp(log_unnorm)
        mask = p_emp > 0
        return float(np.sum(p_emp[mask] * (np.log(p_emp[mask]) - log_b[mask])))

    result = minimize_scalar(objective, bounds=beta_bounds, method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(n_values, h_values, beta_x_values, n_seeds, n_iter, train_samples,
              n_samples, retrain, out_path):
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    sampler = ClassicalSampler("lsb")
    records = []
    draw_id = 0  # deterministic seed counter

    for N in n_values:
        M = N
        for h in h_values:
            label = f"tfim_N{N}_h{h}"
            print(f"\n{'='*60}\nN={N}  h={h}\n{'='*60}")
            rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(0))
            ising = TransverseFieldIsing1D(N, h)
            ckpt_paths = _train_with_checkpoints(
                rbm, ising, n_iter, train_samples, _CKPT_DIR, label, retrain
            )

            for ckpt_tag, ckpt_path in ckpt_paths.items():
                _load_ckpt(rbm, ckpt_path)

                for beta_x in beta_x_values:
                    cfg = {"beta_x": beta_x, **LSB_CFG_BASE}
                    for seed in range(n_seeds):
                        sampler._key = jax.random.PRNGKey(draw_id)
                        draw_id += 1

                        v, h_samp = sampler.sample(
                            rbm, n_samples, config=cfg,
                            return_hidden=True, return_jax=True,
                        )
                        beta_gt = _ground_truth_beta(rbm, N, v)
                        beta_cem = estimate_beta_eff_cem(
                            v, jnp.asarray(h_samp, dtype=jnp.float64), rbm
                        )

                        records.append({
                            "N": N, "h": h, "checkpoint": ckpt_tag,
                            "beta_x": beta_x, "seed": seed,
                            "beta_ground_truth": beta_gt,
                            "beta_cem": beta_cem,
                        })

                    last = records[-n_seeds:]
                    gt_mean = np.mean([r["beta_ground_truth"] for r in last])
                    cem_mean = np.mean([r["beta_cem"] for r in last])
                    print(f"  [{ckpt_tag}] beta_x={beta_x:.2f}  "
                          f"beta_gt={gt_mean:.3f}  beta_cem={cem_mean:.3f}  "
                          f"bias={cem_mean - gt_mean:+.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records -> {out_path}")
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12])
    p.add_argument("--h-values", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--beta-x-values", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--train-samples", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny config (N=8 only, 1 h, 1 beta_x, 1 seed, short training) "
                        "to verify the script runs end-to-end.")
    p.add_argument("--output", default=str(_OUT_DIR / "cem_validation_results.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        run_sweep(
            n_values=[8], h_values=[1.0], beta_x_values=[1.0],
            n_seeds=1, n_iter=20, train_samples=200, n_samples=300,
            retrain=True, out_path=_OUT_DIR / "cem_validation_smoke_test.json",
        )
        return

    run_sweep(
        n_values=args.sizes, h_values=args.h_values, beta_x_values=args.beta_x_values,
        n_seeds=args.n_seeds, n_iter=args.n_iter, train_samples=args.train_samples,
        n_samples=args.n_samples, retrain=args.retrain, out_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

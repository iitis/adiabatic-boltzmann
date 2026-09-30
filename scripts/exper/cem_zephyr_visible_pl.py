"""Visible-marginal temperature of QPU samples via single-spin pseudo-likelihood.

Model p_s(v) ∝ |Psi(v)|^(2s). For each sample and site, p(v_i | v_-i) = 1/(1+exp(s·Δ_i)),
Δ_i = log|Psi|^2(v with i flipped) - log|Psi|^2(v). s=1: samples faithful to |Psi|^2;
s>1: too cold. Uses the final-iteration samples (lastV) and final params of each
cem_zephyr_protocol_qpu run.
"""
import glob, json, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parent.parent.parent


def flip_deltas(a, b, W, V):
    """Δ[n, i] = log|Psi|^2(V_n with spin i flipped) - log|Psi|^2(V_n) for p ∝ exp(-a·v) Π 2cosh(θ)."""
    theta = V @ W + b                                    # (ns, M)
    theta_f = theta[:, None, :] - 2 * V[:, :, None] * W[None, :, :]   # (ns, N, M)
    lc = lambda x: jnp.logaddexp(x, -x)
    return 2 * a[None, :] * V + jnp.sum(lc(theta_f) - lc(theta)[:, None, :], axis=2)


def fit_s(D):
    D = np.asarray(D).ravel()
    nll = lambda s: np.sum(np.logaddexp(0.0, s * D))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/cem_zephyr_protocol_qpu/*_N*.jsonl"
    for f in sorted(glob.glob(str(REPO / pat))):
        for l in open(f):
            r = json.loads(l)
            a, b, W = (jnp.asarray(r[k]) for k in ("a", "b", "W"))
            V = jnp.asarray(r["lastV"], dtype=jnp.float64)
            s = fit_s(flip_deltas(a, b, W, V))
            err = (r["E_true"] - r["exact"]) / abs(r["exact"]) * 100
            print(f"{r['tag']} N={r['N']} seed={r['seed']:2d}: s_PL={s:5.2f}  true err={err:+6.2f}%  "
                  f"dw samples={np.mean(r['sample_dw'][-1]):.2f} vs |Psi|^2 {r['dw']:.2f}  beta_x={r['beta_x'][-1]:.2f}")

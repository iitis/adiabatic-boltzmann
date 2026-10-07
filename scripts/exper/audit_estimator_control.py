"""Controlled accuracy test of both estimators with exact samples (audit response).

Networks: N=M=8, h=0.5, the stored cem_validation checkpoints (early, mid, late) and an initial
network (a=b=0, W ~ N(0, 0.01^2)). Estimators are the functions used in the hardware runs:
  CEM: encoder.estimate_beta_eff_cem (bounded least squares, beta in [0.01, 50])
  PL : cem_zephyr_visible_pl.fit_s (bounded pseudo-likelihood, s in [0.05, 20])
CEM: 200 exact draws from p_beta(v,u) (visible marginal by enumeration, then u|v), fit with the
unscaled parameters. PL: 200 exact draws from p_s(v) ~ pi(v)^s. 12 repetitions per value.

    python scripts/exper/audit_estimator_control.py   # writes results/audit_estimator_control.json
"""
import itertools, json, pickle, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(Path(__file__).parent))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from encoder import estimate_beta_eff_cem
from cem_zephyr_visible_pl import flip_deltas, fit_s


class P:  # minimal parameter holder with the attributes estimate_beta_eff_cem reads
    def __init__(self, a, b, W): self.a, self.b, self.W = (jnp.asarray(x) for x in (a, b, W))


def networks():
    rng = np.random.default_rng(0)
    yield "initial", np.zeros(8), np.zeros(8), rng.normal(0, 0.01, (8, 8))
    for st in ("early", "mid", "late"):
        d = pickle.load(open(REPO / f"checkpoints/cem_validation/tfim_N8_h0.5_{st}.pkl", "rb"))
        yield st, np.array(d["a"]), np.array(d["b"]), np.array(d["W"])


def main():
    rng = np.random.default_rng(20261007)
    vs = np.array(list(itertools.product([-1.0, 1.0], repeat=8)))
    out = []
    for name, a, b, W in networks():
        p = P(a, b, W)
        lpi = -vs @ a + np.logaddexp(vs @ W + b, -(vs @ W + b)).sum(1)
        for bt in (0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
            phi = bt * (vs @ W + b)
            lp = -bt * vs @ a + np.logaddexp(phi, -phi).sum(1); pr = np.exp(lp - logsumexp(lp))
            for rep in range(12):
                v = vs[rng.choice(256, 200, p=pr)]
                u = np.where(rng.random((200, 8)) < (1 + np.tanh(bt * (v @ W + b))) / 2, 1.0, -1.0)
                out.append(dict(est="cem", net=name, true=bt, rep=rep, fit=estimate_beta_eff_cem(jnp.asarray(v), jnp.asarray(u), p)))
        for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
            pr = np.exp(s * lpi - logsumexp(s * lpi))
            for rep in range(12):
                v = jnp.asarray(vs[rng.choice(256, 200, p=pr)])
                out.append(dict(est="pl", net=name, true=s, rep=rep, fit=float(fit_s(flip_deltas(p.a, p.b, p.W, v)))))
        print(name, flush=True)
    (REPO / "results/audit_estimator_control.json").write_text(json.dumps(out))
    for est in ("cem", "pl"):
        for name in ("initial", "early", "mid", "late"):
            R = [r for r in out if r["est"] == est and r["net"] == name]
            rel = np.array([abs(r["fit"] / r["true"] - 1) for r in R])
            fail = np.sum([(r["fit"] / r["true"] > 3) or (r["fit"] / r["true"] < 1 / 3) for r in R])
            print(f"{est} {name:7s} median rel err {np.median(rel)*100:5.1f}%  failures {fail}/{len(R)}")


if __name__ == "__main__":
    main()

"""Two-conditional single-temperature test (audit response).

Under a joint Boltzmann distribution p_beta(v,u) of the programmed RBM energy, both RBM conditionals
    E[u_j|v] = tanh(beta (b_j + sum_i W_ij v_i)),   E[v_i|u] = tanh(beta (-a_i + sum_j W_ij u_j))
hold with the same beta. We fit beta to each by conditional maximum likelihood (convex, monotone
score) and use T = log(beta_u|v / beta_v|u). Its null distribution is simulated at the geometric
mean of the two fits with the same parameters and batch size: exact draws (enumeration of the visible
marginal, then u|v) for N<=16, independent block-Gibbs chains for N>=32.

Sources: P4 training calls (raw npz with V, H, a, b, W per call) and frozen QPU re-sampling.
    python scripts/exper/audit_two_conditional.py   # appends to results/audit_two_conditional.jsonl, resumes
"""
import functools, glob, itertools, json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
rng = np.random.default_rng(20261006)


def cmle(x, f):
    g = lambda b: np.sum(f * (x - np.tanh(b * f)))
    if g(1e-3) < 0:
        return 1e-3
    if g(50.0) > 0:
        return 50.0
    return brentq(g, 1e-3, 50.0)


def stat(v, u, a, b, W):
    bu, bv = cmle(u, v @ W + b), cmle(v, -a + u @ W.T)
    return bu, bv, np.log(bu / bv)


def pm(m):
    return np.where(rng.random(m.shape) < (1 + m) / 2, 1.0, -1.0)


@functools.partial(jax.jit, static_argnums=(5,))
def _gibbs(key, a, b, W, beta, n, sweeps=500):
    """n independent block-Gibbs chains of p_beta(v,u) from random starts (GPU when available)."""
    s = lambda k, m: jnp.where(jax.random.uniform(k, m.shape) < (1 + m) / 2, 1.0, -1.0)
    key, k = jax.random.split(key)
    v = s(k, jnp.zeros((n, a.shape[0])))
    def sweep(i, c):
        v, key = c
        key, k1, k2 = jax.random.split(key, 3)
        u = s(k1, jnp.tanh(beta * (v @ W + b)))
        return s(k2, jnp.tanh(beta * (-a + u @ W.T))), key
    v, key = jax.lax.fori_loop(0, sweeps, sweep, (v, key))
    return v, s(jax.random.split(key)[0], jnp.tanh(beta * (v @ W + b)))


def null_draws(a, b, W, beta, n, reps):
    """reps batches of n joint samples from p_beta(v,u)."""
    N = len(a)
    if N <= 16:
        vs = np.array(list(itertools.product([-1.0, 1.0], repeat=N)))
        phi = beta * (vs @ W + b)
        lp = -beta * vs @ a + np.logaddexp(phi, -phi).sum(1)
        p = np.exp(lp - logsumexp(lp))
        for _ in range(reps):
            v = vs[rng.choice(len(vs), n, p=p)]
            yield v, pm(np.tanh(beta * (v @ W + b)))
    else:
        v, u = (np.asarray(x, dtype=float) for x in
                _gibbs(jax.random.PRNGKey(int(rng.integers(2**31))), *(jnp.asarray(x) for x in (a, b, W)), beta, n * reps))
        for k in range(reps):
            yield v[k * n:(k + 1) * n], u[k * n:(k + 1) * n]


def test(v, u, a, b, W, reps):
    bu, bv, T = stat(v, u, a, b, W)
    null = np.array([stat(vv, uu, a, b, W)[2] for vv, uu in null_draws(a, b, W, np.sqrt(bu * bv), len(v), reps)])
    p = (1 + np.sum(np.abs(null) >= abs(T))) / (reps + 1)  # two-sided Monte Carlo p-value
    return dict(beta_uv=bu, beta_vu=bv, T=T, null_mean=float(null.mean()), null_sd=float(null.std()), p=float(p))


OUT = RES / "audit_two_conditional.jsonl"


def emit(rec, done):
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
    for dev in ["pegasus", "zephyr"]:
        for fn in sorted(glob.glob(str(RES / f"cem_{dev}_protocol_qpu/raw/P4cemPL_N*_seed*.npz"))):
            d = np.load(fn)
            N, seed = d["V"].shape[2], int(Path(fn).stem.split("seed")[1])
            key = f"training_{dev}_N{N}_seed{seed}"
            if key in done:
                continue
            for t in range(98, 103):  # last five training calls
                r = test(d["V"][t].astype(float), d["H"][t].astype(float), d["a"][t], d["b"][t], d["W"][t], 200)
                emit(dict(key=key, src="training", device=dev, N=N, seed=seed, call=t, **r), done)
            print(dev, N, seed, flush=True)
    for fn in sorted(glob.glob(str(RES / "audit_frozen_qpu/*.npz"))):
        key = "frozen_" + Path(fn).stem
        if key in done:
            continue
        d = np.load(fn); job = json.loads(str(d["job"]))
        V, H = d["V"].astype(float), d["H"].astype(float)
        r = test(V, H, d["a"], d["b"], d["W"], 200)
        emit(dict(key=key, src="frozen", **job, cbf=float(d["cbf"].mean()),
                  chain_strength=float(d["chain_strength"]), n_reads=len(V), **r), done)
        print(Path(fn).stem, f"T={r['T']:+.3f} p={r['p']:.3f}", flush=True)


if __name__ == "__main__":
    main()

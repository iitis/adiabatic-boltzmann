"""Independent energy evaluation with error bars and convergence diagnostics (audit response).

Metropolis on |Psi|^2 with sequential single-spin sweeps plus one global spin-flip proposal
per sweep (moves between the +-m modes that single flips cannot cross at large N). Chains
start from three groups: uniform random, all +1, all -1. Reported per network:
  E, se      mean local energy and standard error over chains (chains are the repetitions)
  E_groups   mean per start group; rhat  Gelman-Rubin over chains on the second half
  dw, absm   domain walls and |m|; Ezz, Ex  diagonal and transverse parts of E
Per-chain means are stored so other statistics can be recomputed.

  python scripts/exper/audit_evaluate.py SRC OUT.jsonl     # SRC: jsonl with a, b, W per line
  python scripts/exper/audit_evaluate.py --check            # N=8,16 against enumeration
"""
import argparse, itertools, json, sys, zlib
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def log_p(V, a, b, W):
    """log |Psi|^2 up to a constant: -a.v + sum_j log 2cosh(phi_j)."""
    phi = V @ W + b
    return -V @ a + jnp.sum(jnp.logaddexp(phi, -phi), axis=-1)


def local_energy(V, a, b, W, h):
    """Eq. (A1): -sum v_i v_{i+1} - h sum_i A(v^(i))/A(v), parts returned separately."""
    ezz = -jnp.sum(V * jnp.roll(V, -1, axis=1), axis=1)
    lp = log_p(V, a, b, W)
    N = V.shape[1]
    flips = jax.vmap(lambda i: log_p(V.at[:, i].multiply(-1.0), a, b, W))(jnp.arange(N))  # (N, chains)
    ex = -h * jnp.sum(jnp.exp(0.5 * (flips - lp[None, :])), axis=0)
    return ezz, ex


def evaluate(a, b, W, h=0.5, n_chains=384, burn=1000, n_meas=100, thin=10, seed=0, global_flip=True):
    a, b, W = (jnp.asarray(x, dtype=jnp.float64) for x in (a, b, W))
    N = W.shape[0]
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    g = n_chains // 3
    V = jnp.concatenate([jnp.where(jax.random.bernoulli(k, 0.5, (g, N)), 1.0, -1.0),
                         jnp.ones((g, N)), -jnp.ones((n_chains - 2 * g, N))])
    group = np.repeat([0, 1, 2], [g, g, n_chains - 2 * g])

    @jax.jit
    def sweep(V, key):
        def step(carry, i):
            V, lp, key = carry
            key, k = jax.random.split(key)
            Vp = V.at[:, i].multiply(-1.0)
            lpp = log_p(Vp, a, b, W)
            acc = jnp.log(jax.random.uniform(k, (V.shape[0],))) < lpp - lp
            return (jnp.where(acc[:, None], Vp, V), jnp.where(acc, lpp, lp), key), None
        lp = log_p(V, a, b, W)
        (V, lp, key), _ = jax.lax.scan(step, (V, lp, key), jnp.arange(N))
        key, k = jax.random.split(key)
        lpg = log_p(-V, a, b, W)  # global flip
        acc = (jnp.log(jax.random.uniform(k, (V.shape[0],))) < lpg - lp) & global_flip
        return jnp.where(acc[:, None], -V, V), key

    for _ in range(burn):
        V, key = sweep(V, key)
    ezz, ex, dw, am = [], [], [], []
    for _ in range(n_meas):
        for _ in range(thin):
            V, key = sweep(V, key)
        z, x = local_energy(V, a, b, W, h)
        ezz.append(np.asarray(z)); ex.append(np.asarray(x))
        Vn = np.asarray(V)
        dw.append(np.sum(Vn != np.roll(Vn, -1, 1), 1)); am.append(np.abs(Vn.mean(1)))
    ezz, ex, dw, am = map(np.array, (ezz, ex, dw, am))  # (n_meas, chains)
    E = ezz + ex
    cm = E.mean(0)  # per-chain means
    half = E[n_meas // 2:]
    W_ = half.var(0, ddof=1).mean(); B_ = half.shape[0] * half.mean(0).var(ddof=1)
    rhat = float(np.sqrt(((half.shape[0] - 1) / half.shape[0] * W_ + B_ / half.shape[0]) / W_))
    return dict(N=N, E=float(cm.mean()), se=float(cm.std(ddof=1) / np.sqrt(n_chains)),
                E_groups=[float(cm[group == k].mean()) for k in range(3)],
                se_groups=[float(cm[group == k].std(ddof=1) / np.sqrt((group == k).sum())) for k in range(3)],
                rhat=rhat, Ezz=float(ezz.mean()), Ex=float(ex.mean()), dw=float(dw.mean()), absm=float(am.mean()),
                chain_means=cm.tolist(), settings=dict(n_chains=n_chains, burn=burn, n_meas=n_meas, thin=thin, seed=seed, global_flip=global_flip))


def exact(a, b, W, h=0.5):
    N = len(a)
    V = jnp.array(list(itertools.product([-1.0, 1.0], repeat=N)))
    lp = log_p(V, *(jnp.asarray(x) for x in (a, b, W)))
    p = np.exp(np.asarray(lp - jax.scipy.special.logsumexp(lp)))
    z, x = local_energy(V, *(jnp.asarray(x) for x in (a, b, W)), h)
    return float(p @ np.asarray(z + x))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--seed-offset", type=int, default=0, help="added to the per-network seed (independent repeat)")
    args = ap.parse_args()
    if args.check:
        for N in (8, 16):
            for line in list(open(REPO / f"results/cem_pegasus_protocol_qpu/P4cemPL_N{N}.jsonl"))[:3]:
                r = json.loads(line)
                ex = exact(r["a"], r["b"], r["W"])
                ev = evaluate(r["a"], r["b"], r["W"])
                print(f"N={N} seed={r['seed']}: exact {ex:.6f}  MC {ev['E']:.6f} +- {ev['se']:.6f}  "
                      f"z={(ev['E']-ex)/ev['se']:+.2f}  rhat={ev['rhat']:.3f}  groups={np.round(ev['E_groups'],5)}", flush=True)
        sys.exit()
    done = {(r["tag"], r["N"], r["seed"]) for r in map(json.loads, open(args.out))} if Path(args.out).exists() else set()
    for line in open(args.src):
        r = json.loads(line)
        if (r.get("tag"), r["N"], r["seed"]) in done:
            continue
        ev = evaluate(r["a"], r["b"], r["W"], burn=args.burn,
                      seed=(zlib.crc32(f"{args.src}|{r['seed']}".encode()) + args.seed_offset) % 2**31)  # own seed per network
        rec = dict(tag=r.get("tag"), device=r.get("device"), seed=r["seed"], exact=r["exact"], **ev)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['tag']} N={rec['N']} seed={rec['seed']}: err/N={(ev['E']-r['exact'])/r['N']:.5f} +- {ev['se']/r['N']:.5f} "
              f"rhat={ev['rhat']:.3f}", flush=True)

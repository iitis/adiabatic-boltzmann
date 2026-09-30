"""Unbiased <H>_Psi of saved RBM checkpoints via independent Metropolis on |Psi|^2."""
import gzip, itertools, json, pickle, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model import FullyConnectedRBM, RBMParams
from ising import TransverseFieldIsing1D


def make_rbm(state):
    rbm = FullyConnectedRBM(state["n_visible"], state["n_hidden"], jax.random.PRNGKey(0))
    rbm.params = RBMParams(a=jnp.asarray(state["a"]), b=jnp.asarray(state["b"]), W=jnp.asarray(state["W"]))
    return rbm


def metropolis(rbm, n_chains=4096, n_sweeps=600, burn=200, thin=10, seed=0):
    N = rbm.n_visible
    logp = jax.vmap(lambda v: 2.0 * rbm.log_psi(v))
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    V = jnp.where(jax.random.bernoulli(k, 0.5, (n_chains, N)), 1.0, -1.0)

    @jax.jit
    def sweep(V, key):
        def step(carry, i):
            V, lp, key = carry
            key, k = jax.random.split(key)
            Vp = V.at[:, i].multiply(-1.0)
            lpp = logp(Vp)
            acc = jnp.log(jax.random.uniform(k, (V.shape[0],))) < lpp - lp
            return (jnp.where(acc[:, None], Vp, V), jnp.where(acc, lpp, lp), key), acc.mean()
        (V, _, key), accs = jax.lax.scan(step, (V, logp(V), key), jnp.arange(N))
        return V, key, accs.mean()

    out, accs = [], []
    for s in range(n_sweeps):
        V, key, a = sweep(V, key)
        accs.append(float(a))
        if s >= burn and (s - burn) % thin == 0:
            out.append(V)
    return jnp.concatenate(out), float(np.mean(accs))


def true_energy(rbm, ising, **kw):
    V, acc = metropolis(rbm, **kw)
    E = np.asarray(ising.local_energy_batch(V, rbm))
    # per-snapshot means -> error bar that respects within-chain correlation across snapshots
    n_snap = len(E) // kw.get("n_chains", 4096)
    snap = E.reshape(n_snap, -1).mean(axis=1)
    return E.mean(), snap.std(ddof=1) / np.sqrt(n_snap), acc, V


if __name__ == "__main__":
    # sanity: random RBM at N=8 vs exact enumeration
    ising = TransverseFieldIsing1D(8, 0.5)
    rbm = FullyConnectedRBM(8, 8, jax.random.PRNGKey(3)); rbm.scale = 0.5
    rbm.params = rbm.init_params(jax.random.PRNGKey(3))
    allV = jnp.array(list(itertools.product([-1.0, 1.0], repeat=8)))
    p = np.exp(2 * np.asarray(jax.vmap(rbm.log_psi)(allV))); p /= p.sum()
    exact = float(p @ np.asarray(ising.local_energy_batch(allV, rbm)))
    mc, err, acc, _ = true_energy(rbm, ising)
    print(f"[sanity N=8] exact <H>={exact:.5f}  metropolis={mc:.5f} ± {err:.5f}  acc={acc:.2f}")

    for N in (32, 64):
        ck = pickle.load(open(REPO / f"checkpoints/tfim_1d/{N}/dimod/zephyr/full/checkpoint_1d_h0.5_rbmfull_nh{N}_lr0.08_iter0090.pkl", "rb"))
        assert ck["config"]["seed"] == 19 and ck["config"]["cem"]
        res = json.load(gzip.open(next((REPO / f"results/tfim_1d/{N}/dimod/zephyr").glob("*seed19_*cem1*.gz"))))
        ising = TransverseFieldIsing1D(N, 0.5)
        rbm = make_rbm(ck["rbm_state"])
        mc, err, acc, V = true_energy(rbm, ising)
        mz = np.asarray(V).mean(axis=1)
        ex = res["exact_energy"]
        print(f"[N={N} seed19 iter90] QPU-estimated E={res['history']['energy'][90]:.4f} "
              f"(final {res['final_energy']:.4f})  TRUE <H>_Psi={mc:.4f} ± {err:.4f}  exact={ex:.4f}  "
              f"true rel err={(mc-ex)/abs(ex)*100:.2f}%  acc={acc:.2f}  |m| median={np.median(np.abs(mz)):.3f}")

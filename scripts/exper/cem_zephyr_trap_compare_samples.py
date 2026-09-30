import sys, gzip, pickle, json, glob
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from cem_zephyr_trap_true_energy import *
def stats(V, rbm, ising):
    V = jnp.asarray(V, dtype=jnp.float64)
    dw = np.asarray(jnp.sum(V != jnp.roll(V, -1, axis=1), axis=1))
    m = np.abs(np.asarray(V.mean(axis=1)))
    lp = np.asarray(jax.vmap(lambda v: 2 * rbm.log_psi(v))(V))
    E = np.asarray(ising.local_energy_batch(V, rbm))
    return dict(dw=dw.mean(), dw0=np.mean(dw == 0), m=m.mean(), logpsi2=lp.mean(), Eloc=E.mean())
for N in (32, 64):
    ck = pickle.load(open(REPO / f"checkpoints/tfim_1d/{N}/dimod/zephyr/full/checkpoint_1d_h0.5_rbmfull_nh{N}_lr0.08_iter0090.pkl", "rb"))
    rbm = make_rbm(ck["rbm_state"]); ising = TransverseFieldIsing1D(N, 0.5)
    Vq = pickle.load(gzip.open(REPO / f"dwave_samples/{N}/dimod/zephyr/samples_1d_h0.5_rbmfull_nh{N}_lr0.08_reg0.05_ns200_seed19_iter0091.pkl.gz"))["v"]
    Vp, _ = metropolis(rbm, n_chains=2048, n_sweeps=400, burn=200, thin=20)
    for tag, V in (("Zephyr QPU  ", Vq), ("true |Psi|^2", Vp)):
        s = stats(V, rbm, ising)
        print(f"N={N} {tag}: domain walls={s['dw']:.2f} (P[0 walls]={s['dw0']:.2f})  <|m|>={s['m']:.3f}  <log Psi^2>={s['logpsi2']:.3f}  <E_loc>={s['Eloc']:.3f}")

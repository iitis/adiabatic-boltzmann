"""CPU test: does a large hidden beta_hw (Zephyr-like ~6) alone trap SR+CEM at
N=32 in a domain-wall local minimum, while beta_hw~2.6 (Pegasus-like) and
beta_hw=1 (pre-calibrated) do not?

Synthetic annealer: each read starts from random spins and runs block Gibbs on
p_s(v,h) ∝ exp(s(-a·v + b·h + vᵀWh)), s ramped geometrically up to
s = beta_hw / beta_x (hidden device temperature, trainable software scale),
then held. Returns joint (V, H) exactly like DimodSampler(return_hidden=True),
so the real Trainer + real CEM feedback rule run unmodified.
"""
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from encoder import Trainer


class SyntheticAnnealer:
    def __init__(self, beta_hw, n_ramp=100, n_hold=50, seed=0):
        self.beta_hw, self.n_ramp, self.n_hold = beta_hw, n_ramp, n_hold
        self.key = jax.random.PRNGKey(seed)
        self.last_sampling_time_s = 0.0
        self._run = jax.jit(self._anneal, static_argnums=(4,))

    @staticmethod
    def _anneal(a, b, W, key, ns, s_sched):
        N, M = W.shape
        key, k = jax.random.split(key)
        v = jnp.where(jax.random.bernoulli(k, 0.5, (ns, N)), 1.0, -1.0)

        def body(carry, s):
            v, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            ph = jax.nn.sigmoid(2 * s * (b + v @ W))
            h = jnp.where(jax.random.uniform(k1, ph.shape) < ph, 1.0, -1.0)
            pv = jax.nn.sigmoid(2 * s * (-a + h @ W.T))
            v = jnp.where(jax.random.uniform(k2, pv.shape) < pv, 1.0, -1.0)
            return (v, key), None

        (v, key), _ = jax.lax.scan(body, (v, key), s_sched)
        s = s_sched[-1]
        key, k = jax.random.split(key)
        ph = jax.nn.sigmoid(2 * s * (b + v @ W))
        h = jnp.where(jax.random.uniform(k, ph.shape) < ph, 1.0, -1.0)
        return v, h

    def sample(self, rbm, n_samples, config={}, return_hidden=False):
        s = self.beta_hw / config.get("beta_x", 1.0)
        sched = jnp.concatenate([jnp.geomspace(0.05 * s, s, self.n_ramp), jnp.full(self.n_hold, s)])
        self.key, k = jax.random.split(self.key)
        v, h = self._run(rbm.a, rbm.b, rbm.W, k, n_samples, sched)
        v, h = np.asarray(v), np.asarray(h)
        return (v, h) if return_hidden else v


def run(N, beta_hw, seed, beta_x_init=1.0, iters=100):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    ising = TransverseFieldIsing1D(N, 0.5)
    rbm = FullyConnectedRBM(N, N, mk)
    cfg = dict(learning_rate=0.08, n_iterations=iters, n_samples=200, regularization=0.05,
               use_cem=True, cem_interval=int(__import__("os").environ.get("CEM_INTERVAL", 5)), seed=seed, n_parallel=1, beta_x_init=beta_x_init)
    tr = Trainer(rbm, ising, SyntheticAnnealer(beta_hw, seed=seed + 1000), cfg, args=None)
    h = tr.train()
    ex = ising.exact_ground_energy()
    return dict(N=N, beta_hw=beta_hw, seed=seed, beta_x_init=beta_x_init,
                E=h["energy"], beta_x=h["beta_x"], exact=ex,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist())


if __name__ == "__main__":
    import contextlib, io
    N = int(sys.argv[1]); beta_hw = float(sys.argv[2]); seeds = [int(x) for x in sys.argv[3].split(",")]
    bxi = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    out = Path(sys.argv[5] if len(sys.argv) > 5 else f"synth_N{N}_bhw{beta_hw}_bxi{bxi}_ci{__import__('os').environ.get('CEM_INTERVAL', 5)}.jsonl")
    for sd in seeds:
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(N, beta_hw, sd, bxi)
        E = np.array(r["E"]); rel = (E - r["exact"]) / abs(r["exact"]) * 100
        print(f"N={N} beta_hw={beta_hw} bxi={bxi} seed={sd}: est err@10={rel[10]:+.2f}% @99={rel[99]:+.2f}%  "
              f"beta_x final={r['beta_x'][-1]:.2f}  wall={time.time()-t0:.0f}s", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")

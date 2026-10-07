"""Classical Metropolis training with saved parameters (audit response).

Same SR settings and initialization as the benchmark (scripts/main.py: PRNGKey(seed) split into
model and sampler keys; lr 0.08, reg 0.05, 200 samples, 100 steps; 200 chains restarted every
step, 200 warm-up sweeps + 1). Variants:
  mh       original sampler (single-spin flips only)
  mh_flip  plus one global spin-flip proposal per sweep (src/sampler.py, config global_flip)
  exact    independent exact draws from the Born distribution by enumeration (N<=16 only)
Final parameters are stored so the networks can be evaluated with audit_evaluate.py.

  python scripts/exper/audit_classical_retrain.py VARIANT N SEEDS
"""
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from sampler import ClassicalSampler
from encoder import Trainer

OUT = REPO / "results" / "audit_classical"


class ExactSampler:
    """Independent draws from |Psi|^2 by enumeration of all 2^N configurations."""
    def __init__(self, key):
        self.key = key

    def sample(self, rbm, n_samples, config=None, return_hidden=False, return_jax=False):
        import itertools
        N = rbm.n_visible
        vs = jnp.array(list(itertools.product([-1.0, 1.0], repeat=N)))
        phi = vs @ rbm.W + rbm.b
        lp = -vs @ rbm.a + jnp.sum(jnp.logaddexp(phi, -phi), axis=1)
        self.key, k = jax.random.split(self.key)
        v = vs[jax.random.categorical(k, lp, shape=(n_samples,))]
        return v if return_jax else np.asarray(v)


def run(variant, N, seed):
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    ising = TransverseFieldIsing1D(N, 0.5)
    rbm = FullyConnectedRBM(N, N, model_key)
    key, skey = jax.random.split(key)
    if variant == "exact":
        sampler = ExactSampler(skey)
    else:
        sampler = ClassicalSampler(method="metropolis", n_sweeps=1); sampler._key = skey
    cfg = dict(learning_rate=0.08, n_iterations=100, n_samples=200, regularization=0.05, save_checkpoints=False,
               use_cem=False, seed=seed, n_parallel=1, global_flip=variant == "mh_flip")
    t0 = time.time()
    hist = Trainer(rbm, ising, sampler, cfg, args=None).train()
    return dict(tag=variant, device="classical", N=N, seed=seed, exact=ising.exact_ground_energy(),
                E=[float(e) for e in hist["energy"]], sampling_time_s=[float(t) for t in hist.get("sampling_time_s", [])],
                wall_s=time.time() - t0,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist())


if __name__ == "__main__":
    variant, N, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert variant in ("mh", "mh_flip", "exact")
    lo, hi = map(int, seeds.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{variant}_N{N}.jsonl"
    done = {json.loads(l)["seed"] for l in open(out)} if out.exists() else set()
    for seed in range(lo, hi + 1):
        if seed in done:
            continue
        r = run(variant, N, seed)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{variant} N={N} seed={seed}: sampled err/N={(np.mean(r['E'][-10:]) - r['exact']) / N:.5f} "
              f"wall={r['wall_s']:.0f}s", flush=True)

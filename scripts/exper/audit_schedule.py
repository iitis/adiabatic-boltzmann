"""Training with exact samples of pi_theta^s under a schedule for s (no device).

Same Trainer, SR settings and initialization as the benchmark (PRNGKey(seed) split into model and
sampler keys; lr 0.08, reg 0.05, 200 samples, 100 steps). At step t the 200 samples are drawn exactly,
by enumeration, from p_s(v) ~ pi_theta(v)^s(t). Schedules (name -> s(t)):
  exact          1
  hot0.5         0.5 throughout
  ramp0.3_50     0.3 -> 1 linearly over the first 50 steps, then 1
  ramp0.5_50     0.5 -> 1 over 50 steps
  ramp0.5_25     0.5 -> 1 over 25 steps
  cold2_25       2 -> 1 over 25 steps (a too-cold start)
  hot0.3, ramp0.3_100, ramp0.2_75   hotter and longer variants
For N>16 the samples come from Metropolis chains with a global spin-flip move (ScheduledMH).
The final energy is evaluated at s=1: by enumeration (N<=16) or audit_evaluate.py.

    python scripts/exper/audit_schedule.py SCHEDULE N SEEDS     # e.g. ramp0.5_50 16 0-19
"""
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from encoder import Trainer

OUT = REPO / "results" / "audit_schedule"
SCHEDULES = {"exact": (1.0, 1), "hot0.5": (0.5, None), "ramp0.3_50": (0.3, 50), "ramp0.5_50": (0.5, 50),
             "ramp0.5_25": (0.5, 25), "cold2_25": (2.0, 25),
             "hot0.3": (0.3, None), "ramp0.3_100": (0.3, 100), "ramp0.2_75": (0.2, 75)}


def s_of(name, t):
    s0, K = SCHEDULES[name]
    if K is None:
        return s0
    return s0 + (1.0 - s0) * min(1.0, t / K)


@jax.jit
def _draw(key, vs, a, b, W, s):
    phi = vs @ W + b
    lp = -vs @ a + jnp.sum(jnp.logaddexp(phi, -phi), axis=1)
    return vs[jax.random.categorical(key, s * lp, shape=(200,))]


class ScheduledExact:
    def __init__(self, key, N, name):
        idx = np.arange(2 ** N)
        self.vs = jnp.asarray(((idx[:, None] >> np.arange(N - 1, -1, -1)) & 1) * 2.0 - 1)
        self.key, self.name, self.t, self.s_hist = key, name, 0, []

    def sample(self, rbm, n_samples, config=None, return_hidden=False, return_jax=False):
        assert n_samples == 200
        s = s_of(self.name, self.t); self.s_hist.append(s); self.t += 1
        self.key, k = jax.random.split(self.key)
        v = _draw(k, self.vs, rbm.a, rbm.b, rbm.W, s)
        return v if return_jax else np.asarray(v)


class ScheduledMH:
    """N>16: 200 restarted Metropolis chains (200+1 sweeps, one global flip per sweep) targeting pi^s(t)."""
    def __init__(self, key, name):
        from sampler import ClassicalSampler
        self.inner = ClassicalSampler(method="metropolis", n_sweeps=1); self.inner._key = key
        self.name, self.t, self.s_hist = name, 0, []

    def sample(self, rbm, n_samples, config=None, return_hidden=False, return_jax=False):
        s = s_of(self.name, self.t); self.s_hist.append(s); self.t += 1
        return self.inner.sample(rbm, n_samples, {"global_flip": True, "s_power": s}, return_hidden, return_jax)


def run(name, N, seed):
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    ising = TransverseFieldIsing1D(N, 0.5)
    rbm = FullyConnectedRBM(N, N, model_key)
    key, skey = jax.random.split(key)
    smp = ScheduledExact(skey, N, name) if N <= 16 else ScheduledMH(skey, name)
    cfg = dict(learning_rate=0.08, n_iterations=100, n_samples=200, regularization=0.05, save_checkpoints=False,
               use_cem=False, seed=seed, n_parallel=1)
    t0 = time.time()
    hist = Trainer(rbm, ising, smp, cfg, args=None).train()
    return dict(tag=name, N=N, seed=seed, exact=ising.exact_ground_energy(), s=smp.s_hist,
                E=[float(e) for e in hist["energy"]], wall_s=time.time() - t0,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist())


if __name__ == "__main__":
    name, N, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert name in SCHEDULES
    lo, hi = map(int, seeds.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}_N{N}.jsonl"
    done = {json.loads(l)["seed"] for l in open(out)} if out.exists() else set()
    for seed in range(lo, hi + 1):
        if seed in done:
            continue
        r = run(name, N, seed)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{name} N={N} seed={seed}: wall={r['wall_s']:.0f}s", flush=True)

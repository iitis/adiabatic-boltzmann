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
For N>16 the samples come from Metropolis chains with a global spin-flip move (ScheduledMH), restarted
every step (200+1 sweeps), or persistent across steps with argument "pmh" (200 sweeps once, then 10 per step).
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
N_SAMPLES = 200  # samples per step; changed with the 5th argument (pmh only)
import os
H_FIELD = float(os.environ.get("AUDIT_H", "0.5"))  # transverse field; output files get a suffix if != 0.5
SCHEDULES = {"exact": (1.0, 1), "hot0.5": (0.5, None), "ramp0.3_50": (0.3, 50), "ramp0.5_50": (0.5, 50),
             "ramp0.5_25": (0.5, 25), "cold2_25": (2.0, 25),
             "hot0.3": (0.3, None), "ramp0.3_100": (0.3, 100), "ramp0.2_75": (0.2, 75)}


# Sample corruption mimicking the decoder: "plus0.2_8" sets each spin to +1 with probability 0.2 in the first
# 8 steps (s=1); "rand0.2_8" sets the same fraction to a random +-1 instead (noise without symmetry breaking).
CORRUPT = {"plus0.2_8": ("plus", 0.2, 8), "rand0.2_8": ("rand", 0.2, 8), "plus0.2_16": ("plus", 0.2, 16),
           "plus0.1_8": ("plus", 0.1, 8)}
for _k in CORRUPT:
    SCHEDULES[_k] = (1.0, 1)
# "mfinit": s=1, visible biases initialized to the best uniform product state, a_i = -artanh(sqrt(1-h^2/4))
# (pi ~ exp(-a.v): <v_i> = tanh(-a_i) = cos(alpha), sin(alpha) = h/2); W and b keep the default random start.
SCHEDULES["mfinit"] = (1.0, 1)


def corrupt(name, t, v, rng):
    if name not in CORRUPT:
        return v
    kind, q, T = CORRUPT[name]
    if t >= T:
        return v
    v = np.array(v, dtype=float)
    hit = rng.random(v.shape) < q
    v[hit] = 1.0 if kind == "plus" else rng.choice([-1.0, 1.0], hit.sum())
    return v


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
        self.rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

    def sample(self, rbm, n_samples, config=None, return_hidden=False, return_jax=False):
        assert n_samples == 200, "exact sampler draws 200 samples"
        s = s_of(self.name, self.t); self.s_hist.append(s)
        self.key, k = jax.random.split(self.key)
        v = corrupt(self.name, self.t, np.asarray(_draw(k, self.vs, rbm.a, rbm.b, rbm.W, s)), self.rng); self.t += 1
        return jnp.asarray(v) if return_jax else v


class ScheduledMH:
    """N>16: 200 restarted Metropolis chains (200+1 sweeps, one global flip per sweep) targeting pi^s(t)."""
    def __init__(self, key, name, persistent=False):
        from sampler import ClassicalSampler
        self.inner = ClassicalSampler(method="metropolis", n_sweeps=1); self.inner._key = key
        self.name, self.t, self.s_hist, self.persistent = name, 0, [], persistent
        self.rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

    def sample(self, rbm, n_samples, config=None, return_hidden=False, return_jax=False):
        s = s_of(self.name, self.t); self.s_hist.append(s)
        v = self.inner.sample(rbm, n_samples, {"global_flip": True, "s_power": s, "persistent": self.persistent},
                              return_hidden, return_jax)
        v = corrupt(self.name, self.t, v, self.rng); self.t += 1  # chains keep their uncorrupted state
        return jnp.asarray(v) if return_jax else v


def run(name, N, seed, sampler="auto"):
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    ising = TransverseFieldIsing1D(N, H_FIELD)
    rbm = FullyConnectedRBM(N, N, model_key)
    if name == "mfinit":
        from model import RBMParams
        a0 = -np.arctanh(np.sqrt(max(1 - H_FIELD ** 2 / 4, 0.0)))
        rbm.params = RBMParams(a=jnp.full(N, a0), b=rbm.params.b, W=rbm.params.W)
    key, skey = jax.random.split(key)
    if sampler == "pmh":
        smp = ScheduledMH(skey, name, persistent=True)
    else:
        smp = ScheduledExact(skey, N, name) if N <= 16 else ScheduledMH(skey, name)
    cfg = dict(learning_rate=0.08, n_iterations=100, n_samples=N_SAMPLES, regularization=0.05, save_checkpoints=False,
               use_cem=False, seed=seed, n_parallel=1)
    t0 = time.time()
    hist = Trainer(rbm, ising, smp, cfg, args=None).train()
    return dict(tag=name, N=N, h=H_FIELD, seed=seed, exact=ising.exact_ground_energy(), s=smp.s_hist,
                E=[float(e) for e in hist["energy"]], wall_s=time.time() - t0,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist())


if __name__ == "__main__":
    name, N, seeds = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sampler = sys.argv[4] if len(sys.argv) > 4 else "auto"  # "pmh": persistent Metropolis chains
    if len(sys.argv) > 5:
        N_SAMPLES = int(sys.argv[5]); sampler_tag = f"{sampler}_S{N_SAMPLES}"
    else:
        sampler_tag = sampler
    assert name in SCHEDULES
    lo, hi = map(int, seeds.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    hs = "" if H_FIELD == 0.5 else f"_h{H_FIELD:g}"
    out = OUT / (f"{name}_N{N}{hs}.jsonl" if sampler_tag == "auto" else f"{name}_N{N}{hs}_{sampler_tag}.jsonl")
    done = {json.loads(l)["seed"] for l in open(out)} if out.exists() else set()
    for seed in range(lo, hi + 1):
        if seed in done:
            continue
        r = run(name, N, seed, sampler)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{name} N={N} seed={seed}: wall={r['wall_s']:.0f}s", flush=True)

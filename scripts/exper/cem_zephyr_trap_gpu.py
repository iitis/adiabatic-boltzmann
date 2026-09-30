"""GPU/CPU-only follow-up to cem_zephyr_trap_synthetic.py (no QPU).

Runs the real Trainer + CEM rule with one of two samplers:
  synth : SyntheticAnnealer (hidden beta_hw, optional ICE noise on programmed h/J)
  exact : persistent single-spin Metropolis on |Psi|^2 (ignores beta_x)
plus optional training-schedule knobs (freeze params for the first K iters,
lr warmup, CEM log-EMA alpha). Each finished run is scored with an unbiased
Metropolis <H>_Psi, domain-wall count and |m|.

Usage: python cem_zephyr_trap_gpu.py OUT.jsonl SEEDS key=val ...
  e.g. ... out.jsonl 0-19 sampler=synth beta_hw=6 freeze=30
"""
import contextlib, io, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model import FullyConnectedRBM, RBMParams
from ising import TransverseFieldIsing1D
from encoder import Trainer
from cem_zephyr_trap_true_energy import metropolis
from cem_zephyr_visible_pl import flip_deltas, fit_s

_flip_deltas = jax.jit(flip_deltas)

DEFAULTS = dict(N=32, h=0.5, sampler="synth", beta_hw=1.0, bxi=1.0, lr=0.08, ns=200, reg=0.05,
                iters=100, ci=5, alpha=0.3, freeze=0, warmup=0, ice=0.0, sweeps=10,
                n_ramp=100, n_hold=50, init="", boot=0, calib=0, snap=0, drift0=1.0, driftT=30, fb="cem")


class SyntheticAnnealer:
    """Block Gibbs on p_s(v,h) ∝ exp(s(-a·v + b·h + vᵀWh)), s ramped to beta_hw/beta_x.
    ice > 0 adds fresh N(0, ice) noise per call to the *programmed* h/J = (a,b,W)/beta_x."""

    def __init__(self, beta_hw, ice=0.0, n_ramp=100, n_hold=50, seed=0, drift0=1.0, driftT=30):
        # drift: hidden temperature ramps linearly from drift0*beta_hw to beta_hw over driftT calls
        # (mimics real Zephyr, where beta_eff grows ~2-4 -> ~5-6 as couplings grow)
        self.beta_hw, self.ice, self.n_ramp, self.n_hold = beta_hw, ice, n_ramp, n_hold
        self.drift0, self.driftT, self.calls = drift0, driftT, 0
        self.key = jax.random.PRNGKey(seed)
        self.last_sampling_time_s = 0.0
        self._run = jax.jit(self._anneal, static_argnums=(4,))

    @staticmethod
    def _anneal(a, b, W, key, ns, s_sched):
        key, k = jax.random.split(key)
        v = jnp.where(jax.random.bernoulli(k, 0.5, (ns, W.shape[0])), 1.0, -1.0)

        def body(carry, s):
            v, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            ph = jax.nn.sigmoid(2 * s * (b + v @ W))
            h = jnp.where(jax.random.uniform(k1, ph.shape) < ph, 1.0, -1.0)
            pv = jax.nn.sigmoid(2 * s * (-a + h @ W.T))
            v = jnp.where(jax.random.uniform(k2, pv.shape) < pv, 1.0, -1.0)
            return (v, key), None

        (v, key), _ = jax.lax.scan(body, (v, key), s_sched)
        key, k = jax.random.split(key)
        ph = jax.nn.sigmoid(2 * s_sched[-1] * (b + v @ W))
        return v, jnp.where(jax.random.uniform(k, ph.shape) < ph, 1.0, -1.0)

    def sample(self, rbm, n_samples, config={}, return_hidden=False):
        bx = config.get("beta_x", 1.0)
        a, b, W = rbm.a / bx, rbm.b / bx, rbm.W / bx  # programmed values
        if self.ice > 0:
            self.key, k1, k2, k3 = jax.random.split(self.key, 4)
            a = a + self.ice * jax.random.normal(k1, a.shape)
            b = b + self.ice * jax.random.normal(k2, b.shape)
            W = W + self.ice * jax.random.normal(k3, W.shape)
        s = self.beta_hw * min(1.0, self.drift0 + (1 - self.drift0) * self.calls / self.driftT)
        self.calls += 1
        sched = jnp.concatenate([jnp.geomspace(0.05 * s, s, self.n_ramp), jnp.full(self.n_hold, s)])
        self.key, k = jax.random.split(self.key)
        v, h = self._run(a, b, W, k, n_samples, sched)
        v, h = np.asarray(v), np.asarray(h)
        return (v, h) if return_hidden else v


class ExactSampler:
    """Persistent Metropolis chains on |Psi|^2 (n_chains = n_samples); ignores beta_x."""

    def __init__(self, sweeps=10, burn=200, seed=0):
        self.sweeps, self.burn = sweeps, burn
        self.key = jax.random.PRNGKey(seed)
        self.V = None
        self.last_sampling_time_s = 0.0

        def sweep_n(params, V, key, n):
            def logp(v):
                theta = params.b + params.W.T @ v
                return -params.a @ v + jnp.sum(jnp.logaddexp(theta, -theta))
            logp_b = jax.vmap(logp)

            def one_sweep(carry, _):
                def step(c, i):
                    V, lp, key = c
                    key, k = jax.random.split(key)
                    Vp = V.at[:, i].multiply(-1.0)
                    lpp = logp_b(Vp)
                    acc = jnp.log(jax.random.uniform(k, (V.shape[0],))) < lpp - lp
                    return (jnp.where(acc[:, None], Vp, V), jnp.where(acc, lpp, lp), key), None
                V, key = carry
                (V, _, key), _ = jax.lax.scan(step, (V, logp_b(V), key), jnp.arange(V.shape[1]))
                return (V, key), None
            (V, key), _ = jax.lax.scan(one_sweep, (V, key), None, length=n)
            return V, key
        self._sweep = jax.jit(sweep_n, static_argnums=(3,))

    def sample(self, rbm, n_samples, config={}, return_hidden=False):
        n = self.sweeps
        if self.V is None:
            self.key, k = jax.random.split(self.key)
            self.V = jnp.where(jax.random.bernoulli(k, 0.5, (n_samples, rbm.n_visible)), 1.0, -1.0)
            n = self.burn
        self.V, self.key = self._sweep(rbm.params, self.V, self.key, n)
        v = np.asarray(self.V)
        if not return_hidden:
            return v
        self.key, k = jax.random.split(self.key)
        ph = jax.nn.sigmoid(2 * (rbm.b + self.V @ rbm.W))
        return v, np.asarray(jnp.where(jax.random.uniform(k, ph.shape) < ph, 1.0, -1.0))


class Scheduled:
    """Wraps a sampler; before each draw sets trainer.learning_rate from the schedule
    (freeze: lr=0 for the first `freeze` iters; warmup: linear ramp over `warmup` iters).
    calib=K: the first K draws are a calibration phase (lr=0, CEM every draw, full log step);
    training (freeze/warmup schedule, configured cem_interval/alpha) starts after it.
    boot=1: the first CEM reading of training is applied as a full log step (alpha=1).
    fb="pl": beta_x is driven by the visible-marginal pseudo-likelihood temperature s_PL of each
    draw (beta_x <- beta_x * s_PL**alpha, full step during calib) instead of the Trainer's
    joint-(v,h) CEM; the Trainer must then run with use_cem=False, beta_adapt=0.
    fb="cem+pl": calibration draws use the Trainer's joint CEM (s_PL is undetermined at the
    near-zero initial couplings); from the first training draw on, beta_x follows s_PL, with the
    per-update factor clipped to [1/1.5, 1.5]. The Trainer must start with use_cem=True, beta_adapt=0."""

    def __init__(self, inner, lr, freeze, warmup, calib=0, boot=0, ci=5, alpha=0.3):
        self.inner, self.lr, self.freeze, self.warmup, self.t, self.trainer = inner, lr, freeze, warmup, 0, None
        self.calib, self.boot, self.ci, self.alpha = calib, boot, ci, alpha
        self.snap, self.snaps = 0, []  # snap=k: keep rbm.params every k draws
        self.fb, self.bx, self.bx_hist, self.s_pl = "cem", 1.0, [], []

    @property
    def last_sampling_time_s(self):
        return self.inner.last_sampling_time_s

    def sample(self, rbm, n_samples, config={}, return_hidden=False):
        tr = self.trainer
        if self.snap and self.t % self.snap == 0:
            self.snaps.append((self.t, rbm.params))
        if self.t < self.calib:
            tr.learning_rate, tr.cem_interval, tr.cem_ema_alpha = 0.0, 1, 1.0
        else:
            t = self.t - self.calib - self.freeze
            tr.learning_rate = 0.0 if t < 0 else self.lr * min(1.0, (t + 1) / max(self.warmup, 1))
            # Trainer fires CEM when iteration % cem_interval == 0; re-anchor to training start
            tr.cem_interval = self.ci
            tr.cem_ema_alpha = 1.0 if (self.boot and self.t - self.calib < self.ci) else self.alpha
        if self.fb == "cem+pl" and self.t == self.calib:
            tr.use_cem, self.bx = False, tr.beta_x  # hand over from CEM calibration to PL
        if self.fb == "cem" or (self.fb == "cem+pl" and self.t < self.calib):
            self.bx_hist.append(config.get("beta_x", 1.0))
            self.t += 1
            return self.inner.sample(rbm, n_samples, config, return_hidden)
        r = self.inner.sample(rbm, n_samples, {**config, "beta_x": self.bx}, return_hidden)
        V = jnp.asarray(r[0] if return_hidden else r, dtype=jnp.float64)
        s = float(fit_s(_flip_deltas(rbm.a, rbm.b, rbm.W, V)))
        self.s_pl.append(s)
        self.bx_hist.append(self.bx)  # beta_x these samples were drawn at
        a = 1.0 if self.t < self.calib else self.alpha
        if self.t < self.calib or (self.t - self.calib) % self.ci == 0:
            f = s ** a if self.fb == "pl" else float(np.clip(s, 1 / 1.5, 1.5)) ** a
            self.bx = float(np.clip(self.bx * f, 0.05, 20.0))
        tr.beta_x = self.bx
        self.t += 1
        return r


def score(rbm, ising):
    V, _ = metropolis(rbm, n_chains=1024, n_sweeps=400, burn=200, thin=20)
    E = np.asarray(ising.local_energy_batch(V, rbm))
    Vn = np.asarray(V)
    return dict(E_true=float(E.mean()), dw=float(np.mean(np.sum(Vn != np.roll(Vn, -1, 1), 1))),
                absm=float(np.mean(np.abs(Vn.mean(1)))))


def run(seed, p):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    ising = TransverseFieldIsing1D(p["N"], p["h"])
    rbm = FullyConnectedRBM(p["N"], p["N"], mk)
    if p["init"]:
        st = next(json.loads(l) for l in open(p["init"]) if json.loads(l)["seed"] == seed)
        rbm.params = RBMParams(a=jnp.asarray(st["a"]), b=jnp.asarray(st["b"]), W=jnp.asarray(st["W"]))
    if p["sampler"] == "exact":
        inner = ExactSampler(sweeps=p["sweeps"], seed=seed + 1000)
    else:
        inner = SyntheticAnnealer(p["beta_hw"], p["ice"], p["n_ramp"], p["n_hold"], seed=seed + 1000,
                                  drift0=p["drift0"], driftT=p["driftT"])
    smp = Scheduled(inner, p["lr"], p["freeze"], p["warmup"], p["calib"], p["boot"], p["ci"], p["alpha"])
    cfg = dict(learning_rate=p["lr"], n_iterations=p["iters"] + p["calib"], n_samples=p["ns"], regularization=p["reg"],
               use_cem=p["sampler"] == "synth" and p["fb"] != "pl", beta_adapt=0.05 if p["fb"] == "cem" else 0.0, cem_interval=p["ci"], cem_ema_alpha=p["alpha"],
               seed=seed, n_parallel=1, beta_x_init=p["bxi"])
    tr = Trainer(rbm, ising, smp, cfg, args=None)
    smp.trainer = tr
    smp.snap, smp.fb, smp.bx = p["snap"], p["fb"], p["bxi"]
    hist = tr.train()
    ex = ising.exact_ground_energy()
    traj = []
    for t, prm in smp.snaps:  # true state along the trajectory
        snap_rbm = FullyConnectedRBM(p["N"], p["N"], mk); snap_rbm.params = prm
        traj.append(dict(t=t, **score(snap_rbm, ising)))
    return dict(seed=seed, params=p, exact=ex, E=hist["energy"], beta_x=hist["beta_x"] if p["fb"] == "cem" else smp.bx_hist,
                s_pl=smp.s_pl, traj=traj, **score(rbm, ising),
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist())


def parse_seeds(s):
    if "-" in s:
        lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    out, seeds = Path(sys.argv[1]), parse_seeds(sys.argv[2])
    p = dict(DEFAULTS)
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        p[k] = type(DEFAULTS[k])(v)
    done = {json.loads(l)["seed"] for l in open(out)} if out.exists() else set()
    for sd in seeds:
        if sd in done:
            continue
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(sd, p)
        rel = (r["E_true"] - r["exact"]) / abs(r["exact"]) * 100
        print(f"{out.name} seed={sd}: true err={rel:+.2f}%  est@end={(r['E'][-1]-r['exact'])/abs(r['exact'])*100:+.2f}%  "
              f"dw={r['dw']:.2f} |m|={r['absm']:.2f} beta_x={r['beta_x'][-1]:.2f}  {time.time()-t0:.0f}s", flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")

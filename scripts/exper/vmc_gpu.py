#!/usr/bin/env python3
"""VMC (RBM + SR) on general-graph TFIMs (src/graph_tfim.py), built to fill a GPU with large sample counts.

Question: when can VMC with MCMC (or annealer) sampling compete? This driver varies what the 1D study left
open — samples per step, sampler, hidden density, optimiser settings, initial state, frustration — on
  --model chain --size N        ferromagnetic periodic chain (compare with results/dmrg_baseline)
  --model tri   --size L1xL2    antiferromagnetic triangular torus (frustrated but sign-free; L1, L2 % 3 == 0)
  --model tricyl --size L1xL2   the same on a cylinder, open along L1, periodic around L2 (L2 % 3 == 0)
Arms:
  exact_sr   SR with exact expectations over all 2^N configurations (N <= 18): the ansatz ceiling
  gibbs      persistent block Gibbs, --ns chains, 10 sweeps per step
  mh         persistent single-flip Metropolis + one global spin flip per sweep, --ns chains, 10 sweeps/step
SR: --lr / --shift (defaults: 0.08 / 0.05, the benchmark's, for sampled arms; 0.05 / 1e-4 for exact_sr).
Run exact_sr with --lr 0.08 --shift 0.05 to get the ns -> infinity limit of the sampled arms.
--init mf starts the visible biases at the best product state (per-site angles), W and b random as usual.

Resumable: state (params, sampler key, persistent chains, energy trajectory) in OUT/state/<run>_s<seed>.npz;
parameters of every checkpoint in OUT/ckpt/; one jsonl record per checkpoint in OUT/<model>_<size>_h<h>.jsonl.
Training runs in 100-step sub-segments (exact for both persistent samplers) with a heartbeat on stderr.
A (run, seed) holds an flock while it trains, so duplicate jobs skip instead of colliding.
Evaluation per checkpoint: exact enumeration (N <= 18) or GPU Metropolis with global flips (1024 chains);
records E, se, Var(H), <|m|>, 3-sublattice order <m3> and <m3^2> (tri), reference E0 (ED for N <= 18, free
fermions for the chain) and the best product-state energy.

  python scripts/exper/vmc_gpu.py --model tri --size 6x6 --h 1.0 --arm gibbs --ns 10000 --alpha 2 \\
      --seeds 0-2 --steps 300,1000,3000
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # several workers share one GPU

import argparse
import contextlib
import fcntl
import functools
import io
import itertools
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "exper"))
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from encoder import Trainer
from graph_tfim import _local_energy_graph_jit, make
from ising import TransverseFieldIsing1D
from mf_vs_h import append_jsonl, read_jsonl
from model import FullyConnectedRBM, RBMParams
from sampler import ClassicalSampler

OUT = Path(os.environ.get("VMC_GPU_OUT", REPO / "results" / "vmc_gpu"))
ENUM_MAX_N, CHUNK, SUBSEG = 18, 1 << 15, 100
DEFAULTS = {"exact_sr": (0.05, 1e-4)}  # (lr, shift); sampled arms: (0.08, 0.05)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def all_configs(N):
    return jnp.array(list(itertools.product([-1.0, 1.0], repeat=N)))


def log_p(V, a, b, W):
    phi = V @ W + b
    return -V @ a + jnp.sum(jnp.logaddexp(phi, -phi), axis=-1)


def sublattice_phase(model, size):
    """e^(2 pi i s/3) per site for the triangular 3-sublattice order (s = x - y mod 3), else None."""
    if model not in ("tri", "tricyl"):
        return None
    L1, L2 = map(int, size.split("x"))
    s = np.array([(x - y) % 3 for x in range(L1) for y in range(L2)])
    return jnp.asarray(np.exp(2j * np.pi * s / 3))


class PersistentMH(ClassicalSampler):
    """Metropolis sampler whose persistent chains live under _gibbs_v as well, so the Trainer's warm-up
    (which saves and restores _key and _gibbs_v) leaves them untouched and sub-segments are exact."""

    @property
    def _gibbs_v(self):
        return getattr(self, "_mh_v", None)

    @_gibbs_v.setter
    def _gibbs_v(self, v):
        self._mh_v = v


# ── exact SR (all 2^N configurations) ──────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(3,))
def _exact_sr_step(a, b, W, h, J, V, lr, shift):
    N, M = W.shape
    lp = log_p(V, a, b, W)
    p = jnp.exp(lp - jax.scipy.special.logsumexp(lp))
    el = _local_energy_graph_jit(V, W, a, b, h, J)
    t = jnp.tanh(V @ W + b)
    O = jnp.concatenate([-0.5 * V, 0.5 * t, 0.5 * (V[:, :, None] * t[:, None, :]).reshape(V.shape[0], -1)], 1)
    Oc = O - p @ O
    E = p @ el
    S = Oc.T @ (p[:, None] * Oc)
    F = Oc.T @ (p * (el - E))
    d = jnp.linalg.solve(S + shift * jnp.eye(S.shape[0]), F)
    return a - lr * d[:N], b - lr * d[N:N + M], W - lr * d[N + M:].reshape(N, M), E


# ── run state ──────────────────────────────────────────────────────────────
def run_name(a):
    return (f"{a.model}_{a.size}_h{a.h:g}_{a.arm}_ns{a.ns}_a{a.alpha}_lr{a.lr:g}_sh{a.shift:g}"
            + ("_mfinit" if a.init == "mf" else ""))


def init_state(model, alpha, seed, init):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    p = FullyConnectedRBM(model.size, alpha * model.size, mk).params
    key, sk = jax.random.split(key)
    a = np.asarray(p.a)
    if init == "mf":  # <v_i> = tanh(-a_i) = cos(theta_i) of the best product state
        _, theta = model.mean_field(n_starts=64)
        a = -np.arctanh(np.clip(np.cos(theta), -1 + 1e-12, 1 - 1e-12))
    return dict(step=0, a=a, b=np.asarray(p.b), W=np.asarray(p.W), skey=np.asarray(sk),
                chain_v=None, energy=np.zeros(0), wall_s=0.0)


def load_state(f, model, alpha, seed, init):
    if not f.exists():
        return init_state(model, alpha, seed, init)
    z = np.load(f)
    s = {k: z[k] for k in z.files}
    s["step"], s["wall_s"] = int(s["step"]), float(s["wall_s"])
    s["chain_v"] = s["chain_v"] if s["chain_v"].size else None
    return s


def save_npz(f, d):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **d)
    tmp.rename(f)


def save_state(f, s):
    save_npz(f, {**s, "chain_v": s["chain_v"] if s["chain_v"] is not None else np.zeros(0)})


def train_segment(args, model, s, to_step, V_all):
    if args.arm == "exact_sr":
        a, b, W = (jnp.asarray(s[k]) for k in "abW")
        Es = []
        for _ in range(s["step"], to_step):
            a, b, W, E = _exact_sr_step(a, b, W, model.h, model.J, V_all, args.lr, args.shift)
            Es.append(float(E))
        s.update(a=np.asarray(a), b=np.asarray(b), W=np.asarray(W))
        return Es
    N, M = s["W"].shape
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(0))
    rbm.params = RBMParams(*(jnp.asarray(s[k]) for k in "abW"))
    cfg = dict(learning_rate=args.lr, regularization=args.shift, save_checkpoints=False, use_cem=False,
               n_parallel=1, n_samples=args.ns, n_iterations=to_step, seed=0)
    if args.arm == "gibbs":
        sampler = ClassicalSampler(method="gibbs", n_warmup=200, n_sweeps=10)
    else:
        sampler = PersistentMH(method="metropolis", n_warmup=200, n_sweeps=1)
        cfg.update(persistent=True, global_flip=True, n_sweeps_persistent=10)
    sampler._gibbs_v = jnp.asarray(s["chain_v"]) if s["chain_v"] is not None else None
    sampler._key = jnp.asarray(s["skey"])
    with contextlib.redirect_stdout(io.StringIO()):
        hist = Trainer(rbm, model, sampler, cfg, args=None).train(start_iteration=s["step"])
    Es = [float(e) for e in hist["energy"]]
    if len(Es) != to_step - s["step"] or not np.all(np.isfinite(Es)):
        raise RuntimeError(f"segment {s['step']}->{to_step}: {len(Es)} steps, finite={np.all(np.isfinite(Es))}")
    s.update(a=np.asarray(rbm.a), b=np.asarray(rbm.b), W=np.asarray(rbm.W), skey=np.asarray(sampler._key))
    s["chain_v"] = np.asarray(sampler._gibbs_v) if sampler._gibbs_v is not None else None
    return Es


# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, s, V_all, phase, seed, chains=1024, burn=2000, n_meas=200, thin=5):
    N, M = s["W"].shape
    a, b, W = (jnp.asarray(s[k]) for k in "abW")
    rbm = FullyConnectedRBM(N, M, jax.random.PRNGKey(0))
    rbm.params = RBMParams(a, b, W)
    if V_all is not None:
        lp = jnp.concatenate([log_p(V_all[i:i + CHUNK], a, b, W) for i in range(0, V_all.shape[0], CHUNK)])
        p = jnp.exp(lp - jax.scipy.special.logsumexp(lp))
        el = model.local_energy_batch(V_all, rbm, chunk=CHUNK)
        E = float(p @ el)
        out = dict(E=E, se=0.0, var=float(p @ (el - E) ** 2), absm=float(p @ jnp.abs(V_all.mean(1))),
                   evaluator="enumeration")
        if phase is not None:
            m3 = jnp.abs(V_all @ phase) / N
            out.update(m3=float(p @ m3), m3sq=float(p @ m3 ** 2))
        return out
    sampler = PersistentMH(method="metropolis", n_warmup=burn, n_sweeps=1)
    sampler._key = jax.random.PRNGKey(10_000 + seed)
    cfg = dict(persistent=True, global_flip=True, n_sweeps_persistent=thin)
    E_sum = E2_sum = m_sum = m3_sum = m3sq_sum = 0.0
    chain_E = np.zeros(chains)
    with contextlib.redirect_stdout(io.StringIO()):
        sampler.sample(rbm, chains, config=cfg)  # first call: burn-in sweeps
        for _ in range(n_meas):
            V = jnp.asarray(sampler.sample(rbm, chains, config=cfg))
            el = np.asarray(model.local_energy_batch(V, rbm))
            chain_E += el
            E_sum += el.sum(); E2_sum += (el ** 2).sum()
            m_sum += float(jnp.abs(V.mean(1)).sum())
            if phase is not None:
                m3 = jnp.abs(V @ phase) / N
                m3_sum += float(m3.sum()); m3sq_sum += float((m3 ** 2).sum())
    n = chains * n_meas
    E = E_sum / n
    cm = chain_E / n_meas
    out = dict(E=float(E), se=float(cm.std(ddof=1) / np.sqrt(chains)), var=float(E2_sum / n - E ** 2),
               absm=m_sum / n, evaluator=f"metropolis_globalflip_{chains}x{n_meas}")
    if phase is not None:
        out.update(m3=m3_sum / n, m3sq=m3sq_sum / n)
    return out


@contextlib.contextmanager
def locked(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def references(args, model):
    """Exact energy and best product state, cached per (model, h) in OUT/references.json (flock)."""
    f = OUT / "references.json"
    key = f"{model.name}_N{model.size}_h{model.h:g}"
    with locked(OUT / "locks" / "references.lock"):
        refs = json.loads(f.read_text()) if f.exists() else {}
        if key not in refs:
            if args.model == "chain":
                e0 = TransverseFieldIsing1D(model.size, model.h)._compute_exact_ground_energy()  # free fermions
            elif model.size <= ENUM_MAX_N:
                e0 = model.exact_ground_energy()
            else:
                e0 = None
            refs[key] = dict(E0=e0, E_mf=model.mean_field(n_starts=64)[0])
            tmp = f.with_name(f"references.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(refs, indent=1, sort_keys=True))
            tmp.rename(f)
        return refs[key]["E0"], refs[key]["E_mf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("chain", "tri", "tricyl"), required=True)
    ap.add_argument("--size", required=True, help="N for chain, L1xL2 for tri")
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--arm", choices=("exact_sr", "gibbs", "mh"), required=True)
    ap.add_argument("--ns", type=int, default=200, help="samples (= chains) per SR step")
    ap.add_argument("--alpha", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--shift", type=float, default=None, help="SR diagonal shift (Trainer 'regularization')")
    ap.add_argument("--init", choices=("random", "mf"), default="random")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--steps", default="300,1000,3000")
    args = ap.parse_args()
    dlr, dsh = DEFAULTS.get(args.arm, (0.08, 0.05))
    args.lr = dlr if args.lr is None else args.lr
    args.shift = dsh if args.shift is None else args.shift
    if args.model in ("tri", "tricyl"):  # 3-sublattice colouring must close around periodic directions
        L1, L2 = map(int, args.size.split("x"))
        assert L2 % 3 == 0 and (L1 % 3 == 0 or args.model == "tricyl"), "periodic triangular lengths: multiples of 3"
    model = make(args.model, args.size, args.h)
    N = model.size
    if args.arm == "exact_sr":
        assert N <= ENUM_MAX_N and not (N > 16 and args.alpha > 2), "exact_sr: N <= 18 (alpha <= 2 at N > 16)"
        args.ns = 0
    V_all = all_configs(N) if N <= ENUM_MAX_N else None
    phase = sublattice_phase(args.model, args.size)
    OUT.mkdir(parents=True, exist_ok=True)
    e0, emf = references(args, model)
    out = OUT / f"{args.model}_{args.size}_h{args.h:g}.jsonl"
    name = run_name(args)
    lo, hi = map(int, args.seeds.split("-"))
    steps = sorted(int(x) for x in args.steps.split(","))
    log(f"start {name} seeds {lo}-{hi} steps {steps} on {jax.devices()[0].platform}")
    for seed in range(lo, hi + 1):
        with locked(OUT / "locks" / f"{name}_s{seed}.lock", blocking=False) as got:
            if not got:
                log(f"{name} seed={seed}: held by another worker, skipping")
                continue
            done = {(r["run"], r["seed"], r["step"]) for r in read_jsonl(out)}
            sf = OUT / "state" / f"{name}_s{seed}.npz"
            s = load_state(sf, model, args.alpha, seed, args.init)
            for step in steps:
                if step <= s["step"]:
                    continue
                t0 = time.time()
                Es = []
                while s["step"] < step:  # sub-segments: heartbeat; exact for both persistent samplers
                    to = min(step, s["step"] + SUBSEG)
                    seg = train_segment(args, model, s, to, V_all)
                    Es += seg
                    s["step"], s["energy"] = to, np.concatenate([s["energy"], seg])
                    log(f"{name} seed={seed} step={to} E_train/N={np.mean(seg[-10:]) / N:.5f}")
                s["wall_s"] += time.time() - t0
                save_npz(OUT / "ckpt" / f"{name}_s{seed}_step{step}.npz", {k: s[k] for k in "abW"})
                if (name, seed, step) not in done:
                    t1 = time.time()
                    ev = evaluate(model, s, V_all, phase, seed)
                    rec = dict(run=name, model=args.model, size=args.size, N=N, h=args.h, arm=args.arm,
                               ns=args.ns, alpha=args.alpha, lr=args.lr, shift=args.shift, init=args.init,
                               seed=seed, step=step, E0=e0, E_mf=emf, **ev, eval_s=time.time() - t1,
                               train_E_last10=float(np.mean(s["energy"][-10:])), wall_s=s["wall_s"],
                               device=str(jax.devices()[0].platform))
                    append_jsonl(out, rec)
                    ref = e0 if e0 is not None else emf
                    print(f"{name} seed={seed} step={step}: E/N={ev['E'] / N:.6f} "
                          f"(vs {'E0' if e0 is not None else 'MF'} {ref / N:.6f}; MF {emf / N:.6f}) "
                          f"var/N={ev['var'] / N:.2e} wall={s['wall_s']:.0f}s eval={rec['eval_s']:.0f}s", flush=True)
                save_state(sf, s)
    log(f"done {name}")


if __name__ == "__main__":
    main()

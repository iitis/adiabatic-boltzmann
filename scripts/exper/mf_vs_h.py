#!/usr/bin/env python3
"""Trained RBM vs the best product state (mean field) across the transverse field h, 1D TFIM.

Arms (full RBM, M = N, same random init per seed as the benchmark: PRNGKey(seed) -> model key):
  exact     SR with 200 i.i.d. samples drawn exactly from |Psi|^2 by enumeration (N <= 16)
  gibbs     SR with persistent block Gibbs, 200 chains, 10 sweeps per step (the paper's baseline)
  exact_sr  SR with exact expectations over all 2^N configurations (N <= 16): the ansatz ceiling
SR settings for exact/gibbs are the benchmark's (lr 0.08, reg 0.05, 200 samples) via encoder.Trainer.
Variants (any arm): --alpha K uses M = K*N hidden units; --init mf starts the visible biases at the best
product state, a_i = -artanh(cos alpha_MF) (W, b keep the random start; as "mfinit" in audit_schedule.py).
A variant is stored under its own name, e.g. gibbs_mfinit, gibbs_a2 (field "variant" in each record).

Runs are resumable and extendable: the state (parameters, sampler key and persistent chains, energy
trajectory) is saved under results/mf_vs_h/state/ at every checkpoint in --steps, and a later call
with a longer --steps list continues from the last saved step.
At each checkpoint the network is evaluated (exact enumeration for N <= 16, global-flip Metropolis
from audit_evaluate.py otherwise) and one line is appended to results/mf_vs_h/<arm>_N<N>_h<h>.jsonl.

  python scripts/exper/mf_vs_h.py --arm gibbs --N 64 --seeds 0-2 --steps 100,300
  python scripts/exper/mf_vs_h.py --arm gibbs --N 64 --seeds 0-2 --steps 100,300,1000,3000   # extend
  python scripts/exper/mf_vs_h.py --arm gibbs --N 64 --init mf --steps 100,300,1000,3000       # variant
"""
import argparse
import contextlib
import functools
import io
import itertools
import json
import os
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

from audit_evaluate import evaluate
from encoder import Trainer
from ising import TransverseFieldIsing1D, _local_energy_1d_jit
from mean_field import mean_field_angle, mean_field_energy
from model import FullyConnectedRBM, RBMParams
from sampler import ClassicalSampler

OUT = Path(os.environ.get("MF_VS_H_OUT", REPO / "results" / "mf_vs_h"))
H_GRID = (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0)
SR_CFG = dict(learning_rate=0.08, n_samples=200, regularization=0.05, save_checkpoints=False,
              use_cem=False, n_parallel=1)
EXACT_SR_LR, EXACT_SR_SHIFT = 0.05, 1e-4


def all_configs(N):
    return jnp.array(list(itertools.product([-1.0, 1.0], repeat=N)))


class CachedExactSampler:
    """Independent draws from |Psi|^2 by enumeration (configurations built once)."""

    def __init__(self, key, N):
        self._key, self.V = key, all_configs(N)

    def sample(self, rbm, n_samples, config=None, return_hidden=False, return_jax=False):
        phi = self.V @ rbm.W + rbm.b
        lp = -self.V @ rbm.a + jnp.sum(jnp.logaddexp(phi, -phi), axis=1)
        self._key, k = jax.random.split(self._key)
        v = self.V[jax.random.categorical(k, lp, shape=(n_samples,))]
        return v if return_jax else np.asarray(v)


def log_p(V, a, b, W):
    phi = V @ W + b
    return -V @ a + jnp.sum(jnp.logaddexp(phi, -phi), axis=-1)


def exact_moments(a, b, W, h, V):
    """Exact <H>, Var(H) and <|m|> by enumeration."""
    N = V.shape[1]
    lp = log_p(V, a, b, W)
    p = jnp.exp(lp - jax.scipy.special.logsumexp(lp))
    el = _local_energy_1d_jit(V, W, a, b, h, N)
    E = jnp.sum(p * el)
    return float(E), float(jnp.sum(p * (el - E) ** 2)), float(jnp.sum(p * jnp.abs(V.mean(1))))


@functools.partial(jax.jit, static_argnums=3)
def _exact_sr_step(a, b, W, h, V):
    N = V.shape[1]
    lp = log_p(V, a, b, W)
    p = jnp.exp(lp - jax.scipy.special.logsumexp(lp))
    el = _local_energy_1d_jit(V, W, a, b, h, N)
    t = jnp.tanh(V @ W + b)
    # d log Psi / d(a, b, W) for Psi = e^(-a.v/2) prod_j (2cosh phi_j)^(1/2)
    O = jnp.concatenate([-0.5 * V, 0.5 * t, 0.5 * (V[:, :, None] * t[:, None, :]).reshape(V.shape[0], -1)], 1)
    Oc = O - p @ O
    E = p @ el
    S = Oc.T @ (p[:, None] * Oc)
    F = Oc.T @ (p * (el - E))
    d = jnp.linalg.solve(S + EXACT_SR_SHIFT * jnp.eye(S.shape[0]), F)
    M = b.shape[0]
    return a - EXACT_SR_LR * d[:N], b - EXACT_SR_LR * d[N:N + M], W - EXACT_SR_LR * d[N + M:].reshape(N, M), E


def variant_name(arm, init="random", alpha=1):
    return arm + ("_mfinit" if init == "mf" else "") + (f"_a{alpha}" if alpha != 1 else "")


def state_path(name, N, h, seed):
    return OUT / "state" / f"{name}_N{N}_h{h:g}_s{seed}.npz"


def init_run(N, h, seed, init="random", alpha=1):
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    p = FullyConnectedRBM(N, alpha * N, model_key).params
    key, skey = jax.random.split(key)
    a = np.asarray(p.a)
    if init == "mf":
        a = np.full(N, -np.arctanh(np.cos(mean_field_angle(TransverseFieldIsing1D(N, h)))))
    return dict(step=0, a=a, b=np.asarray(p.b), W=np.asarray(p.W), skey=np.asarray(skey),
                gibbs_v=None, energy=np.zeros(0), wall_s=0.0)


def load_run(name, N, h, seed, init="random", alpha=1):
    f = state_path(name, N, h, seed)
    if not f.exists():
        return init_run(N, h, seed, init, alpha)
    z = np.load(f)
    s = {k: z[k] for k in z.files}
    s["step"], s["wall_s"] = int(s["step"]), float(s["wall_s"])
    s["gibbs_v"] = s["gibbs_v"] if s["gibbs_v"].size else None
    return s


def save_run(name, N, h, seed, s):
    f = state_path(name, N, h, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, **{**s, "gibbs_v": s["gibbs_v"] if s["gibbs_v"] is not None else np.zeros(0)})
    tmp.rename(f)


def train_segment(arm, N, h, s, to_step, V_all):
    if arm == "exact_sr":
        a, b, W = (jnp.asarray(s[k]) for k in "abW")
        Es = []
        for _ in range(s["step"], to_step):
            a, b, W, E = _exact_sr_step(a, b, W, h, V_all)
            Es.append(float(E))
        s.update(a=np.asarray(a), b=np.asarray(b), W=np.asarray(W))
        return Es
    ising = TransverseFieldIsing1D(N, h)
    rbm = FullyConnectedRBM(N, s["b"].shape[0], jax.random.PRNGKey(0))
    rbm.params = RBMParams(*(jnp.asarray(s[k]) for k in "abW"))
    if arm == "exact":
        sampler = CachedExactSampler(jnp.asarray(s["skey"]), N)
        sampler.V = V_all
    else:
        sampler = ClassicalSampler(method="gibbs", n_warmup=200, n_sweeps=10)
        sampler._key = jnp.asarray(s["skey"])
        sampler._gibbs_v = jnp.asarray(s["gibbs_v"]) if s["gibbs_v"] is not None else None
    cfg = dict(SR_CFG, n_iterations=to_step, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        hist = Trainer(rbm, ising, sampler, cfg, args=None).train(start_iteration=s["step"])
    s.update(a=np.asarray(rbm.a), b=np.asarray(rbm.b), W=np.asarray(rbm.W), skey=np.asarray(sampler._key))
    if arm == "gibbs":
        s["gibbs_v"] = np.asarray(sampler._gibbs_v)
    Es = [float(e) for e in hist["energy"]]
    if len(Es) != to_step - s["step"] or not np.all(np.isfinite(Es)):  # Trainer stops early on a NaN energy
        raise RuntimeError(f"segment {s['step']}->{to_step}: {len(Es)} steps, finite={np.all(np.isfinite(Es))}")
    return Es


def read_jsonl(path):
    """Records of a jsonl file, skipping a line left partial by a killed writer."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_jsonl(path, rec):
    """Append one record on its own line (starts a new line if a killed writer left a partial one)."""
    lead = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a") as f:
        f.write(lead + json.dumps(rec) + "\n")


def evaluate_net(s, N, h, V_all, seed):
    if V_all is not None:
        E, var, absm = exact_moments(*(jnp.asarray(s[k]) for k in "abW"), h, V_all)
        return dict(E=E, se=0.0, var=var, absm=absm, evaluator="enumeration")
    r = evaluate(s["a"], s["b"], s["W"], h=h, seed=seed)
    return dict(E=r["E"], se=r["se"], var=None, absm=r["absm"], dw=r["dw"], rhat=r["rhat"],
                E_groups=r["E_groups"], evaluator="global_flip_metropolis")


def with_eval(rec, ev):
    err = (ev["E"] - rec["E0"]) / rec["N"]
    return dict(ev, err=err, err_over_mf=err / rec["mf_err"])


def evaluate_pending(poll_s, pattern="*.jsonl"):
    """Evaluate deferred checkpoints (on the default device, i.e. the GPU) until none are left.

    With poll_s > 0, keep polling while training processes are still writing, and stop after a pass that
    started with OUT/TRAINING_DONE present and found nothing new. Several evaluators can share the work
    with disjoint --eval-pattern globs (each <file>.eval.jsonl has a single writer)."""
    while True:
        training_done = (OUT / "TRAINING_DONE").exists()  # read before the pass: no lost last record
        n_new = 0
        for f in sorted(OUT.glob(pattern)):
            if f.name.endswith(".eval.jsonl"):
                continue
            ef = f.with_name(f.stem + ".eval.jsonl")
            have = {(r["seed"], r["step"]) for r in read_jsonl(ef)}
            for r in read_jsonl(f):
                if r["E"] is not None or (r["seed"], r["step"]) in have:
                    continue
                s = {k: np.asarray(r[k]) for k in "abW"}
                ev = with_eval(r, evaluate_net(s, r["N"], r["h"], None, r["seed"]))
                append_jsonl(ef, dict(seed=r["seed"], step=r["step"], **ev))
                n_new += 1
                print(f"eval {f.stem} seed={r['seed']} step={r['step']}: err/N={ev['err']:.2e} "
                      f"ratio={ev['err_over_mf']:.3f}", flush=True)
        if poll_s <= 0 or (n_new == 0 and training_done):
            return
        if n_new == 0:
            time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate-pending", type=float, metavar="POLL_S", default=None,
                    help="evaluate deferred checkpoints; poll every POLL_S s until OUT/TRAINING_DONE exists (0: one pass)")
    ap.add_argument("--defer-eval", action="store_true",
                    help="N > 16: skip the Metropolis evaluation here (run --evaluate-pending on a GPU instead)")
    ap.add_argument("--eval-pattern", default="*.jsonl", help="glob of result files for --evaluate-pending")
    ap.add_argument("--arm", choices=("exact", "gibbs", "exact_sr"))
    ap.add_argument("--init", choices=("random", "mf"), default="random")
    ap.add_argument("--alpha", type=int, default=1, help="hidden units per visible unit")
    ap.add_argument("--N", type=int)
    ap.add_argument("--hs", default=",".join(f"{h:g}" for h in H_GRID))
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--steps", default="100,300")
    args = ap.parse_args()
    if args.evaluate_pending is not None:
        return evaluate_pending(args.evaluate_pending, args.eval_pattern)
    N, arm = args.N, args.arm
    name = variant_name(arm, args.init, args.alpha)
    if arm != "gibbs" and N > 16:
        sys.exit(f"--arm {arm} enumerates 2^N configurations; N <= 16 only")
    hs = [float(x) for x in args.hs.split(",")]
    lo, hi = map(int, args.seeds.split("-"))
    steps = sorted(int(x) for x in args.steps.split(","))
    V_all = all_configs(N) if N <= 16 else None
    OUT.mkdir(parents=True, exist_ok=True)

    for h in hs:
        out = OUT / f"{name}_N{N}_h{h:g}.jsonl"  # one file per process-sized unit: no interleaved appends
        done = {(r["seed"], r["step"]) for r in read_jsonl(out)}
        ising = TransverseFieldIsing1D(N, h)
        e0, emf = ising.exact_ground_energy(), mean_field_energy(ising)
        for seed in range(lo, hi + 1):
            s = load_run(name, N, h, seed, args.init, args.alpha)
            for step in steps:
                if step <= s["step"]:
                    continue
                t0 = time.time()
                Es = train_segment(arm, N, h, s, step, V_all)
                s["step"], s["energy"] = step, np.concatenate([s["energy"], Es])
                s["wall_s"] += time.time() - t0
                if (seed, step) in done:  # record exists (state was lost after it): just re-save the state
                    save_run(name, N, h, seed, s)
                    continue
                rec = dict(arm=arm, variant=name, init=args.init, alpha=args.alpha,
                           N=N, h=h, seed=seed, step=step, E0=e0, E_mf=emf, mf_err=(emf - e0) / N,
                           train_E_last10=float(np.mean(s["energy"][-10:])), wall_s=s["wall_s"],
                           a=s["a"].tolist(), b=s["b"].tolist(), W=s["W"].tolist())
                if args.defer_eval and V_all is None:
                    rec["E"] = None  # filled in later by --evaluate-pending (results in <file>.eval.jsonl)
                else:
                    rec.update(with_eval(rec, evaluate_net(s, N, h, V_all, seed)))
                # Record first, then state: a crash in between retrains this segment (deterministic) on restart
                # and finds the record already written, instead of skipping a checkpoint that has no record.
                append_jsonl(out, rec)
                save_run(name, N, h, seed, s)
                print(f"{name} N={N} h={h:g} seed={seed} step={step}: " + (
                    f"err/N={rec['err']:.2e} MF={rec['mf_err']:.2e} ratio={rec['err_over_mf']:.3f}"
                    if rec["E"] is not None else "eval deferred") + f" wall={s['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

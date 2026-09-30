"""Zephyr (Advantage2) +CEM runs under a calibration protocol, budget-guarded.

Same training as the report headline (h=0.5, full RBM, lr=0.08, reg=0.05,
ns=200, 100 training iters, CEM every 5 iters, log-EMA alpha=0.3, beta_x_init=1)
plus the protocol knobs of cem_zephyr_trap_gpu.Scheduled (calib=K frozen
full-step CEM draws before training; boot=1 full-step first reading;
--fb cem+pl: after calibration, beta_x tracks the visible-marginal
pseudo-likelihood temperature instead of the joint-(v,h) CEM).
Each run is scored with an unbiased CPU Metropolis <H>_Psi.

Device time is tracked in a private ledger (sum of per-call qpu_access_time
from each run's history), shared by concurrent processes via flock. A run is
only started if ledger + n_procs * RESERVE_S stays below --cap-s.

Usage (from repo root):
  python scripts/exper/cem_zephyr_protocol_qpu.py TAG N SEEDS --calib 3 --cap-s 470 --n-procs 3
"""
import argparse, contextlib, fcntl, io, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
assert Path.cwd().resolve() == REPO, f"must run from repo root, cwd={Path.cwd()}"
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from sampler import DimodSampler
from encoder import Trainer
from cem_zephyr_trap_gpu import Scheduled, score, parse_seeds

OUT = REPO / "results" / "cem_zephyr_protocol_qpu"
LEDGER = OUT / "ledger.jsonl"
RESERVE_S = 9.0  # worst-case device time of one run (observed ~6.2-6.5 s for 100 iters)


def ledger_s():
    if not LEDGER.exists():
        return 0.0
    with LEDGER.open() as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        return sum(json.loads(l)["qpu_s"] for l in f if l.strip())


def ledger_add(rec):
    with LEDGER.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")


class Recording(Scheduled):
    """Scheduled + per-draw sample diagnostics (domain walls, |m|) and last V."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dw, self.absm, self.lastV = [], [], None

    def sample(self, rbm, n_samples, config={}, return_hidden=False):
        r = super().sample(rbm, n_samples, config, return_hidden)
        V = np.asarray(r[0] if return_hidden else r)
        self.dw.append(float(np.mean(np.sum(V != np.roll(V, -1, 1), 1))))
        self.absm.append(float(np.mean(np.abs(V.mean(1)))))
        self.lastV = V.astype(np.int8)
        return r


def run(tag, N, seed, sampler, a):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    ising = TransverseFieldIsing1D(N, 0.5)
    rbm = FullyConnectedRBM(N, N, mk)
    smp = Recording(sampler, 0.08, 0, 0, a.calib, a.boot, a.ci, a.alpha)
    cfg = dict(learning_rate=0.08, n_iterations=100 + a.calib, n_samples=200, regularization=0.05,
               use_cem=a.fb != "pl", beta_adapt=0.05 if a.fb == "cem" else 0.0,
               cem_interval=a.ci, cem_ema_alpha=a.alpha, seed=seed, n_parallel=1, beta_x_init=1.0)
    tr = Trainer(rbm, ising, smp, cfg, args=None)
    smp.trainer, smp.fb, smp.bx = tr, a.fb, 1.0
    t0 = time.time()
    hist = tr.train()
    wall = time.time() - t0
    qpu_s = float(sum(hist["sampling_time_s"]))
    ledger_add(dict(tag=tag, N=N, seed=seed, qpu_s=qpu_s, t=time.time()))
    sc = score(rbm, ising)
    return dict(tag=tag, N=N, seed=seed, calib=a.calib, boot=a.boot, fb=a.fb, ci=a.ci, alpha=a.alpha,
                exact=ising.exact_ground_energy(), E=hist["energy"],
                beta_x=hist["beta_x"] if a.fb == "cem" else smp.bx_hist, s_pl=smp.s_pl, beta_eff_cem=hist["beta_eff_cem"],
                sampling_time_s=hist["sampling_time_s"], qpu_s=qpu_s, wall_s=wall,
                sample_dw=smp.dw, sample_absm=smp.absm, lastV=smp.lastV.tolist(), **sc,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag"); p.add_argument("N", type=int); p.add_argument("seeds")
    p.add_argument("--calib", type=int, default=0); p.add_argument("--boot", type=int, default=0)
    p.add_argument("--fb", default="cem", choices=["cem", "pl", "cem+pl"])
    p.add_argument("--ci", type=int, default=5); p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--cap-s", type=float, required=True); p.add_argument("--n-procs", type=int, default=1)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}_N{a.N}.jsonl"
    sampler = DimodSampler(method="zephyr")
    for sd in parse_seeds(a.seeds):
        if out.exists() and any(json.loads(l)["seed"] == sd for l in open(out)):
            continue
        used = ledger_s()
        if used + a.n_procs * RESERVE_S > a.cap_s:
            print(f"[budget] ledger {used:.1f}s + reserve > cap {a.cap_s}s; stopping", flush=True)
            break
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(a.tag, a.N, sd, sampler, a)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        rel = (r["E_true"] - r["exact"]) / abs(r["exact"]) * 100
        print(f"{a.tag} N={a.N} seed={sd}: true err={rel:+.2f}% dw={r['dw']:.2f} |m|={r['absm']:.2f} "
              f"beta_x: first-train={r['beta_x'][a.calib]:.2f} final={r['beta_x'][-1]:.2f}  "
              f"qpu={r['qpu_s']:.2f}s wall={r['wall_s']:.0f}s  ledger={ledger_s():.1f}s", flush=True)

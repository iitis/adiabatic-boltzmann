"""Zephyr (Advantage2) or Pegasus (Advantage) +CEM runs under a calibration protocol, budget-guarded.

Same training as the report headline (h=0.5, full RBM, lr=0.08, reg=0.05,
ns=200, 100 training iters, CEM every 5 iters, log-EMA alpha=0.3, beta_x_init=1)
plus the protocol knobs of cem_zephyr_trap_gpu.Scheduled (calib=K frozen
full-step CEM draws before training; boot=1 full-step first reading;
--fb cem+pl: after calibration, beta_x tracks the visible-marginal
pseudo-likelihood temperature instead of the joint-(v,h) CEM).
Each run is scored with an unbiased CPU Metropolis <H>_Psi.

Budget: time.json in the repo root is the ground truth. DimodSampler adds every
call's qpu_access_time to it under a lock shared by all processes. --cap-s is
an absolute limit on time.json (in seconds). It is checked
  - before each run: a run starts only if time.json + n_procs * RESERVE_S <= cap;
  - before every QPU call: if time.json >= cap the process raises and stops.
A missing or unreadable time.json is an error, never a zero.
The per-run ledger.jsonl in the output dir is only a record, not the budget.

--device pegasus writes to results/cem_pegasus_protocol_qpu/,
so the Zephyr results and ledger are never touched.

Usage (from repo root):
  python scripts/exper/cem_zephyr_protocol_qpu.py TAG N SEEDS --calib 3 --cap-s 470 --n-procs 3
  Pegasus P4 rerun: scripts/exper/cem_pegasus_p4_launch.sh
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
import jax.numpy as jnp
from helpers import get_solver_name
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from sampler import DimodSampler
from encoder import Trainer, estimate_beta_eff_cem, estimate_beta_visible_pl, is_cem_fit_degenerate
from cem_zephyr_trap_gpu import Scheduled, score, parse_seeds

OUT_DIRS = {"zephyr": REPO / "results" / "cem_zephyr_protocol_qpu",
            "pegasus": REPO / "results" / "cem_pegasus_protocol_qpu"}
OUT = LEDGER = None  # set from --device in __main__
RESERVE_S = 9.0  # worst-case device time of one run (observed ~6.2-6.5 s for 100 iters)
TIME_JSON = REPO / "time.json"


class BudgetExceeded(RuntimeError):
    pass


def device_s():
    """Accumulated device time from time.json, in seconds. Raises if missing/unreadable."""
    if not TIME_JSON.exists():
        raise FileNotFoundError(f"{TIME_JSON} missing; it is the QPU budget ground truth, refusing to run")
    with TIME_JSON.open() as f:
        return float(json.load(f)["time_ms"]) / 1000.0


class BudgetedDimodSampler(DimodSampler):
    """DimodSampler that refuses every QPU call once time.json reaches cap_s."""

    def __init__(self, method, cap_s):
        super().__init__(method)
        self.cap_s = cap_s

    def sample(self, *args, **kwargs):
        used = device_s()
        if used >= self.cap_s:
            raise BudgetExceeded(f"time.json at {used:.1f}s >= cap {self.cap_s}s; refusing QPU call")
        return super().sample(*args, **kwargs)


def ledger_add(rec):
    with LEDGER.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")


class Recording(Scheduled):
    """Scheduled + per-draw diagnostics and the raw data to re-fit any estimator offline.

    Hidden units are always requested from the device (same call, they are sampled anyway),
    so every draw keeps its joint (v, h) samples together with the parameters it was drawn
    from. Per draw (lists, one entry per QPU call, calibration included):
      cem_hat / cem_degenerate: pooled CEM fit of the joint samples (estimates beta_hw/beta_x)
      s_pl_all: visible pseudo-likelihood temperature (s_pl is the subset used for feedback)
      timing: the device's full timing dict; problem_id; chain_break: mean chain-break fraction
      wall_call_s: wall-clock time of the call; n_unique: distinct visible configurations
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dw, self.absm, self.lastV = [], [], None
        self.cem_hat, self.cem_degenerate, self.s_pl_all = [], [], []
        self.timing, self.problem_id, self.chain_break, self.wall_call_s, self.n_unique = [], [], [], [], []
        self.raw_V, self.raw_H, self.raw_a, self.raw_b, self.raw_W = [], [], [], [], []

    def sample(self, rbm, n_samples, config={}, return_hidden=False):
        prm = rbm.params  # parameters these samples are drawn from (before the SR update)
        t0 = time.time()
        V, H = super().sample(rbm, n_samples, config, True)
        self.wall_call_s.append(time.time() - t0)
        Vn, Hn = np.asarray(V), np.asarray(H)
        self.dw.append(float(np.mean(np.sum(Vn != np.roll(Vn, -1, 1), 1))))
        self.absm.append(float(np.mean(np.abs(Vn.mean(1)))))
        self.n_unique.append(int(len(np.unique(Vn, axis=0))))
        self.lastV = Vn.astype(np.int8)
        Vj, Hj = jnp.asarray(Vn, dtype=jnp.float64), jnp.asarray(Hn, dtype=jnp.float64)
        bh = estimate_beta_eff_cem(Vj, Hj, rbm)
        self.cem_hat.append(bh); self.cem_degenerate.append(bool(is_cem_fit_degenerate(bh)))
        self.s_pl_all.append(float(estimate_beta_visible_pl(Vj, rbm)))
        ss = self.inner.last_sampleset
        self.timing.append({k: float(v) for k, v in ss.info["timing"].items()})
        self.problem_id.append(ss.info.get("problem_id"))
        cbf = ss.record["chain_break_fraction"] if "chain_break_fraction" in ss.record.dtype.names else None
        self.chain_break.append(None if cbf is None else
                                float(np.average(cbf, weights=ss.record["num_occurrences"])))
        self.raw_V.append(Vn.astype(np.int8)); self.raw_H.append(Hn.astype(np.int8))
        self.raw_a.append(np.asarray(prm.a)); self.raw_b.append(np.asarray(prm.b)); self.raw_W.append(np.asarray(prm.W))
        return (V, H) if return_hidden else V

    def save_raw(self, path):
        """Write the per-draw samples and parameters; never overwrites an existing file."""
        if path.exists():
            path = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, V=np.stack(self.raw_V), H=np.stack(self.raw_H),
                                a=np.stack(self.raw_a), b=np.stack(self.raw_b), W=np.stack(self.raw_W),
                                beta_x=np.asarray(self.bx_hist))
        tmp.rename(path)
        return path


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


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
    ledger_add(dict(tag=tag, device=a.device, N=N, seed=seed, qpu_s=qpu_s, t=time.time()))
    sc = score(rbm, ising)
    raw = smp.save_raw(OUT / "raw" / f"{tag}_N{N}_seed{seed}.npz")
    return dict(tag=tag, device=a.device, N=N, seed=seed, calib=a.calib, boot=a.boot, fb=a.fb, ci=a.ci, alpha=a.alpha,
                exact=ising.exact_ground_energy(), E=hist["energy"],
                beta_x=hist["beta_x"] if a.fb == "cem" else smp.bx_hist, s_pl=smp.s_pl, beta_eff_cem=hist["beta_eff_cem"],
                sampling_time_s=hist["sampling_time_s"], qpu_s=qpu_s, wall_s=wall,
                sample_dw=smp.dw, sample_absm=smp.absm, lastV=smp.lastV.tolist(), **sc,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist(),
                cem_hat=smp.cem_hat, cem_degenerate=smp.cem_degenerate, s_pl_all=smp.s_pl_all,
                timing=smp.timing, problem_id=smp.problem_id, chain_break=smp.chain_break,
                wall_call_s=smp.wall_call_s, n_unique=smp.n_unique,
                embedding=_jsonable(sampler.last_embedding_info), solver=get_solver_name(a.device),
                history=_jsonable(hist), raw_file=str(raw.relative_to(REPO)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag"); p.add_argument("N", type=int); p.add_argument("seeds")
    p.add_argument("--calib", type=int, default=0); p.add_argument("--boot", type=int, default=0)
    p.add_argument("--fb", default="cem", choices=["cem", "pl", "cem+pl"])
    p.add_argument("--ci", type=int, default=5); p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--cap-s", type=float, required=True); p.add_argument("--n-procs", type=int, default=1)
    p.add_argument("--device", default="zephyr", choices=sorted(OUT_DIRS))
    a = p.parse_args()
    OUT = OUT_DIRS[a.device]
    LEDGER = OUT / "ledger.jsonl"
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}_N{a.N}.jsonl"
    device_s()  # fail now, not mid-run, if time.json is missing/unreadable
    sampler = BudgetedDimodSampler(a.device, a.cap_s)
    for sd in parse_seeds(a.seeds):
        if out.exists() and any(json.loads(l)["seed"] == sd for l in open(out)):
            continue
        used = device_s()
        if used + a.n_procs * RESERVE_S > a.cap_s:
            print(f"[budget] time.json {used:.1f}s + reserve {a.n_procs * RESERVE_S:.0f}s > cap {a.cap_s}s; stopping", flush=True)
            break
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(a.tag, a.N, sd, sampler, a)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        rel = (r["E_true"] - r["exact"]) / abs(r["exact"]) * 100
        print(f"{a.tag} N={a.N} seed={sd}: true err={rel:+.2f}% dw={r['dw']:.2f} |m|={r['absm']:.2f} "
              f"beta_x: first-train={r['beta_x'][a.calib]:.2f} final={r['beta_x'][-1]:.2f}  "
              f"qpu={r['qpu_s']:.2f}s wall={r['wall_s']:.0f}s  time.json={device_s():.1f}s/{a.cap_s:.0f}s", flush=True)

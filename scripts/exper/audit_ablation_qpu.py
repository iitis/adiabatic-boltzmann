"""Pegasus ablation of the beta_x protocol for the audit response (budget-guarded).

Same training, initialization (PRNGKey(seed)), recording and scoring as the P4 runs of
cem_zephyr_protocol_qpu.py, so each run pairs with the P4 run of the same seed. Modes:
  fixed      beta_x constant at --bx (no fits used)
  auto       auto_scale=True, no estimation (device default rescaling)
  calib_only 3 CEM calibration draws (as P4), then beta_x held
  pl_only    no calibration, visible-PL feedback from beta_x=1 (P4 rule: clip [2/3,3/2], sqrt)
  cem_step   3 CEM calibration draws, then CEM feedback every step with the P4 clip and sqrt
P4 itself (calibration + PL feedback) is the existing results/cem_pegasus_protocol_qpu data.

  python scripts/exper/audit_ablation_qpu.py MODE N SEEDS --cap-s ABS [--bx 2.9]
--cap-s is an absolute limit on time.json, as in cem_zephyr_protocol_qpu.py.
"""
import argparse, contextlib, io, json, time
import numpy as np
import cem_zephyr_protocol_qpu as P  # asserts cwd == repo root, sets up paths and x64
from cem_zephyr_protocol_qpu import (Recording, BudgetedDimodSampler, device_s, ledger_add, score,
                                     parse_seeds, _jsonable, REPO, RESERVE_S)
from cem_zephyr_protocol_qpu import jax, jnp, FullyConnectedRBM, TransverseFieldIsing1D, Trainer
from encoder import estimate_beta_eff_cem, is_cem_fit_degenerate

OUT = REPO / "results" / "audit_ablation_qpu"


class Ablation(Recording):
    """Recording with a per-mode beta_x control (Scheduled.sample is replaced below)."""
    mode = "fixed"


def _controlled(self, rbm, n_samples, config={}, return_hidden=False):
    """Replaces Scheduled.sample for the ablation modes."""
    tr, m = self.trainer, self.mode
    if m == "pl_only" or (m in ("calib_only", "cem_step") and self.t < self.calib):
        return _scheduled_sample(self, rbm, n_samples, config, return_hidden)  # P4 code path
    if m in ("calib_only", "cem_step") and self.t == self.calib:
        tr.use_cem, self.bx = False, tr.beta_x  # hand over from the CEM calibration
    tr.learning_rate = self.lr
    cfg = {**config, "beta_x": self.bx}
    if m == "auto":
        cfg["auto_scale"] = True
    V, H = self.inner.sample(rbm, n_samples, cfg, True)
    self.bx_hist.append(self.bx)
    if m == "cem_step":
        bh = float(estimate_beta_eff_cem(jnp.asarray(V, dtype=jnp.float64), jnp.asarray(H, dtype=jnp.float64), rbm))
        if not is_cem_fit_degenerate(bh):
            self.bx = float(np.clip(self.bx * float(np.clip(bh, 1 / 1.5, 1.5)) ** self.alpha, 0.05, 20.0))
    tr.beta_x = self.bx
    self.t += 1
    return (V, H) if return_hidden else V


_scheduled_sample = P.Scheduled.sample
P.Scheduled.sample = _controlled  # only this process; Recording.sample's super() now lands here


def run(mode, N, seed, sampler, bx):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    ising = TransverseFieldIsing1D(N, 0.5)
    rbm = FullyConnectedRBM(N, N, mk)
    calib = 3 if mode in ("calib_only", "cem_step") else 0
    smp = Ablation(sampler, 0.08, 0, 0, calib, 0, 1, 0.5)
    smp.mode = mode
    cfg = dict(learning_rate=0.08, n_iterations=100 + calib, n_samples=200, regularization=0.05,
               use_cem=mode in ("calib_only", "cem_step", "pl_only"), beta_adapt=0.0,
               cem_interval=1, cem_ema_alpha=0.5, seed=seed, n_parallel=1, beta_x_init=1.0)
    tr = Trainer(rbm, ising, smp, cfg, args=None)
    smp.trainer, smp.fb = tr, "cem+pl"
    smp.bx = bx if mode == "fixed" else 1.0
    t0 = time.time()
    hist = tr.train()
    wall = time.time() - t0
    qpu_s = float(sum(hist["sampling_time_s"]))
    ledger_add(dict(tag=mode, device="pegasus", N=N, seed=seed, qpu_s=qpu_s, t=time.time()))
    sc = score(rbm, ising)
    raw = smp.save_raw(OUT / "raw" / f"{mode}_N{N}_seed{seed}.npz")
    return dict(tag=mode, device="pegasus", N=N, seed=seed, calib=calib, bx_fixed=bx if mode == "fixed" else None,
                exact=ising.exact_ground_energy(), E=hist["energy"], beta_x=smp.bx_hist, s_pl=smp.s_pl,
                sampling_time_s=hist["sampling_time_s"], qpu_s=qpu_s, wall_s=wall,
                sample_dw=smp.dw, sample_absm=smp.absm, lastV=smp.lastV.tolist(), **sc,
                a=np.asarray(rbm.a).tolist(), b=np.asarray(rbm.b).tolist(), W=np.asarray(rbm.W).tolist(),
                cem_hat=smp.cem_hat, cem_degenerate=smp.cem_degenerate, s_pl_all=smp.s_pl_all,
                timing=smp.timing, problem_id=smp.problem_id, chain_break=smp.chain_break,
                wall_call_s=smp.wall_call_s, n_unique=smp.n_unique,
                embedding=_jsonable(sampler.last_embedding_info), solver="Advantage_system6",
                history=_jsonable(hist), raw_file=str(raw.relative_to(REPO)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["fixed", "auto", "calib_only", "pl_only", "cem_step"])
    p.add_argument("N", type=int); p.add_argument("seeds")
    p.add_argument("--cap-s", type=float, required=True); p.add_argument("--n-procs", type=int, default=1)
    p.add_argument("--bx", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="simulated annealing instead of the QPU, scratch output")
    a = p.parse_args()
    if a.smoke:
        OUT = REPO / "results" / "audit_ablation_smoke"
    assert a.mode != "fixed" or a.bx, "--bx required for fixed"
    P.OUT, P.LEDGER = OUT, OUT / "ledger.jsonl"
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.mode}_N{a.N}.jsonl"
    device_s()
    sampler = BudgetedDimodSampler("simulated_annealing" if a.smoke else "pegasus", a.cap_s)
    if a.smoke:  # neal stand-in with the attributes Recording reads from a QPU call
        import dimod
        rng = np.random.default_rng(0)
        def _sa(bqm, n_samples, config={}, return_hidden=False):
            # thermal stand-in: block Gibbs of the programmed energy at a hidden beta_hw=2.5
            n = sampler.n_visible
            hv = np.array([bqm.linear[i] for i in range(n)]); hh = np.array([bqm.linear[n + j] for j in range(n)])
            J = np.zeros((n, n))
            for (u, v), w in bqm.quadratic.items():
                i, j = (u, v - n) if u < n else (v, u - n)
                J[i, j] = w
            pm = lambda f: np.where(rng.random(f.shape) < 1 / (1 + np.exp(2 * 2.5 * f)), 1, -1)
            v = pm(np.zeros((n_samples, n)))
            for _ in range(200):
                u = pm(hh + v @ J); v = pm(hv + u @ J.T)
            u = pm(hh + v @ J)
            ss = dimod.SampleSet.from_samples(np.hstack([v, u]), "SPIN", 0)
            ss.info["timing"] = {"qpu_access_time": 0.0}
            sampler.last_sampleset, sampler.last_sampling_time_s = ss, 0.0
            return v, u
        sampler.simulated_annealing = _sa
        sampler.last_embedding_info = {}
    for sd in parse_seeds(a.seeds):
        if out.exists() and any(json.loads(l)["seed"] == sd for l in open(out)):
            continue
        used = device_s()
        if used + a.n_procs * RESERVE_S > a.cap_s:
            print(f"[budget] time.json {used:.1f}s + reserve > cap {a.cap_s}s; stopping", flush=True)
            break
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(a.mode, a.N, sd, sampler, a.bx)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        rel = (r["E_true"] - r["exact"]) / abs(r["exact"]) * 100
        print(f"{a.mode} N={a.N} seed={sd}: true err={rel:+.3f}% dw={r['dw']:.2f} beta_x final={r['beta_x'][-1]:.2f} "
              f"qpu={r['qpu_s']:.2f}s wall={r['wall_s']:.0f}s time.json={device_s():.1f}s", flush=True)

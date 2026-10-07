"""Frozen-network QPU sampling for the audit response (no training).

Re-samples stored P4 networks at fixed parameters and fixed beta_x, recording what the
training runs did not: hidden spins, physical-qubit samples, the embedding, the numerical
chain strength and per-read chain breaks. Used for the two-conditional temperature test
(beta from u|v vs v|u), same-parameter visible diagnostics and chain-strength scans.

Experiments (all 2000 reads, 20 us anneal, auto_scale=False, raw answers):
  cs   : Pegasus and Zephyr, N=8..64, seeds 0-3, chain strength x{0.5,1,2} of the Ocean default
  bx   : Pegasus N=16, seeds 0-3, beta_x x{0.7,1.4} at default chain strength

    python scripts/exper/audit_frozen_qpu.py --dry-run     # build problems, check ranges, no QPU
    python scripts/exper/audit_frozen_qpu.py --cap-s 120   # sample, stop if this script used >cap
"""
import argparse, fcntl, json, time
from pathlib import Path
import numpy as np
import dimod
from dwave.system import DWaveSampler
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
import minorminer.busclique as bc

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "audit_frozen_qpu"
SOLVER = {"pegasus": "Advantage_system6", "zephyr": "Advantage2_system1"}
READS, ANNEAL = 2000, 20


def load_network(device, N, seed):
    """Parameters and beta_x of the last training call (Pegasus: exact call params; Zephyr: final params)."""
    if device == "pegasus":
        d = np.load(RES / f"cem_pegasus_protocol_qpu/raw/P4cemPL_N{N}_seed{seed}.npz")
        return d["a"][-1], d["b"][-1], d["W"][-1], float(d["beta_x"][-1]), "params_of_last_call"
    if N == 8:  # Zephyr N=8 also has raw files
        d = np.load(RES / f"cem_zephyr_protocol_qpu/raw/P4cemPL_N8_seed{seed}.npz")
        return d["a"][-1], d["b"][-1], d["W"][-1], float(d["beta_x"][-1]), "params_of_last_call"
    for line in open(RES / f"cem_zephyr_protocol_qpu/P4cemPL_N{N}.jsonl"):
        r = json.loads(line)
        if r["seed"] == seed:
            return (np.array(r["a"]), np.array(r["b"]), np.array(r["W"]), float(r["beta_x"][-1]),
                    "final_params_after_last_update")
    raise KeyError((device, N, seed))


def logical_bqm(a, b, W, beta_x):
    """Same mapping as Sampler.rbm_to_ising: visible i -> i, hidden j -> N+j."""
    N, M = W.shape
    h = {i: a[i] / beta_x for i in range(N)} | {N + j: -b[j] / beta_x for j in range(M)}
    J = {(i, N + j): -W[i, j] / beta_x for i in range(N) for j in range(M) if abs(W[i, j]) > 1e-6}
    return dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)


def in_range(tbqm, props):
    hl, hh = props["h_range"]; jl, jh = props["extended_j_range"]
    ql, qh = props.get("per_qubit_coupling_range", (-np.inf, np.inf))
    lin, quad = tbqm.linear, tbqm.quadratic
    if not all(hl <= v <= hh for v in lin.values()) or not all(jl <= v <= jh for v in quad.values()):
        return False
    tot = {}
    for (u, v), w in quad.items():
        tot[u] = tot.get(u, 0) + w; tot[v] = tot.get(v, 0) + w
    return all(ql <= t <= qh for t in tot.values())


def log_time(us, cap_s, used):
    """Add to the shared time.json counter (same lock as Sampler._log_access_time)."""
    tp = ROOT / "time.json"
    with (tp.with_name("time.json.lock")).open("a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            d = json.load(tp.open()); d["time_ms"] += us * 1e-3
            tmp = tp.with_name("time.json.tmp"); json.dump(d, tmp.open("w")); tmp.replace(tp)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return used + us * 1e-6


def plan():
    jobs = []
    for device in ["pegasus", "zephyr"]:
        for N in [8, 16, 32, 64]:
            for seed in range(4):
                for cs in [1.0, 0.5, 2.0]:
                    jobs.append(dict(exp="cs", device=device, N=N, seed=seed, cs_mult=cs, bx_mult=1.0))
    for seed in range(4):
        for bx in [0.7, 1.4]:
            jobs.append(dict(exp="bx", device="pegasus", N=16, seed=seed, cs_mult=1.0, bx_mult=bx))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap-s", type=float, default=120.0, help="max QPU access time for this script")
    ap.add_argument("--only", default=None, help="run only jobs whose exp matches")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samplers, embs, used = {}, {}, 0.0
    for job in plan():
        if a.only and job["exp"] != a.only:
            continue
        tag = "{exp}_{device}_N{N}_seed{seed}_cs{cs_mult}_bx{bx_mult}".format(**job)
        path = OUT / f"{tag}.npz"
        if path.exists():
            continue
        dev, N = job["device"], job["N"]
        if dev not in samplers:
            samplers[dev] = DWaveSampler(solver=SOLVER[dev])
        smp = samplers[dev]
        if (dev, N) not in embs:
            emb = bc.busgraph_cache(smp.to_networkx_graph()).find_biclique_embedding(N, N)
            embs[(dev, N)] = {int(k): [int(q) for q in v] for k, v in emb.items()}
        emb = embs[(dev, N)]
        pa, pb, pW, bx0, src = load_network(dev, N, job["seed"])
        bx = bx0 * job["bx_mult"]
        bqm = logical_bqm(pa, pb, pW, bx)
        cs0 = uniform_torque_compensation(bqm, emb)
        cs = cs0 * job["cs_mult"]
        tbqm = embed_bqm(bqm, emb, smp.adjacency, chain_strength=cs)
        ok = in_range(tbqm, smp.properties)
        print(f"{tag}: beta_x={bx:.3f} cs={cs:.3f} qubits={len(tbqm)} in_range={ok}", flush=True)
        if a.dry_run or not ok:
            continue
        if used > a.cap_s:
            print(f"cap reached ({used:.1f}s)"); break
        ss = smp.sample(tbqm, num_reads=READS, annealing_time=ANNEAL, answer_mode="raw", auto_scale=False)
        us = ss.info["timing"]["qpu_access_time"]
        used = log_time(us, a.cap_s, used)
        lss = unembed_sampleset(ss, emb, bqm, chain_break_method=majority_vote, chain_break_fraction=True)
        L = lss.record.sample[:, np.argsort(lss.variables)]
        phys_vars = np.array(ss.variables)
        np.savez_compressed(
            path, V=L[:, :N].astype(np.int8), H=L[:, N:2 * N].astype(np.int8),
            cbf=lss.record.chain_break_fraction, phys=ss.record.sample.astype(np.int8), phys_vars=phys_vars,
            a=pa, b=pb, W=pW, beta_x=bx, chain_strength=cs, chain_strength_default=cs0,
            embedding=json.dumps(emb), timing=json.dumps({k: float(v) for k, v in ss.info["timing"].items()}),
            job=json.dumps(job), param_source=src, solver=SOLVER[dev], problem_id=ss.info.get("problem_id", ""),
            t=time.time())
        print(f"   qpu {us*1e-3:.1f} ms, total {used:.2f} s, mean cbf {lss.record.chain_break_fraction.mean():.3f}", flush=True)
    print(f"done, QPU access time used by this run: {used:.2f} s")


if __name__ == "__main__":
    main()

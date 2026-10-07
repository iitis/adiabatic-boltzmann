"""Full protocol (P4) on Pegasus with unbiased tie-breaking in the calibration calls (budget-guarded).

Identical to the P4 runs of cem_zephyr_protocol_qpu.py (same seeds, initialization, calibration and
visible feedback), except that the three calibration calls decode chains by majority vote with ties
broken at random instead of Ocean's majority vote (ties -> +1). Training calls are unchanged.

    python scripts/exper/audit_tiefix_qpu.py N SEEDS --cap-s ABS
"""
import argparse, contextlib, io, json
import numpy as np
import cem_zephyr_protocol_qpu as P
from cem_zephyr_protocol_qpu import BudgetedDimodSampler, device_s, parse_seeds, REPO, RESERVE_S

OUT = REPO / "results" / "audit_tiefix_qpu"
_rng = np.random.default_rng(20261008)


def majority_random_ties(samples, chains):
    """Ocean chain-break method: majority vote, ties broken uniformly at random."""
    import dimod
    samples, labels = dimod.as_samples(samples)  # same label handling as Ocean's majority_vote
    if labels != range(len(labels)):
        relabel = {v: i for i, v in enumerate(labels)}
        chains = [[relabel[v] for v in chain] for chain in chains]
    out = np.empty((samples.shape[0], len(chains)), dtype="int8")
    for c, chain in enumerate(chains):
        s = samples[:, list(chain)].sum(axis=1)
        out[:, c] = np.where(s > 0, 1, np.where(s < 0, -1, _rng.choice([-1, 1], len(s))))
    return out, np.arange(samples.shape[0])


_scheduled_sample = P.Scheduled.sample


def _sample(self, rbm, n_samples, config={}, return_hidden=False):
    if self.t < self.calib:
        config = {**config, "chain_break_method": majority_random_ties}
    return _scheduled_sample(self, rbm, n_samples, config, return_hidden)


P.Scheduled.sample = _sample  # only this process

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("N", type=int); p.add_argument("seeds")
    p.add_argument("--cap-s", type=float, required=True); p.add_argument("--n-procs", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    args = argparse.Namespace(calib=3, boot=0, ci=1, alpha=0.5, fb="cem+pl", device="pegasus")
    P.OUT, P.LEDGER = OUT, OUT / "ledger.jsonl"
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    out = OUT / f"P4tiefix_N{a.N}.jsonl"
    device_s()
    sampler = BudgetedDimodSampler("pegasus", a.cap_s)
    for sd in parse_seeds(a.seeds):
        if out.exists() and any(json.loads(l)["seed"] == sd for l in open(out)):
            continue
        if device_s() + a.n_procs * RESERVE_S > a.cap_s:
            print(f"[budget] stopping before seed {sd}", flush=True); break
        with contextlib.redirect_stdout(None if a.verbose else io.StringIO()):
            r = P.run("P4tiefix", a.N, sd, sampler, args)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"N={a.N} seed={sd}: beta_x after calibration={r['beta_x'][3]:.2f} final={r['beta_x'][-1]:.2f} "
              f"qpu={r['qpu_s']:.2f}s time.json={device_s():.1f}s", flush=True)

"""Per-spin chi2 of audit_unit_test.py averaged over random tie-breaking decodings, plus decoding facts.

For every fixed-parameter call at default chain strength: chi2 for 5 independent random decodings,
the fraction of logical spins whose chain was tied (decoded to +1 by Ocean's majority vote), and
the fraction of broken chains.
    python scripts/exper/audit_unit_decodings.py   # writes results/audit_unit_decodings.jsonl
"""
import glob, json
from pathlib import Path
import numpy as np
import audit_unit_test as U
from audit_two_conditional import RES


def main():
    out = RES / "audit_unit_decodings.jsonl"
    with out.open("w") as f:
        for fn in sorted(glob.glob(str(RES / "audit_frozen_qpu/cs_*_cs1.0_bx1.0.npz"))):
            d = np.load(fn); job = json.loads(str(d["job"])); N = job["N"]
            emb = json.loads(str(d["embedding"]))
            idx = {int(q): i for i, q in enumerate(d["phys_vars"])}
            sums = np.stack([d["phys"][:, [idx[q] for q in emb[str(k)]]].sum(1) for k in range(2 * N)], 1)
            lens = np.array([len(emb[str(k)]) for k in range(2 * N)])
            chi2 = []
            for rep in range(5):
                U.rng = np.random.default_rng(100 + rep)
                v, u = U.decode(d["phys"], d["phys_vars"], emb, N)
                chi2.append(U.stats(v, u, d["a"], d["b"], d["W"])[0])
            rec = dict(file=Path(fn).stem, device=job["device"], N=N, seed=job["seed"], df=2 * N - 1,
                       chi2_decodings=chi2, tie_fraction=float(np.mean(sums == 0)),
                       broken_fraction=float(np.mean(np.abs(sums) != lens)),
                       chain_strength=float(d["chain_strength"]), beta_x=float(d["beta_x"]))
            f.write(json.dumps(rec) + "\n")
            print(rec["file"], np.round(np.array(chi2) / rec["df"], 1), f"ties {rec['tie_fraction']:.4f}", flush=True)


if __name__ == "__main__":
    main()

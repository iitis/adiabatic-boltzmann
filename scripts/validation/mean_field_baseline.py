#!/usr/bin/env python3
"""
Mean-field (best product state) baseline for the 1D TFIM benchmark, N = 8..64.

Checks the review's claim (CRITICAL_REVIEW F1 / Appendix A.1) that at h = 0.5 the best product
state is already within ~0.1% of the ground state, and places the stored trained networks
relative to it.

Part A  reference levels per N and h: exact E0 (free fermions; sparse ED cross-check at N <= 16),
        mean field E_MF = -N(1 + h^2/4), all-up -N, x-polarised -hN, and the paper's tolerances
        (eps = 0.1 per spin, 1% relative). E_MF is also re-measured through the training code's
        own local_energy_batch on the product-state RBM (exact i.i.d. samples), which shows the
        ansatz contains the state.
Part B  final error per spin of the stored h = 0.5 runs, two energy estimates:
          global    re-evaluation with global-flip Metropolis (results/audit_eval, audit_evaluate.py);
                    the reference estimate
          E_true    the run file's own evaluation (single-flip Metropolis); biased high at N >= 32,
                    where its chains get trapped in the sector the network suppresses
        The last training-step energy (E[-1]) is not used: it is one 200-sample estimate whose standard
        error (~0.001 per spin) is as large as the mean-field error itself, and it often lies below E0.
        with the median, range, ratio of the median to the mean-field error, and the number of
        runs below the mean-field error.

    python scripts/validation/mean_field_baseline.py
Output: results/mean_field_baseline/summary.json (+ tables on stdout)
"""

import json
import sys
from pathlib import Path

import jax
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from ising import TransverseFieldIsing1D
from mean_field import product_state_rbm_params, reference_levels, sample_product_state
from model import FullyConnectedRBM

NS = (8, 16, 32, 64)
HS = (0.5, 1.0)
H_RUNS = 0.5  # every stored benchmark run is at h = 0.5
N_MC = 200000
OUT = REPO / "results" / "mean_field_baseline" / "summary.json"
R = REPO / "results"

# (label, file pattern, energy estimate): estimate is "E_true" or "global"
RUN_SOURCES = [
    ("Pegasus P4", "cem_pegasus_protocol_qpu/P4cemPL_N{N}.jsonl", "E_true"),
    ("Pegasus P4", "audit_eval/pegasus_P4_N{N}.jsonl", "global"),
    ("Zephyr P4", "cem_zephyr_protocol_qpu/P4cemPL_N{N}.jsonl", "E_true"),
    ("Zephyr P4", "audit_eval/zephyr_P4_N{N}.jsonl", "global"),
    ("classical MH", "audit_eval/classical_mh_N{N}.jsonl", "global"),
    ("classical MH+flip", "audit_eval/classical_mh_flip_N{N}.jsonl", "global"),
]


def sparse_ed(N, h):
    """Ground energy of -sum z_i z_{i+1} - h sum x_i (periodic), built without the free-fermion map."""
    dim = 2**N
    idx = np.arange(dim)
    spins = 1 - 2 * ((idx[:, None] >> np.arange(N)) & 1)  # (dim, N), bit 0 -> +1
    diag = -np.sum(spins * np.roll(spins, -1, axis=1), axis=1).astype(float)
    rows = np.concatenate([idx] + [idx for _ in range(N)])
    cols = np.concatenate([idx] + [idx ^ (1 << i) for i in range(N)])
    vals = np.concatenate([diag] + [np.full(dim, -h) for _ in range(N)])
    H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))
    return float(spla.eigsh(H, k=1, which="SA")[0][0])


def mf_through_training_code(model, rng):
    """<H> of the product-state RBM via TransverseFieldIsing1D.local_energy_batch."""
    N = model.size
    rbm = FullyConnectedRBM(N, N, jax.random.PRNGKey(0))
    rbm.params = product_state_rbm_params(model, N)
    V = sample_product_state(model, N_MC, rng)
    el = np.asarray(model.local_energy_batch(V, rbm))
    return float(el.mean()), float(el.std(ddof=1) / np.sqrt(N_MC))


def part_a():
    rng = np.random.default_rng(0)
    rows = []
    for h in HS:
        for N in NS:
            model = TransverseFieldIsing1D(N, h)
            lv = reference_levels(model)
            e0 = lv["exact"]
            mc, mc_se = mf_through_training_code(model, rng)
            rows.append(
                {
                    "N": N,
                    "h": h,
                    "E0_per_spin": e0 / N,
                    "E0_ed_per_spin": sparse_ed(N, h) / N if N <= 16 else None,
                    "E_mf_per_spin": lv["mean_field"] / N,
                    "E_mf_mc_per_spin": mc / N,
                    "E_mf_mc_se_per_spin": mc_se / N,
                    "err_mf": (lv["mean_field"] - e0) / N,
                    "rel_mf": (lv["mean_field"] - e0) / abs(e0),
                    "err_all_up": (lv["all_up"] - e0) / N,
                    "err_x_pol": (lv["x_polarized"] - e0) / N,
                    "err_eps": 0.1,
                    "err_1pct": 0.01 * abs(e0) / N,
                }
            )
    return rows


def final_energy(rec, estimate):
    if estimate == "E_true":
        return rec.get("E_true")
    return rec["E"]  # audit_eval files store the re-evaluated energy as a scalar


def part_b(mf_err):
    rows = []
    for label, pattern, estimate in RUN_SOURCES:
        for N in NS:
            path = R / pattern.format(N=N)
            if not path.exists():
                continue
            errs = []
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                E = final_energy(rec, estimate)
                if E is not None:
                    errs.append((E - rec["exact"]) / N)
            if not errs:
                continue
            errs = np.array(errs)
            med = float(np.median(errs))
            rows.append(
                {
                    "runs": label,
                    "estimate": estimate,
                    "N": N,
                    "file": str(path.relative_to(REPO)),
                    "n": len(errs),
                    "median_err": med,
                    "min_err": float(errs.min()),
                    "max_err": float(errs.max()),
                    "median_over_mf": med / mf_err[N],
                    "n_below_mf": int(np.sum(errs < mf_err[N])),
                }
            )
    return rows


def main():
    a = part_a()
    print("Part A: reference levels, error per spin relative to the exact ground state")
    print(f"{'h':>4} {'N':>3} {'E0/N':>10} {'ED/N':>10} {'E_MF/N':>9} {'MC check':>17}"
          f" {'MF err':>9} {'MF rel':>7} {'all-up':>7} {'x-pol':>6} {'1% rel':>7}")
    for r in a:
        ed = f"{r['E0_ed_per_spin']:10.6f}" if r["E0_ed_per_spin"] is not None else f"{'-':>10}"
        mc = f"{r['E_mf_mc_per_spin']:.5f}±{r['E_mf_mc_se_per_spin']:.5f}"
        print(f"{r['h']:>4} {r['N']:>3} {r['E0_per_spin']:10.6f} {ed} {r['E_mf_per_spin']:9.6f} {mc:>17}"
              f" {r['err_mf']:9.5f} {100 * r['rel_mf']:6.3f}% {r['err_all_up']:7.4f}"
              f" {r['err_x_pol']:6.4f} {r['err_1pct']:7.4f}")

    mf_err = {r["N"]: r["err_mf"] for r in a if r["h"] == H_RUNS}
    b = part_b(mf_err)
    print(f"\nPart B: final error per spin of stored h = {H_RUNS} runs vs the mean-field error")
    print(f"{'runs':<18} {'estimate':<9} {'N':>3} {'n':>3} {'median':>8} {'range':>17}"
          f" {'MF err':>8} {'med/MF':>7} {'<MF':>6}")
    for r in b:
        rng_s = f"{r['min_err']:.4f}-{r['max_err']:.4f}"
        print(f"{r['runs']:<18} {r['estimate']:<9} {r['N']:>3} {r['n']:>3} {r['median_err']:8.5f}"
              f" {rng_s:>17} {mf_err[r['N']]:8.5f} {r['median_over_mf']:7.2f}"
              f" {r['n_below_mf']:>3}/{r['n']:<2}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"reference_levels": a, "runs_vs_mean_field": b}, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

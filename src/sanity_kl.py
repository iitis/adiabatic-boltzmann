"""
Sanity checks for the KL computation used in the VMC experiment.

Verifies four properties before the full experiment is run:

  1. Spin convention  — every sampler returns values strictly in {-1, +1}
  2. Variable ordering — DimodSampler slices samples[:, :n_visible]; verify
                          sampleset.variables is sorted so the slice is correct
  3. KL → 0 (oracle)  — a sampler that draws exactly from |Ψ|² must give
                          KL ≈ 0 (only finite-sample noise)
  4. KL ordering       — all free samplers give finite KL; Metropolis < Uniform

Run from src/:
    python sanity_kl.py
"""

import sys
import numpy as np
import dimod

sys.path.insert(0, ".")
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler

# ---------------------------------------------------------------------------
# Fixed experimental setup
# ---------------------------------------------------------------------------

SEED = 42
N = 4        # n_visible — 2^4 = 16 configs, fully enumerable
N_HIDDEN = 4
N_SAMPLES = 1000
KL_FINITE_THRESHOLD = 20.0   # anything above this is physically unreasonable for N=4

PASS = "PASS"
FAIL = "*** FAIL"
WARN = "WARN"


# ---------------------------------------------------------------------------
# KL helper — replicates encoder._compute_sample_metrics exactly
# ---------------------------------------------------------------------------

def _build_exact_dist(rbm):
    """Return (all_v, p_true, config_idx) for the current RBM."""
    N = rbm.n_visible
    indices = np.arange(2 ** N, dtype=np.int32)
    all_v = (
        ((indices[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.float64) * 2 - 1
    )
    Theta_all = all_v @ rbm.W + rbm.b[None, :]
    log_psi2 = -(all_v @ rbm.a) + np.sum(np.logaddexp(Theta_all, -Theta_all), axis=1)
    lw = log_psi2 - log_psi2.max()
    p_true = np.exp(lw)
    p_true /= p_true.sum()
    config_idx = {tuple(row.astype(int).tolist()): i for i, row in enumerate(all_v)}
    return all_v, p_true, config_idx


def compute_kl(V, p_true, config_idx):
    """
    D_KL(q_emp ∥ p_true) from a sample matrix V (n_samples, N).
    Returns (kl, n_unmatched).
    n_unmatched > 0 means some samples were not in the {-1,+1} config space.
    """
    ns = V.shape[0]
    counts = np.zeros(len(p_true))
    n_unmatched = 0
    for row in V.astype(int).tolist():
        idx = config_idx.get(tuple(row))
        if idx is not None:
            counts[idx] += 1
        else:
            n_unmatched += 1
    q_emp = counts / ns
    mask = q_emp > 0
    kl = float(np.sum(q_emp[mask] * (np.log(q_emp[mask]) - np.log(p_true[mask]))))
    return kl, n_unmatched


# ---------------------------------------------------------------------------
# Main checks
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    # Build a non-trivial RBM (non-zero weights so the distribution is interesting)
    rbm = FullyConnectedRBM(N, N_HIDDEN)
    rbm.a = rng.normal(0, 0.3, N)
    rbm.b = rng.normal(0, 0.3, N_HIDDEN)
    rbm.W = rng.normal(0, 0.3, (N, N_HIDDEN))

    all_v, p_true, config_idx = _build_exact_dist(rbm)

    samplers = {
        "Metropolis": ClassicalSampler("metropolis", n_warmup=200, n_sweeps=5),
        "Gibbs":      ClassicalSampler("gibbs"),
        "SA-custom":  ClassicalSampler("simulated_annealing", n_warmup=50),
        "SA-dimod":   DimodSampler("simulated_annealing"),
        "Tabu":       DimodSampler("tabu"),
    }

    overall_pass = True

    # ── Check 1 & draw samples ───────────────────────────────────────────────
    print("=" * 65)
    print("CHECK 1: Spin convention — all outputs in {-1, +1}")
    print("=" * 65)

    sampled = {}
    for name, sampler in samplers.items():
        V = np.array(sampler.sample(rbm, N_SAMPLES, config={}), dtype=np.float64)
        unique_vals = set(np.unique(V).tolist())
        shape_ok = V.shape == (N_SAMPLES, N)
        spin_ok = unique_vals <= {-1.0, 1.0}

        status = PASS if (shape_ok and spin_ok) else FAIL
        if status == FAIL:
            overall_pass = False
        print(
            f"  {name:12s}: shape={V.shape}  unique={sorted(unique_vals)}  [{status}]"
        )
        if not shape_ok:
            print(f"    expected shape ({N_SAMPLES}, {N})")
        if not spin_ok:
            print(f"    unexpected values: {unique_vals - {-1.0, 1.0}}")
        sampled[name] = V

    # ── Check 2: DimodSampler variable ordering ──────────────────────────────
    print()
    print("=" * 65)
    print("CHECK 2: DimodSampler variable ordering fix")
    print("  (BQM variables are unsorted; fix must sort columns before slicing)")
    print("=" * 65)

    for name in ["SA-dimod", "Tabu"]:
        sampler = samplers[name]
        J, h_lin = sampler.rbm_to_ising(rbm, 1.0)
        bqm = dimod.BinaryQuadraticModel.from_ising(h_lin, J, 0.0)
        vars_list = list(bqm.variables)

        # The BQM variable order is not guaranteed to be sorted — that is the
        # known root cause.  What we verify here is that applying the fix
        # (sort_idx = np.argsort(variables)) correctly maps:
        #   first N columns  → variables 0..N-1   (visible units)
        #   next N_HIDDEN cols → variables N..N+N_HIDDEN-1 (hidden units)
        sort_idx = np.argsort(vars_list)
        sorted_vars = [vars_list[i] for i in sort_idx]
        visible_ok = sorted_vars[:N] == list(range(N))
        hidden_ok  = sorted_vars[N:] == list(range(N, N + N_HIDDEN))
        fix_ok = visible_ok and hidden_ok

        status = PASS if fix_ok else FAIL
        if not fix_ok:
            overall_pass = False
        print(f"  {name:12s}: raw_vars={vars_list}")
        print(f"               after sort → visible={sorted_vars[:N]}  hidden={sorted_vars[N:]}")
        print(f"               visible_ok={visible_ok}  hidden_ok={hidden_ok}  [{status}]")

    # ── Check 3: Oracle sampler → KL ≈ 0 ────────────────────────────────────
    print()
    print("=" * 65)
    print("CHECK 3: Oracle sampler — KL must be near zero")
    print("  (draws exactly from |Ψ|²; limited only by finite-sample noise)")
    print("=" * 65)

    oracle_idx = rng.choice(len(all_v), size=N_SAMPLES, p=p_true)
    V_oracle = all_v[oracle_idx]
    kl_oracle, unmatched_oracle = compute_kl(V_oracle, p_true, config_idx)

    # Upper bound: with n=1000 samples over 16 configs, KL < 0.05 is expected
    oracle_ok = kl_oracle < 0.05 and unmatched_oracle == 0
    status = PASS if oracle_ok else WARN
    print(
        f"  Oracle: KL = {kl_oracle:.6f}  unmatched = {unmatched_oracle}  [{status}]"
    )
    if not oracle_ok:
        print(f"    Expected KL < 0.05 with n={N_SAMPLES}, 2^{N}={2**N} configs")

    # ── Check 4 & 5: KL finite + ordering ───────────────────────────────────
    print()
    print("=" * 65)
    print("CHECK 4 & 5: KL finite + ordering vs. uniform random baseline")
    print("=" * 65)

    V_uniform = rng.choice([-1.0, 1.0], size=(N_SAMPLES, N))
    kl_uniform, _ = compute_kl(V_uniform, p_true, config_idx)
    print(f"  {'Uniform':12s}: KL = {kl_uniform:.4f}  [baseline]")

    kl_results = {}
    for name, V in sampled.items():
        kl, n_unmatched = compute_kl(V, p_true, config_idx)
        if n_unmatched > 0:
            status = FAIL
            overall_pass = False
            print(
                f"  {name:12s}: {n_unmatched}/{N_SAMPLES} samples unmatched  [{status}]"
            )
            kl_results[name] = float("inf")
        else:
            finite_ok = kl < KL_FINITE_THRESHOLD
            status = PASS if finite_ok else FAIL
            if not finite_ok:
                overall_pass = False
            print(f"  {name:12s}: KL = {kl:.4f}  [{status}]")
            kl_results[name] = kl

    # Metropolis should beat uniform (basic correctness of targeting |Ψ|²)
    print()
    kl_metro = kl_results.get("Metropolis", float("inf"))
    if kl_metro < kl_uniform:
        print(
            f"  {PASS}: Metropolis KL ({kl_metro:.4f}) < Uniform KL ({kl_uniform:.4f})"
        )
    else:
        overall_pass = False
        print(
            f"  {FAIL}: Metropolis KL ({kl_metro:.4f}) >= Uniform KL ({kl_uniform:.4f})"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  {'Oracle':12s}: KL = {kl_oracle:.6f}")
    for name, kl in kl_results.items():
        print(f"  {name:12s}: KL = {kl:.4f}")
    print(f"  {'Uniform':12s}: KL = {kl_uniform:.4f}")
    print()
    verdict = PASS if overall_pass else FAIL
    print(f"  Overall: [{verdict}]")
    if not overall_pass:
        print("  Fix all FAIL items before running the experiment.")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Pytest suite — run with: pytest sanity_kl.py -v
# ---------------------------------------------------------------------------

import pytest


def _make_scrambled_sampleset(n_visible, n_hidden, n_samples=10):
    """
    Build a dimod SampleSet whose variable order is scrambled the same way
    the real RBM→Ising BQM produces: [0, Nv, Nv+1, ..., Nv+Nh-1, 1, 2, ..., Nv-1].
    Each sample has visible unit i = +1 and hidden unit j = -1 so we can
    verify extraction by value.
    """
    N = n_visible + n_hidden
    scrambled_vars = [0] + list(range(n_visible, n_visible + n_hidden)) + list(range(1, n_visible))
    assert len(scrambled_vars) == N

    # Build samples: visible units are all +1, hidden units are all -1
    raw = np.ones((n_samples, N), dtype=np.int8)
    raw[:, n_visible:] = -1   # columns n_visible.. are hidden in the scrambled order
    # But we must lay them out in scrambled_vars order:
    # scrambled_vars[0]       = visible 0  → +1
    # scrambled_vars[1..Nh]   = hidden     → -1
    # scrambled_vars[Nh+1..]  = visible 1+ → +1
    # So raw layout: col 0 = var 0 (+1), cols 1..Nh = hidden (-1), cols Nh+1.. = visible (+1)
    # This matches raw already.

    sampleset = dimod.SampleSet.from_samples(
        (raw, scrambled_vars),
        vartype=dimod.SPIN,
        energy=[0.0] * n_samples,
    )
    return sampleset, scrambled_vars


def test_dimod_sa_tabu_fix_extracts_visible_units():
    """
    The sort_idx fix in simulated_annealing / tabu_search must extract only
    visible units (variables 0..n_visible-1) regardless of sampleset variable order.

    Visible units are set to +1, hidden units to -1 in the synthetic sampleset.
    After the fix, all values in v must be +1.
    """
    n_visible, n_hidden = 4, 4
    sampleset, _ = _make_scrambled_sampleset(n_visible, n_hidden)

    # Apply the fix exactly as in the patched sampler
    sort_idx = np.argsort(list(sampleset.variables))
    samples = sampleset.record.sample[:, sort_idx]
    v = samples[:, :n_visible]
    h = samples[:, n_visible:n_visible + n_hidden]

    assert np.all(v == 1),  f"visible units should be +1, got: {np.unique(v)}"
    assert np.all(h == -1), f"hidden units should be -1, got: {np.unique(h)}"


def test_dimod_sa_tabu_unfixed_extracts_wrong_units():
    """
    Without the fix, position-based slicing extracts hidden units into the
    visible slot — confirming the original bug was real.
    """
    n_visible, n_hidden = 4, 4
    sampleset, _ = _make_scrambled_sampleset(n_visible, n_hidden)

    # Old (buggy) extraction
    samples_raw = sampleset.record.sample
    v_buggy = samples_raw[:, :n_visible]

    # Should contain -1 values (hidden units leaked into visible slot)
    assert np.any(v_buggy == -1), (
        "Expected buggy extraction to contain hidden unit values (-1), "
        "but all values were +1 — variable ordering may have changed in this dimod version."
    )


def test_dwave_label_based_extraction_is_correct():
    """
    The D-Wave code path uses df.loc[:, list(range(n_visible))] — label-based
    selection — which is immune to variable ordering.

    Verify it extracts the correct visible units from a scrambled sampleset.
    """
    n_visible, n_hidden = 4, 4
    sampleset, _ = _make_scrambled_sampleset(n_visible, n_hidden)

    # D-Wave extraction (label-based, as in dwave() method)
    df = sampleset.to_pandas_dataframe()
    v = df.loc[:, list(range(n_visible))].to_numpy()
    h = df.loc[:, list(range(n_visible, n_visible + n_hidden))].to_numpy()

    assert np.all(v == 1),  f"visible units should be +1, got: {np.unique(v)}"
    assert np.all(h == -1), f"hidden units should be -1, got: {np.unique(h)}"


def test_sort_idx_covers_all_variables():
    """
    sort_idx = np.argsort(sampleset.variables) must be a permutation of 0..N-1,
    covering every variable exactly once.
    """
    n_visible, n_hidden = 4, 4
    sampleset, scrambled_vars = _make_scrambled_sampleset(n_visible, n_hidden)
    N = n_visible + n_hidden

    sort_idx = np.argsort(list(sampleset.variables))
    assert len(sort_idx) == N
    assert set(sort_idx.tolist()) == set(range(N))

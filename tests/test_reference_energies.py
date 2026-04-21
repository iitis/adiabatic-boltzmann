"""
Correctness tests for reference energy computations.

Two independent algorithms are cross-checked for each geometry:

  1D  — free-fermion discrete sum  vs  NetKet sparse eigsh
  2D  — NetKet sparse eigsh        vs  direct scipy.sparse construction
        (built without NetKet, using the same bond convention as the JAX kernel)

Run from the repo root:
    python -m pytest tests/test_reference_energies.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from ising import TransverseFieldIsing1D, TransverseFieldIsing2D

# ── tolerance ─────────────────────────────────────────────────────────────────
# Both sides are numerically exact; 1e-5 leaves room for eigsh convergence.
ATOL = 1e-5

# ── test parameters ───────────────────────────────────────────────────────────
PARAMS_1D = [(N, h) for N in [4, 6, 8] for h in [0.5, 1.0, 1.5, 2.0]]
# L=2,4 appear in actual results; L=3 is included for coverage of odd-L case.
PARAMS_2D = [(L, h) for L in [2, 3, 4] for h in [0.5, 1.0, 1.5, 2.0]]


# ── independent reference implementations ─────────────────────────────────────


def _netket_ground_energy_1d(N: int, h: float) -> float:
    """NetKet sparse eigsh for 1D TFIM — independent of the free-fermion formula."""
    nk = pytest.importorskip("netket")
    from scipy.sparse.linalg import eigsh

    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    ha = nk.operator.LocalOperator(hilbert)
    for i in range(N):
        ha += -1.0 * nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(
            hilbert, (i + 1) % N
        )
        ha += -h * nk.operator.spin.sigmax(hilbert, i)
    vals, _ = eigsh(ha.to_sparse(), k=1, which="SA")
    return float(vals[0])


def _direct_ground_energy_2d(L: int, h: float) -> float:
    """
    Direct scipy.sparse construction for 2D TFIM — independent of NetKet.

    Uses the same bond convention as _local_energy_2d_jit:
    for each site i, count bond to right neighbor and bond to down neighbor.
    For L≥3 each bond is counted once; for L=2 each bond is counted twice
    (same double-counting present in _exact_diag_2d and the JAX kernel).
    """
    N = L * L
    dim = 1 << N

    i_idx = np.arange(N)
    cols = i_idx % L
    rows = i_idx // L
    right = rows * L + (cols + 1) % L   # shape (N,)
    down = ((rows + 1) % L) * L + cols  # shape (N,)

    # Diagonal: -∑_i [ σz_i · σz_{right[i]}  +  σz_i · σz_{down[i]} ]
    all_states = np.arange(dim, dtype=np.int64)
    # spins[s, i] = +1 if bit i of s is 0, else -1
    spins = 1 - 2 * ((all_states[:, None] >> i_idx[None, :]) & 1)  # (dim, N)
    diag = -np.sum(
        spins * spins[:, right] + spins * spins[:, down], axis=1
    ).astype(float)  # (dim,)

    # Off-diagonal: -h ∑_i σx_i  ↔  flip bit i
    flip_masks = (1 << i_idx).astype(np.int64)                         # (N,)
    row_idx = np.repeat(all_states, N)                                  # (dim*N,)
    col_idx = (all_states[:, None] ^ flip_masks[None, :]).ravel()       # (dim*N,)
    off_data = np.full(dim * N, -h)

    all_rows = np.concatenate([all_states, row_idx])
    all_cols = np.concatenate([all_states, col_idx])
    all_data = np.concatenate([diag, off_data])

    H = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(dim, dim))

    if dim <= 512:
        return float(np.linalg.eigvalsh(H.toarray())[0])
    vals, _ = spla.eigsh(H, k=1, which="SA")
    return float(vals[0])


# ── 1D tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("N,h", PARAMS_1D)
def test_1d_free_fermion_vs_netket(N, h):
    """Free-fermion discrete sum must match NetKet exact diag to ATOL."""
    pytest.importorskip("netket")
    model = TransverseFieldIsing1D(N, h)
    our = model._compute_exact_ground_energy()
    ref = _netket_ground_energy_1d(N, h)
    assert abs(our - ref) < ATOL, (
        f"1D N={N} h={h}: free-fermion={our:.8f}, netket={ref:.8f}, "
        f"diff={abs(our - ref):.2e}"
    )


@pytest.mark.parametrize("N,h", PARAMS_1D)
def test_1d_cache_matches_computation(N, h):
    """Cached reference value must equal direct recomputation."""
    model = TransverseFieldIsing1D(N, h)
    cached = model.exact_ground_energy()
    direct = model._compute_exact_ground_energy()
    assert abs(cached - direct) < ATOL, (
        f"1D N={N} h={h}: cached={cached:.8f}, direct={direct:.8f}"
    )


# ── 2D tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("L,h", PARAMS_2D)
def test_2d_netket_vs_direct_scipy(L, h):
    """NetKet exact diag must match direct scipy.sparse construction to ATOL."""
    pytest.importorskip("netket")
    model = TransverseFieldIsing2D(L, h)
    our = model._exact_diag_2d()
    ref = _direct_ground_energy_2d(L, h)
    assert abs(our - ref) < ATOL, (
        f"2D L={L} h={h}: netket={our:.8f}, direct={ref:.8f}, "
        f"diff={abs(our - ref):.2e}"
    )


@pytest.mark.parametrize("L,h", PARAMS_2D)
def test_2d_cache_matches_computation(L, h):
    """Cached reference value must equal direct recomputation."""
    model = TransverseFieldIsing2D(L, h)
    cached = model.exact_ground_energy()
    direct = model._exact_diag_2d()
    assert abs(cached - direct) < ATOL, (
        f"2D L={L} h={h}: cached={cached:.8f}, direct={direct:.8f}"
    )


# ── NotImplementedError for large 2D ─────────────────────────────────────────


@pytest.mark.parametrize("L", [5, 6])
def test_2d_large_raises(L):
    """exact_ground_energy must raise NotImplementedError for L>4."""
    with pytest.raises(NotImplementedError):
        TransverseFieldIsing2D(L, 1.0).exact_ground_energy()

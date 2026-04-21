"""
Pre-compute and cache exact ground state reference energies.

Run from the repo root:
    python scripts/fill_reference_energies.py

Covers every (model, size, h) combination seen in existing result files plus
h=1.5 for all relevant small sizes.  2D L>4 is skipped (no exact method
available within 0.001 accuracy).
"""

import sys
from itertools import product
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from ising import TransverseFieldIsing1D, TransverseFieldIsing2D

# All 1D sizes that appear in results
SIZES_1D = [4, 6, 8, 12, 14, 16, 24, 28, 32, 42, 64, 96, 128, 512, 768, 1024, 2048]

# 2D linear dimensions: only L≤4 can be exactly diagonalised (2^16 states)
SIZES_2D = [2, 3, 4]

H_VALUES = [0.5, 1.0, 1.5, 2.0]


def _fill_1d(h_values, sizes):
    ok = err = 0
    print("=== 1D chains (free-fermion exact) ===")
    for N, h in product(sizes, h_values):
        try:
            e = TransverseFieldIsing1D(N, h).exact_ground_energy()
            print(f"  N={N:5d}  h={h:.1f}  E={e:.8f}")
            ok += 1
        except Exception as exc:
            print(f"  N={N:5d}  h={h:.1f}  ERROR: {exc}", file=sys.stderr)
            err += 1
    return ok, err


def _fill_2d(h_values, sizes):
    ok = skipped = err = 0
    print("\n=== 2D lattices (NetKet exact diag, L≤4 only) ===")
    for L, h in product(sizes, h_values):
        try:
            e = TransverseFieldIsing2D(L, h).exact_ground_energy()
            print(f"  L={L}  h={h:.1f}  E={e:.8f}")
            ok += 1
        except NotImplementedError:
            print(f"  L={L}  h={h:.1f}  SKIPPED (L>4, no exact method)")
            skipped += 1
        except Exception as exc:
            print(f"  L={L}  h={h:.1f}  ERROR: {exc}", file=sys.stderr)
            err += 1
    return ok, skipped, err


def main() -> int:
    ok1, err1 = _fill_1d(H_VALUES, SIZES_1D)
    ok2, skipped2, err2 = _fill_2d(H_VALUES, SIZES_2D)

    total_ok = ok1 + ok2
    total_err = err1 + err2
    print(f"\nDone: {total_ok} values cached, {skipped2} skipped, {total_err} errors.")
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())

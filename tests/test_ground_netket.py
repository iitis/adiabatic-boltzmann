import numpy as np
import pytest
import netket as nk
from src.ising import TransverseFieldIsing1D


def netket_ground_energy(N, h):
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    ha = nk.operator.LocalOperator(hilbert)
    for i in range(N):
        ha += (
            -1.0
            * nk.operator.spin.sigmaz(hilbert, i)
            @ nk.operator.spin.sigmaz(hilbert, (i + 1) % N)
        )
        ha += -h * nk.operator.spin.sigmax(hilbert, i)
    H_sparse = ha.to_sparse()
    from scipy.sparse.linalg import eigsh

    vals, _ = eigsh(H_sparse, k=1, which="SA")
    return vals[0]


print(netket_ground_energy(16, 0.5))


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12, 14, 16])
@pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 1.5, 2.0])
def test_ground_state_energy(N, h):
    E_nk = netket_ground_energy(N, h)
    model = TransverseFieldIsing1D(N, h)
    E_ours = model.exact_ground_energy()
    assert np.allclose(E_nk, E_ours, atol=1e-8), (
        f"Mismatch for N={N}, h={h}: NetKet={E_nk}, Ours={E_ours}"
    )


print()

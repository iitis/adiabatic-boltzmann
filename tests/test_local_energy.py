import numpy as np
import pytest
import netket as nk

from src.ising import TransverseFieldIsing1D
from src.model import FullyConnectedRBM

import jax.numpy as jnp

def compute_local_energies(model, parameters, hamiltonian_jax, sigma):
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

    logpsi_sigma = model.apply(parameters, sigma)
    logpsi_eta = model.apply(parameters, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)

    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

    return res

def netket_local_energy(N, h, v, rbm):
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    ha = nk.operator.LocalOperator(hilbert)
    for i in range(N):
        ha += -1.0 * nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(hilbert, (i+1)%N)
        ha += -h * nk.operator.spin.sigmax(hilbert, i)
    # NetKet expects spins in {-1, +1}
    v = np.asarray(v)
    # Use RBM parameters from our model
    W = rbm.W
    a = rbm.a
    b = rbm.b
    # NetKet RBM for reference
    nk_rbm = nk.models.RBM(alpha=W.shape[1]//N, use_visible_bias=True, use_hidden_bias=True)
    params = nk_rbm.init(nk.jax.PRNGKey(0), (1, N))
    params["params"]["a"] = np.array(a)
    params["params"]["b"] = np.array(b)
    params["params"]["W"] = np.array(W)
    # Local energy
    v = v.reshape(1, -1)
    le = ha.local_energy(nk_rbm, params, v)
    return float(np.real(le[0]))

@pytest.mark.parametrize("N,h", [
    (4, 0.1), (4, 1.0), (4, 2.0),
    (8, 0.5), (8, 1.5),
    (12, 1.0),
    (16, 0.5),
])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_matches_netket(N, h, seed):
    rng = np.random.default_rng(seed)
    ising = TransverseFieldIsing1D(N, h)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    # Test 3 random spin configurations per (N, h, seed)
    for _ in range(3):
        v = rng.choice([-1, 1], size=N)
        E_ours = ising.local_energy(v, rbm.psi_ratio)
        E_ref = netket_local_energy(N, h, v, rbm)
        assert np.allclose(E_ours, E_ref, atol=1e-6), f"Mismatch for N={N}, h={h}, v={v}: ours={E_ours}, netket={E_ref}"
v

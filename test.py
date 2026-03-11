
import sys
import netket as nk 
import matplotlib.pyplot as plt
import json
import time
import numpy as np
from netket.operator.spin import sigmax, sigmaz
import jax
import jax.numpy as jnp
N = 16 


def compute_local_energies(model, parameters, hamiltonian_jax, sigma):
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

    logpsi_sigma = model.apply(parameters, sigma)
    logpsi_eta = model.apply(parameters, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)

    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

    return res

g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
ha = nk.operator.IsingJax(hi, g, h=0.5, J=1)


# RBM ansatz with alpha=1
ma = nk.models.RBM(alpha=1)

# Optimizer
ha_jax = ha.to_jax_operator()
parameters = ma.init(jax.random.key(0), np.ones((hi.size,)))
samples =np.ones((1,N )) 
E = compute_local_energies(ma, parameters, ha_jax, samples )
print(E)
print(len(E))
print(samples[0][0])
print(samples.dtype)


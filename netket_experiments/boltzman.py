import netket as nk
import time
import json
from matplotlib import pyplot as plt

L = 16
calculate_exact_ground = True

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
ha = nk.operator.IsingJax(hi, g, h=0.1, J=1)

if calculate_exact_ground:
    evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
    exact_gs_energy = evals[0]
    print("The exact ground-state energy is E0=", exact_gs_energy)
# Build the sampler
sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)
ma = nk.models.RBM(alpha=1)
# Build the sampler
sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

# The variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008)

# The ground-state optimization loop
gs = nk.driver.VMC_SR(
    hamiltonian=ha, optimizer=op, diag_shift=0.1, variational_state=vs
)

start = time.time()
gs.run(out="RBM", n_iter=600)
end = time.time()

print("### RBM calculation")
print("Has", vs.n_parameters, "parameters")
print("The RBM calculation took", end - start, "seconds")

# import the data from log file
data = json.load(open("RBM.log"))

# Extract the relevant information
iters_RBM = data["Energy"]["iters"]
energy_RBM = data["Energy"]["Mean"]

fig, ax1 = plt.subplots()
ax1.plot(iters_RBM, energy_RBM, color="red", label="Energy (RBM)")
ax1.set_ylabel("Energy")
ax1.set_xlabel("Iteration")
plt.axhline(
    y=exact_gs_energy, xmin=0, xmax=iters_RBM[-1], linewidth=2, color="k", label="Exact"
)
ax1.legend()
plt.show()

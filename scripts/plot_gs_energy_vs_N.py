import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipe

H_VALUES = [0.5, 1.0, 2.0]
COLORS = ["tab:blue", "tab:orange", "tab:green"]
N_VALUES = np.arange(2, 201)


def gs_energy_per_spin(N, h):
    m = np.arange(N)
    k_R = np.pi * (2 * m + 1) / N
    k_NS = 2.0 * np.pi * m / N
    E_R = -np.sum(np.sqrt(1 + h**2 - 2 * h * np.cos(k_R)))
    E_NS = -np.sum(np.sqrt(1 + h**2 - 2 * h * np.cos(k_NS)))
    return min(E_R, E_NS) / N


def thermodynamic_limit(h):
    # E0/N = -(2/pi)(1+h) E(2*sqrt(h)/(1+h))
    k = 2 * np.sqrt(h) / (1 + h)
    return -(2 / np.pi) * (1 + h) * ellipe(k**2)


fig, ax = plt.subplots(figsize=(8, 5))

for h, color in zip(H_VALUES, COLORS):
    energies = [gs_energy_per_spin(N, h) for N in N_VALUES]
    e_inf = thermodynamic_limit(h)
    label = f"h = {h}" + (" (critical)" if h == 1.0 else "")
    ax.plot(N_VALUES, energies, color=color, lw=1.5, label=label)
    ax.axhline(e_inf, color=color, lw=0.8, ls="--", alpha=0.6)

ax.set_xlabel("N (number of spins)")
ax.set_ylabel("Ground state energy per spin  $E_0 / N$")
ax.set_title("1D TFIM exact ground state energy vs system size")
ax.legend()

# Annotate thermodynamic limits on the right edge
for h, color in zip(H_VALUES, COLORS):
    e_inf = thermodynamic_limit(h)
    ax.annotate(
        f"  $N\\to\\infty$: {e_inf:.4f}",
        xy=(200, e_inf),
        fontsize=8,
        color=color,
        va="center",
    )

plt.tight_layout()
out = "gs_energy_vs_N.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")

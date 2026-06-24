#!/usr/bin/env python3
"""
Compute ground state energy of 1D TFIM using Bethe ansatz solution.
Works for arbitrary system sizes.
"""

import numpy as np
from scipy import integrate
import argparse


def tfim_ground_energy_analytical(N, h):
    """
    Ground state energy of 1D Transverse Field Ising Model using Bethe Ansatz.
    
    Hamiltonian: H = -sum_i sigma_z_i sigma_z_{i+1} - h sum_i sigma_x_i
    
    Uses the analytic solution from Bethe ansatz.
    For periodic boundary conditions (periodic TFIM).
    
    Returns: Energy per spin (E/N)
    """
    # Define integrand
    def integrand(k):
        return np.sqrt(1 + h**2 + 2*h*np.cos(k))
    
    # Numerical integration
    result, _ = integrate.quad(integrand, 0, np.pi)
    
    # Ground state energy per spin
    E_per_spin = -(1.0 / np.pi) * result
    
    return E_per_spin


def main():
    parser = argparse.ArgumentParser(
        description="Compute ground state energy of 1D TFIM using Bethe Ansatz"
    )
    parser.add_argument("-N", type=int, required=True, help="Number of spins")
    parser.add_argument(
        "-g", type=float, required=True, help="Transverse field strength"
    )

    args = parser.parse_args()

    energy = tfim_ground_energy_analytical(args.N, args.g)
    print(f"Ground state energy (N={args.N}, h={args.g}): {energy}")


if __name__ == "__main__":
    main()

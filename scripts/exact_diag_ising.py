import numpy as np
import netket as nk
import argparse


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


def main():
    parser = argparse.ArgumentParser(
        description="Compute ground state energy of 1D TFIM using NetKet"
    )
    parser.add_argument("-N", type=int, required=True, help="Number of spins")
    parser.add_argument(
        "-g", type=float, required=True, help="Transverse field strength"
    )

    args = parser.parse_args()

    energy = netket_ground_energy(args.N, args.g)
    print(f"Ground state energy (N={args.N}, h={args.g}): {energy}")


if __name__ == "__main__":
    main()

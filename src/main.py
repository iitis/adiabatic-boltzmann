import argparse
import numpy as np

from helpers import save_results
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D


def parse_arguments():
    """
    Example usage:
    python homework/main_skeleton.py --model 1d --size 8 --h 0.5 --rbm full --sampler classical

    Returns: argparse.Namespace with all arguments
    """
    parser = argparse.ArgumentParser(
        description="Train RBM to learn Ising model ground states"
    )

    # Model parameters
    parser.add_argument(
        "--model", choices=["1d", "2d"], default="1d", help="Ising model type"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="System size (chain length or square lattice dimension)",
    )
    parser.add_argument(
        "--h", type=float, default=0.5, help="Transverse field strength"
    )

    # RBM architecture
    parser.add_argument(
        "--rbm",
        choices=["full", "dwave"],
        default="full",
        help="RBM connectivity pattern",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=None,
        help="Number of hidden units (default: equal to visible)",
    )

    # Sampling
    parser.add_argument(
        "--sampler",
        choices=["custom", "dimod", "velox"],
        default="velox",
        help="Sampling backend",
    )
    parser.add_argument(
        "--sampling-method",
        choices=["metropolis", "simulated_annealing", "tabu"],
        default="simulated_annealing",
        help="Classical sampling algorithm",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Samples per iteration"
    )

    # Training
    parser.add_argument(
        "--iterations", type=int, default=30, help="Training iterations"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Gradient step size"
    )
    parser.add_argument(
        "--regularization", type=float, default=1e-5, help="SR matrix regularization"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results/", help="Directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--visualize", action="store_true", help="Plot convergence curves", default=True
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set seed for reproducibility
    np.random.seed(args.seed)

    print(f"Configuration:")
    print(f"  Model: {args.model} with h={args.h}")
    print(f"  System size: {args.size}")
    print(f"  RBM: {args.rbm}")
    print(f"  Sampler: {args.sampler} ({args.sampling_method})")
    print(f"  Training: {args.iterations} iterations, lr={args.learning_rate}")

    # TODO:
    # 1. Instantiate Ising model
    if args.model == "1d":
        ising = TransverseFieldIsing1D(args.size, args.h)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size, args.h)

    # 2. Instantiate RBM
    n_hidden = args.n_hidden or args.size
    if args.rbm == "full":
        rbm = FullyConnectedRBM(args.size, n_hidden)
    else:
        rbm = DWaveTopologyRBM(args.size, n_hidden)

    # 3. Instantiate sampler
    if args.sampler == "custom":
        sampler = ClassicalSampler(method=args.sampling_method)
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.sampling_method)
    elif args.sampler == "velox":
        sampler = VeloxSampler(method=args.sampling_method)
    # 4. Build trainer config
    trainer_config = {
        "learning_rate": args.learning_rate,
        "n_iterations": args.iterations,
        "n_samples": args.n_samples,
        "regularization": args.regularization,
    }

    # 5. Create trainer and run
    trainer = Trainer(rbm, ising, sampler, trainer_config)

    print(f"\nStarting training...")
    history = trainer.train()
    save_results(args, history, ising)


if __name__ == "__main__":
    main()

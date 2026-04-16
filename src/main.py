import jax
jax.config.update("jax_enable_x64", True)

import argparse

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
        choices=["full", "pegasus", "zephyr"],
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
        default="dimod",
        help="Sampling backend",
    )
    parser.add_argument(
        "--sampling-method",
        choices=[
            "pegasus",
            "zephyr",
            "metropolis",
            "velox",
            "simulated_annealing",
            "tabu",
            "gibbs",
            "lsb",
        ],
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

    # CEM beta scheduling
    parser.add_argument(
        "--cem",
        action="store_true",
        default=False,
        help="Enable CEM-based β scheduling (estimates β_eff every --cem-interval iterations)",
    )
    parser.add_argument(
        "--cem-interval",
        type=int,
        default=5,
        help="Iterations between β_eff estimates when --cem is active",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help='LSB noise precision σ⁻² (paper convention). σ = 1/√(σ⁻²). Default 1.0 → σ=1.0. Only used if --sampling-method is "lsb".',
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

    # PRNG key — single source of randomness for RBM init, sampler, and trainer
    key = jax.random.PRNGKey(args.seed)

    print(f"Configuration:")
    print(f"  Model: {args.model} with h={args.h}")
    print(f"  System size: {args.size}")
    print(f"  RBM: {args.rbm}")
    print(f"  Sampler: {args.sampler} ({args.sampling_method})")
    print(f"  Training: {args.iterations} iterations, lr={args.learning_rate}")
    print(
        f"  CEM β scheduling: {'ON (interval=' + str(args.cem_interval) + ')' if args.cem else 'OFF'}"
    )
    print(f"  JAX devices: {jax.devices()}")

    # 1. Instantiate Ising model
    if args.model == "1d":
        ising = TransverseFieldIsing1D(args.size, args.h)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size, args.h)

    # 2. Instantiate RBM
    if args.n_hidden is not None:
        n_hidden = args.n_hidden
    elif args.model == "1d":
        n_hidden = args.size
    elif args.model == "2d":
        n_hidden = args.size**2
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    n_visible = args.size if args.model == "1d" else args.size**2
    args.n_hidden = n_hidden
    key, rbm_key = jax.random.split(key)
    if args.rbm == "full":
        rbm = FullyConnectedRBM(n_visible, n_hidden, rbm_key)
    else:
        rbm = DWaveTopologyRBM(n_visible, n_hidden, rbm_key, solver=args.rbm)

    # 3. Instantiate sampler
    if args.sampler == "custom":
        sampler = ClassicalSampler(
            method=args.sampling_method,
            n_sweeps=getattr(args, "gibbs_sweeps", 10)
            if args.sampling_method == "gibbs"
            else 1,
        )
        key, sampler_key = jax.random.split(key)
        sampler._key = sampler_key
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.sampling_method)
    elif args.sampler == "velox":
        sampler = VeloxSampler(method=args.sampling_method)

    # 4. Build trainer config
    _is_dwave = args.sampling_method in ("pegasus", "zephyr")
    trainer_config = {
        "learning_rate": args.learning_rate,
        "n_iterations": args.iterations,
        "n_samples": args.n_samples,
        "regularization": args.regularization,
        "save_checkpoints": _is_dwave,
        "checkpoint_interval": 10,
        "use_cem": args.cem,
        "cem_interval": args.cem_interval,
        "lsb_sigma": args.sigma,
        "seed": args.seed,
    }

    # 5. Create trainer and run
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args)

    print(f"\nStarting training...")
    history = trainer.train()
    save_results(args, history, ising, rbm)
    if args.rbm != "full":
        print(f"sparsity: {rbm.connectivity_summary()['sparsity']}")


if __name__ == "__main__":
    main()

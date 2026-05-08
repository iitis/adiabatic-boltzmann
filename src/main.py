import jax
jax.config.update("jax_enable_x64", True)

import argparse

from helpers import save_results
from model import FullyConnectedRBM, DWaveTopologyRBM
from model_vit import ViTWaveFunction
from sampler import ClassicalSampler, DimodSampler, VeloxSampler, GenericClassicalSampler
from encoder import Trainer
from encoder_generic import TrainerGeneric
from ising import (
    TransverseFieldIsing1D,
    TransverseFieldIsing2D,
    HeisenbergXXZ1D,
    LongRangeTFIM1D,
    J1J2Ising1D,
    HeisenbergXY1D,
    HeisenbergXXZ2D,
)


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
        "--model",
        choices=["1d", "2d", "heisenberg_xxz_1d", "lr1d", "j1j2_1d", "heisenberg_xy_1d", "heisenberg_xxz_2d"],
        default="1d",
        help="Physical model",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="System size (chain length or square lattice dimension)",
    )
    parser.add_argument(
        "--h", type=float, default=0.5, help="Transverse field strength (TFIM only)"
    )
    parser.add_argument(
        "--J", type=float, default=1.0, help="Coupling strength (Heisenberg/XY)"
    )
    parser.add_argument(
        "--J1", type=float, default=1.0, help="NN coupling J₁ (J1J2 model)"
    )
    parser.add_argument(
        "--J2", type=float, default=0.5, help="NNN coupling J₂ (J1J2 model)"
    )
    parser.add_argument(
        "--delta", type=float, default=1.0, help="XXZ anisotropy Δ (Heisenberg)"
    )
    parser.add_argument(
        "--alpha", type=float, default=2.0, help="Power-law exponent α (LR-TFIM)"
    )

    # Ansatz selection
    parser.add_argument(
        "--ansatz",
        choices=["rbm", "vit"],
        default="rbm",
        help="Wave function ansatz: RBM or Vision Transformer",
    )

    # RBM architecture (used when --ansatz rbm)
    parser.add_argument(
        "--rbm",
        choices=["full", "pegasus", "zephyr"],
        default="full",
        help="RBM connectivity pattern (--ansatz rbm only)",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=None,
        help="Number of RBM hidden units (default: equal to visible)",
    )

    # ViT architecture (used when --ansatz vit)
    parser.add_argument(
        "--d-model", type=int, default=32,
        help="ViT embedding dimension (--ansatz vit only)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=2,
        help="ViT number of transformer encoder blocks",
    )
    parser.add_argument(
        "--n-heads", type=int, default=4,
        help="ViT number of attention heads (must divide d-model)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=2,
        help="Spins per patch for 1D, or patch side length for 2D",
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
            "exchange",
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

    if args.model == "heisenberg_xxz_1d":
        _model_desc = f"{args.model} with J={args.J}, Δ={args.delta}"
    elif args.model == "lr1d":
        _model_desc = f"{args.model} with h={args.h}, α={args.alpha}"
    elif args.model == "j1j2_1d":
        _model_desc = f"{args.model} with J1={args.J1}, J2={args.J2}, h={args.h}"
    elif args.model == "heisenberg_xy_1d":
        _model_desc = f"{args.model} with J={args.J}"
    elif args.model == "heisenberg_xxz_2d":
        _model_desc = f"{args.model} with J={args.J}, Δ={args.delta}"
    else:
        _model_desc = f"{args.model} with h={args.h}"

    if args.ansatz == "vit" and args.sampler in ("dimod", "velox"):
        raise ValueError(
            f"--ansatz vit is incompatible with --sampler {args.sampler}. "
            "ViT cannot be mapped to an Ising problem. Use --sampler custom."
        )

    print(f"Configuration:")
    print(f"  Model: {_model_desc}")
    print(f"  System size: {args.size}")
    print(f"  Ansatz: {args.ansatz}", end="")
    if args.ansatz == "rbm":
        print(f" ({args.rbm})")
    else:
        print(f" (d_model={args.d_model}, n_layers={args.n_layers}, "
              f"n_heads={args.n_heads}, patch_size={args.patch_size})")
    print(f"  Sampler: {args.sampler} ({args.sampling_method})")
    print(f"  Training: {args.iterations} iterations, lr={args.learning_rate}")
    if args.ansatz == "rbm":
        print(
            f"  CEM β scheduling: {'ON (interval=' + str(args.cem_interval) + ')' if args.cem else 'OFF'}"
        )
    print(f"  JAX devices: {jax.devices()}")

    _1d_models = ("1d", "heisenberg_xxz_1d", "lr1d", "j1j2_1d", "heisenberg_xy_1d")
    _2d_models = ("2d", "heisenberg_xxz_2d")

    # 1. Instantiate model
    if args.model == "1d":
        ising = TransverseFieldIsing1D(args.size, args.h)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size, args.h)
    elif args.model == "heisenberg_xxz_1d":
        ising = HeisenbergXXZ1D(args.size, J=args.J, delta=args.delta)
    elif args.model == "lr1d":
        ising = LongRangeTFIM1D(args.size, args.h, alpha=args.alpha)
    elif args.model == "j1j2_1d":
        ising = J1J2Ising1D(args.size, J1=args.J1, J2=args.J2, h=args.h)
    elif args.model == "heisenberg_xy_1d":
        ising = HeisenbergXY1D(args.size, J=args.J)
    elif args.model == "heisenberg_xxz_2d":
        ising = HeisenbergXXZ2D(args.size, J=args.J, delta=args.delta)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    n_visible = args.size if args.model in _1d_models else args.size**2
    key, model_key = jax.random.split(key)

    # 2. Instantiate ansatz
    if args.ansatz == "rbm":
        if args.n_hidden is not None:
            n_hidden = args.n_hidden
        elif args.model in _1d_models:
            n_hidden = args.size
        elif args.model in _2d_models:
            n_hidden = args.size**2
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        args.n_hidden = n_hidden
        if args.rbm == "full":
            wave_fn = FullyConnectedRBM(n_visible, n_hidden, model_key)
        else:
            wave_fn = DWaveTopologyRBM(n_visible, n_hidden, model_key, solver=args.rbm)
    else:  # vit
        geometry = "2d" if args.model in _2d_models else "1d"
        wave_fn = ViTWaveFunction(
            n_visible=n_visible,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            patch_size=args.patch_size,
            key=model_key,
            geometry=geometry,
        )
        # Set n_hidden to None so helpers don't crash on RBM-specific fields
        args.n_hidden = None

    # 3. Instantiate sampler
    if args.ansatz == "vit":
        key, sampler_key = jax.random.split(key)
        sampler = GenericClassicalSampler(
            n_warmup=getattr(args, "n_warmup", 20),
            n_sweeps=1,
        )
        sampler._key = sampler_key
    elif args.sampler == "custom":
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

    # 4. Build trainer config and run
    if args.ansatz == "vit":
        trainer_config = {
            "learning_rate": args.learning_rate,
            "n_iterations": args.iterations,
            "n_samples": args.n_samples,
            "regularization": args.regularization,
            "seed": args.seed,
        }
        trainer = TrainerGeneric(wave_fn, ising, sampler, trainer_config, args=args)
        print("\nStarting ViT training...")
        history = trainer.train()
        save_results(args, history, ising, rbm=None)
    else:
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
        trainer = Trainer(wave_fn, ising, sampler, trainer_config, args=args)
        print(f"\nStarting RBM training...")
        history = trainer.train()
        save_results(args, history, ising, wave_fn)
        if args.rbm != "full" and hasattr(wave_fn, "connectivity_summary"):
            print(f"sparsity: {wave_fn.connectivity_summary()['sparsity']}")


if __name__ == "__main__":
    main()

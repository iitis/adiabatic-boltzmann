import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)

import argparse

from helpers import save_results
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler
from encoder import Trainer
from ising import (
    TransverseFieldIsing1D,
    TransverseFieldIsing2D,
    J1J2HeisenbergXXZ1D,
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

    parser.add_argument(
        "--model",
        choices=["1d", "2d", "heisenberg_j1j2_1d"],
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
        "--J1", type=float, default=1.0, help="NN coupling J₁ (J1J2 model)"
    )
    parser.add_argument(
        "--J2", type=float, default=0.5, help="NNN coupling J₂ (J1J2 model)"
    )
    parser.add_argument(
        "--delta", type=float, default=1.0, help="XXZ anisotropy Δ (Heisenberg)"
    )

    parser.add_argument(
        "--rbm",
        choices=["full", "pegasus", "zephyr"],
        default="full",
        help="RBM connectivity pattern.",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=None,
        help="Number of RBM hidden units (default: equal to visible)",
    )

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
            "gibbs",
            "lsb",
            "exchange",
        ],
        default="simulated_annealing",
        help="Classical sampling algorithm.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Samples per iteration"
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of disjoint chip embeddings to sample in a single QPU call "
             "(parallel embedding). Only valid with --sampler dimod, "
             "--sampling-method pegasus/zephyr, and --rbm full; --n-samples must "
             "be divisible by this value.",
    )

    parser.add_argument(
        "--iterations", type=int, default=30, help="Training iterations"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Gradient step size"
    )
    parser.add_argument(
        "--regularization", type=float, default=1e-5, help="SR matrix regularization"
    )

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

    parser.add_argument(
        "--output-dir", type=str, default=str(Path(__file__).parent.parent / "results"), help="Directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--visualize", action="store_true", help="Plot convergence curves", default=True
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # shared PRNG key for RBM init, sampler, trainer
    key = jax.random.PRNGKey(args.seed)

    if args.model == "heisenberg_j1j2_1d":
        _model_desc = f"{args.model} with J1={args.J1}, J2={args.J2}, Δ={args.delta}"
    else:
        _model_desc = f"{args.model} with h={args.h}"

    if args.n_parallel > 1:
        if not (
            args.rbm == "full"
            and args.sampler == "dimod"
            and args.sampling_method in ("pegasus", "zephyr")
        ):
            raise ValueError(
                f"--n-parallel {args.n_parallel} requires --rbm full "
                "--sampler dimod --sampling-method pegasus/zephyr (parallel "
                "embedding replicates the dense biclique across disjoint chip "
                "regions; DWaveTopologyRBM is not supported). Got --rbm "
                f"{getattr(args, 'rbm', None)} --sampler "
                f"{args.sampler} --sampling-method {args.sampling_method}."
            )
        if args.n_samples % args.n_parallel != 0:
            raise ValueError(
                f"--n-samples {args.n_samples} must be divisible by "
                f"--n-parallel {args.n_parallel}."
            )

    print(f"Configuration:")
    print(f"  Model: {_model_desc}")
    print(f"  System size: {args.size}")
    print(f"  RBM: {args.rbm}")
    print(f"  Sampler: {args.sampler} ({args.sampling_method})")
    print(f"  Training: {args.iterations} iterations, lr={args.learning_rate}")
    print(
        f"  CEM β scheduling: {'ON (interval=' + str(args.cem_interval) + ')' if args.cem else 'OFF'}"
    )
    print(f"  JAX devices: {jax.devices()}")

    _1d_models = ("1d", "heisenberg_j1j2_1d")
    _2d_models = ("2d",)

    # 1. Instantiate model
    if args.model == "1d":
        ising = TransverseFieldIsing1D(args.size, args.h)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size, args.h)
    elif args.model == "heisenberg_j1j2_1d":
        ising = J1J2HeisenbergXXZ1D(args.size, J1=args.J1, J2=args.J2, delta=args.delta)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    n_visible = args.size if args.model in _1d_models else args.size**2
    key, model_key = jax.random.split(key)

    # 2. Instantiate ansatz
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

    # 4. Build trainer config and run
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
        "n_parallel": args.n_parallel,
    }
    trainer = Trainer(wave_fn, ising, sampler, trainer_config, args=args)
    print(f"\nStarting RBM training...")
    history = trainer.train()
    save_results(args, history, ising, wave_fn, energy_j=trainer.total_energy_j, sampler=sampler)
    if args.rbm != "full" and hasattr(wave_fn, "connectivity_summary"):
        print(f"sparsity: {wave_fn.connectivity_summary()['sparsity']}")


if __name__ == "__main__":
    main()

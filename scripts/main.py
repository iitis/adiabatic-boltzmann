import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)

import argparse

from helpers import save_results
from model import FullyConnectedRBM, DWaveTopologyRBM, FullBoltzmannMachine
from model_vit import ViTWaveFunction
from model_dbm import DeepBoltzmannMachine
from sampler import ClassicalSampler, DimodSampler, DWaveMHSampler, VeloxSampler, GenericClassicalSampler
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

    # Model parameters
    parser.add_argument(
        "--model",
        choices=["1d", "2d", "heisenberg_xxz_1d", "lr1d", "j1j2_1d", "heisenberg_xy_1d", "heisenberg_xxz_2d", "heisenberg_j1j2_1d"],
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
        choices=["rbm", "vit", "dbm"],
        default="rbm",
        help="Wave function ansatz: RBM, Vision Transformer, or Deep Boltzmann Machine",
    )

    # DBM architecture (used when --ansatz dbm)
    parser.add_argument(
        "--dbm-hidden",
        type=str,
        default="8",
        help="Comma-separated hidden layer sizes for DBM, e.g. '8,4' (--ansatz dbm only)",
    )
    parser.add_argument(
        "--n-mf-steps",
        type=int,
        default=10,
        help="Mean-field iterations for DBM log_psi approximation (--ansatz dbm only)",
    )

    # RBM architecture (used when --ansatz rbm)
    parser.add_argument(
        "--rbm",
        choices=["full", "fullbm", "pegasus", "zephyr"],
        default="full",
        help="RBM/FBM connectivity pattern (--ansatz rbm only). "
             "'fullbm' adds visible-visible couplings J.",
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
            "pegasus_mh",
            "zephyr_mh",
            "pegasus_ra",
            "zephyr_ra",
            "pegasus_fast",
            "zephyr_fast",
            "metropolis",
            "velox",
            "simulated_annealing",
            "tabu",
            "gibbs",
            "lsb",
            "exchange",
        ],
        default="simulated_annealing",
        help="Classical sampling algorithm. "
             "pegasus_mh/zephyr_mh use D-Wave proposals inside an MH chain.",
    )
    parser.add_argument(
        "--mh-warmup",
        type=int,
        default=0,
        help="D-Wave query rounds used as MH warmup (no QPU budget savings; "
             "only used with pegasus_mh / zephyr_mh)",
    )
    parser.add_argument(
        "--mh-sweeps",
        type=int,
        default=1,
        help="D-Wave query rounds per training iteration for MH collection "
             "(default 1; only used with pegasus_mh / zephyr_mh)",
    )
    parser.add_argument(
        "--ra-s-target",
        type=float,
        default=0.45,
        help="Reverse anneal: target s value to reverse to (0=full quantum, 1=classical). "
             "Default 0.45. Only used with pegasus_ra / zephyr_ra.",
    )
    parser.add_argument(
        "--ra-pause-time",
        type=int,
        default=10,
        help="Reverse anneal: microseconds to hold at s_target. Default 10.",
    )
    parser.add_argument(
        "--ra-anneal-time",
        type=int,
        default=10,
        help="Reverse anneal: microseconds for each reverse/forward anneal leg. Default 10.",
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
        "--output-dir", type=str, default=str(Path(__file__).parent.parent / "results"), help="Directory for results"
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
    elif args.model == "heisenberg_j1j2_1d":
        _model_desc = f"{args.model} with J1={args.J1}, J2={args.J2}, Δ={args.delta}"
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
    if args.ansatz == "dbm" and args.sampler == "velox":
        raise ValueError(
            "--ansatz dbm is not compatible with --sampler velox. "
            "Use --sampler custom (Metropolis) or --sampler dimod (SA / D-Wave QPU)."
        )

    print(f"Configuration:")
    print(f"  Model: {_model_desc}")
    print(f"  System size: {args.size}")
    print(f"  Ansatz: {args.ansatz}", end="")
    if args.ansatz == "rbm":
        print(f" ({args.rbm})")
    elif args.ansatz == "dbm":
        print(f" (hidden={args.dbm_hidden}, n_mf_steps={args.n_mf_steps})")
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

    _1d_models = ("1d", "heisenberg_xxz_1d", "lr1d", "j1j2_1d", "heisenberg_xy_1d", "heisenberg_j1j2_1d")
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
    elif args.model == "heisenberg_j1j2_1d":
        ising = J1J2HeisenbergXXZ1D(args.size, J1=args.J1, J2=args.J2, delta=args.delta)
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
        elif args.rbm == "fullbm":
            wave_fn = FullBoltzmannMachine(n_visible, n_hidden, model_key)
        else:
            wave_fn = DWaveTopologyRBM(n_visible, n_hidden, model_key, solver=args.rbm)
    elif args.ansatz == "dbm":
        hidden_sizes = [int(x) for x in args.dbm_hidden.split(",")]
        wave_fn = DeepBoltzmannMachine(
            n_visible=n_visible,
            hidden_sizes=hidden_sizes,
            key=model_key,
            n_mf_steps=args.n_mf_steps,
        )
        args.n_hidden = None
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
    elif args.ansatz == "dbm":
        if args.sampler == "custom":
            key, sampler_key = jax.random.split(key)
            sampler = GenericClassicalSampler(
                n_warmup=getattr(args, "n_warmup", 20),
                n_sweeps=1,
            )
            sampler._key = sampler_key
        elif args.sampler == "dimod":
            if args.sampling_method in ("pegasus_mh", "zephyr_mh"):
                sampler = DWaveMHSampler(
                    method=args.sampling_method,
                    n_warmup=args.mh_warmup,
                    n_sweeps=args.mh_sweeps,
                )
            else:
                sampler = DimodSampler(method=args.sampling_method)
        else:
            raise ValueError(f"Unsupported sampler '{args.sampler}' for --ansatz dbm.")
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
        if args.sampling_method in ("pegasus_mh", "zephyr_mh"):
            sampler = DWaveMHSampler(
                method=args.sampling_method,
                n_warmup=args.mh_warmup,
                n_sweeps=args.mh_sweeps,
            )
        else:
            sampler = DimodSampler(method=args.sampling_method)
    elif args.sampler == "velox":
        sampler = VeloxSampler(method=args.sampling_method)

    # 4. Build trainer config and run
    if args.ansatz in ("vit", "dbm"):
        trainer_config = {
            "learning_rate": args.learning_rate,
            "n_iterations": args.iterations,
            "n_samples": args.n_samples,
            "regularization": args.regularization,
            "seed": args.seed,
        }
        trainer = TrainerGeneric(wave_fn, ising, sampler, trainer_config, args=args)
        label = "ViT" if args.ansatz == "vit" else "DBM"
        print(f"\nStarting {label} training...")
        history = trainer.train()
        save_results(args, history, ising, rbm=None)
    else:
        _is_dwave = args.sampling_method in (
            "pegasus", "zephyr", "pegasus_mh", "zephyr_mh", "pegasus_ra", "zephyr_ra",
            "pegasus_fast", "zephyr_fast",
        )
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
            "ra_s_target": args.ra_s_target,
            "ra_pause_time": args.ra_pause_time,
            "ra_anneal_time": args.ra_anneal_time,
        }
        trainer = Trainer(wave_fn, ising, sampler, trainer_config, args=args)
        print(f"\nStarting RBM training...")
        history = trainer.train()
        save_results(args, history, ising, wave_fn)
        if args.rbm != "full" and hasattr(wave_fn, "connectivity_summary"):
            print(f"sparsity: {wave_fn.connectivity_summary()['sparsity']}")


if __name__ == "__main__":
    main()

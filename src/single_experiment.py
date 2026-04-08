# run_single_experiment.py
"""
Runs exactly one experiment and exits.
All arguments passed via CLI — no shared state with any other process.
"""

import argparse
import numpy as np

from pathlib import Path
from helpers import save_results, save_rbm_checkpoint, restore_rbm_from_checkpoint
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from argparse import Namespace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--sampler", type=str, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=str, default="results/")
    p.add_argument("--model", choices=["1d", "2d"], default="1d")

    p.add_argument(
        "--n-hidden",
        type=int,
        default=None,
        help="Number of hidden units. Defaults to size (1D) or size (2D linear dim).",
    )

    p.add_argument("--h", type=float, required=True, help="Transverse field strength.")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--rbm", choices=["full", "pegasus", "zephyr"], default="full")
    p.add_argument("--lsb-sigma", type=float, default=0.0,
                   help="LSB noise std σ (0 = auto-scale to RMS of local fields)")
    p.add_argument("--lsb-sigma-scale", type=float, default=1.0,
                   help="Multiplier applied on top of auto-scaled σ")
    p.add_argument("--lsb-steps", type=int, default=100,
                   help="LSB steps per sample M (only used when --sampler lsb)")
    p.add_argument("--sb-mode", choices=["discrete", "ballistic"], default="discrete",
                   help="SBM algorithm variant (only used when --sampler custom --method sbm)")
    p.add_argument("--sb-heated", action="store_true", default=False,
                   help="Enable heated SBM variant")
    p.add_argument("--sb-max-steps", type=int, default=10000,
                   help="Max SBM iterations per agent")
    p.add_argument("--gibbs-sweeps", type=int, default=10,
                   help="Block Gibbs sweeps per sample call k (only used when --method gibbs)")
    p.add_argument("--sbm-steps", type=int, default=5000,
                   help="SBM num_steps (only used when --sampler velox --method sbm)")
    p.add_argument("--sbm-dt", type=float, default=1.0,
                   help="SBM time step dt (only used when --sampler velox --method sbm)")
    p.add_argument("--sbm-discrete", action="store_true", default=False,
                   help="Use discrete SBM algorithm")
    p.add_argument("--resume", action="store_true",
                   help="Warm-start from the latest checkpoint matching this config, if one exists.")
    return p.parse_args()


def find_latest_checkpoint(ns_args) -> Path | None:
    """Return the highest-iteration checkpoint matching this config, or None."""
    checkpoint_dir = Path(
        f"{ns_args.output_dir.replace('results', 'checkpoints')}"
        f"/{ns_args.size}/{ns_args.sampler}/{ns_args.sampling_method}/{ns_args.rbm}"
    )
    pattern = (
        f"checkpoint_{ns_args.model}_h{ns_args.h}_rbm{ns_args.rbm}"
        f"_nh{ns_args.n_hidden}_lr{ns_args.learning_rate}_iter*.pkl"
    )
    matches = sorted(checkpoint_dir.glob(pattern))
    return matches[-1] if matches else None


def main():
    args = parse_args()
    np.random.seed(args.seed)

    n_visible = args.size if args.model == "1d" else args.size**2
    # 1. Instantiate Ising model
    if args.model == "1d":
        ising = TransverseFieldIsing1D(args.size, h=args.h)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size, h=args.h)
    n_hidden = args.n_hidden
    if args.rbm == "full":
        rbm = FullyConnectedRBM(n_visible, n_hidden)
    elif args.rbm in ("pegasus", "zephyr"):
        rbm = DWaveTopologyRBM(n_visible, n_hidden, solver=args.rbm, seed=args.seed)
    else:
        raise ValueError(f"Unknown RBM type: {args.rbm}")

    if args.sampler == "custom":
        sampler = ClassicalSampler(
            method=args.method,
            n_sweeps=args.gibbs_sweeps if args.method == "gibbs" else 1,
            sb_mode=args.sb_mode,
            sb_heated=args.sb_heated,
            sb_max_steps=args.sb_max_steps,
        )
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.method)
    elif args.sampler == "velox":
        sampler = VeloxSampler(
            method=args.method,
            sbm_steps=args.sbm_steps,
            sbm_dt=args.sbm_dt,
            sbm_discrete=args.sbm_discrete,
        )
    elif args.sampler == "lsb":
        sampler = LSBSampler(
            sigma=args.lsb_sigma,
            n_steps=args.lsb_steps,
            sigma_scale=args.lsb_sigma_scale,
        )
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    trainer_config = {
        "learning_rate": args.lr,
        "n_iterations": args.iterations,
        "n_samples": 1000,
        "regularization": 1e-3,
        "stop_at_convergence": False,
        "save_checkpoints": args.method in ("pegasus", "zephyr"),
        "checkpoint_interval": 10,
    }

    ns_args = Namespace(
        model=args.model,
        size=args.size,
        h=args.h,
        rbm=args.rbm,
        n_hidden=n_hidden,
        sampler=args.sampler,
        sampling_method=args.method,
        iterations=args.iterations,
        learning_rate=args.lr,
        regularization=1e-3,
        n_samples=1000,
        output_dir=args.output_dir,
        seed=args.seed,
        visualize=False,
        cem=False,
        lsb_sigma=args.lsb_sigma,
        lsb_sigma_scale=args.lsb_sigma_scale,
        lsb_steps=args.lsb_steps,
        sbm_steps=args.sbm_steps,
        sbm_dt=args.sbm_dt,
        sbm_discrete=args.sbm_discrete,
        sb_mode=args.sb_mode,
        sb_heated=args.sb_heated,
        sb_max_steps=args.sb_max_steps,
    )

    if args.resume:
        ckpt = find_latest_checkpoint(ns_args)
        if ckpt:
            restore_rbm_from_checkpoint(rbm, ckpt)
        else:
            print("  [resume] No checkpoint found for this config — starting from scratch.")

    trainer = Trainer(rbm, ising, sampler, trainer_config, args=ns_args)
    history = trainer.train()
    save_results(ns_args, history, ising, rbm)

    ckpt_path = save_rbm_checkpoint(rbm, ns_args, args.iterations)
    print(f"  Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()

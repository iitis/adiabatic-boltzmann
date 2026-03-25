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

    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--rbm", choices=["full", "pegasus", "zephyr"], default="full")
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
        ising = TransverseFieldIsing1D(args.size)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size)
    n_hidden = args.n_hidden
    if args.rbm == "full":
        rbm = FullyConnectedRBM(n_visible, n_hidden)
    elif args.rbm in ("pegasus", "zephyr"):
        rbm = DWaveTopologyRBM(n_visible, n_hidden, solver=args.rbm, seed=args.seed)
    else:
        raise ValueError(f"Unknown RBM type: {args.rbm}")

    if args.sampler == "custom":
        sampler = ClassicalSampler(method=args.method)
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.method)
    elif args.sampler == "velox":
        sampler = VeloxSampler(method=args.method)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    trainer_config = {
        "learning_rate": args.lr,
        "n_iterations": args.iterations,
        "n_samples": 1000,
        "regularization": 1e-3,
        "stop_at_convergence": False,
    }

    ns_args = Namespace(
        model=args.model,
        size=args.size,
        h=0.5,
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
    )

    if args.resume:
        ckpt = find_latest_checkpoint(ns_args)
        if ckpt:
            restore_rbm_from_checkpoint(rbm, ckpt)
        else:
            print("  [resume] No checkpoint found for this config — starting from scratch.")

    trainer = Trainer(rbm, ising, sampler, trainer_config)
    history = trainer.train()
    save_results(ns_args, history, ising)

    ckpt_path = save_rbm_checkpoint(rbm, ns_args, args.iterations)
    print(f"  Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()

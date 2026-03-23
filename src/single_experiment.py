# run_single_experiment.py
"""
Runs exactly one experiment and exits.
All arguments passed via CLI — no shared state with any other process.
"""

import argparse
import numpy as np

from helpers import save_results
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D
from argparse import Namespace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--sampler", type=str, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=str, default="results/")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    ising = TransverseFieldIsing1D(args.size, 0.5)
    rbm = FullyConnectedRBM(args.size, args.size)

    if args.sampler == "custom":
        sampler = ClassicalSampler(method=args.method)
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.method)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    trainer_config = {
        "learning_rate": args.lr,
        "n_iterations": 300,
        "n_samples": 1000,
        "regularization": 1e-3,
    }

    ns_args = Namespace(
        model="1d",
        size=args.size,
        h=0.5,
        rbm="full",
        n_hidden=args.size,
        sampler=args.sampler,
        sampling_method=args.method,
        iterations=300,
        learning_rate=args.lr,
        regularization=1e-3,
        n_samples=1000,
        output_dir=args.output_dir,
        seed=args.seed,
        visualize=False,
    )

    trainer = Trainer(rbm, ising, sampler, trainer_config)
    history = trainer.train()
    save_results(ns_args, history, ising)


if __name__ == "__main__":
    main()

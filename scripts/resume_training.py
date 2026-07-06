#!/usr/bin/env python3
"""
Resume training from an RBM checkpoint.

Example usage:
    python scripts/resume_training.py --checkpoint checkpoints/16/custom/metropolis/full/checkpoint_1d_h0.5_rbmfull_nh16_lr0.01_iter0050.pkl --iterations 100
"""

import argparse
import sys
import numpy as np
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from helpers import load_rbm_checkpoint, restore_rbm_from_checkpoint, save_results
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume RBM training from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pkl)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Additional iterations to train"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (default: use checkpoint value)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory for results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    rbm_state, config, start_iteration = load_rbm_checkpoint(checkpoint_path)
    
    print(f"Checkpoint config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"Starting from iteration: {start_iteration}")
    
    # Create Ising model
    model_type = config["model"]
    size = config["size"]
    h = config["h"]
    
    if model_type == "1d":
        ising = TransverseFieldIsing1D(size, h)
    elif model_type == "2d":
        ising = TransverseFieldIsing2D(size, h)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    # Create RBM and restore from checkpoint
    rbm_type = config["rbm"]
    n_hidden = config["n_hidden"]
    
    if rbm_type == "full":
        rbm = FullyConnectedRBM(size, n_hidden)
    else:
        rbm = DWaveTopologyRBM(size, n_hidden, solver=rbm_type)
    
    restore_rbm_from_checkpoint(rbm, checkpoint_path)
    
    # Create sampler
    sampler_type = config["sampler"]
    sampling_method = config["sampling_method"]
    
    if sampler_type == "custom":
        sampler = ClassicalSampler(method=sampling_method)
    elif sampler_type == "dimod":
        sampler = DimodSampler(method=sampling_method)
    elif sampler_type == "velox":
        sampler = VeloxSampler(method=sampling_method)
    else:
        raise ValueError(f"Unknown sampler: {sampler_type}")
    
    # Update config
    config["iterations"] = start_iteration + args.iterations
    config["output_dir"] = args.output_dir
    lr = args.learning_rate if args.learning_rate else config["learning_rate"]
    config["learning_rate"] = lr
    
    # Build trainer config
    trainer_config = {
        "learning_rate": lr,
        "n_iterations": args.iterations,  # Train for additional iterations
        "n_samples": config["n_samples"],
        "regularization": config["regularization"],
        "save_checkpoints": True,
        "checkpoint_interval": 10,
    }
    
    # Convert config dict to Namespace for compatibility
    from argparse import Namespace
    args_namespace = Namespace(**config)
    
    # Create trainer and resume training
    print(f"\nResuming training for {args.iterations} additional iterations...")
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=args_namespace)
    history = trainer.train()
    
    # Save results
    save_results(args_namespace, history, ising, rbm, energy_j=trainer.total_energy_j)
    
    print(f"\nTraining resumed from iteration {start_iteration} to {start_iteration + args.iterations}")
    print(f"Final energy: {history['energy'][-1]:.6f}")


if __name__ == "__main__":
    main()

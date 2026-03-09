import argparse
import json
import numpy as np
from pathlib import Path

# TODO: Import your implementations
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DWaveSampler
from encoder import Trainer, ExperimentRunner
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
    parser.add_argument('--model', choices=['1d', '2d'], default='1d',
                        help='Ising model type')
    parser.add_argument('--size', type=int, default=16,
                        help='System size (chain length or square lattice dimension)')
    parser.add_argument('--h', type=float, default=0.5,
                        help='Transverse field strength')
    
    # RBM architecture
    parser.add_argument('--rbm', choices=['full', 'dwave'], default='full',
                        help='RBM connectivity pattern')
    parser.add_argument('--n-hidden', type=int, default=None,
                        help='Number of hidden units (default: equal to visible)')
    
    # Sampling
    parser.add_argument('--sampler', choices=['classical', 'dwave'], default='classical',
                        help='Sampling backend')
    parser.add_argument('--sampling-method', choices=['metropolis', 'simulated_annealing'],
                        default='metropolis',
                        help='Classical sampling algorithm')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Samples per iteration')
    
    # Training
    parser.add_argument('--iterations', type=int, default=50,
                        help='Training iterations')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Gradient step size')
    parser.add_argument('--regularization', type=float, default=1e-5,
                        help='SR matrix regularization')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/',
                        help='Directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--visualize', action='store_true',
                        help='Plot convergence curves',default=True)
    
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
    if args.model == '1d':
        ising = TransverseFieldIsing1D(args.size, args.h)
    elif args.model == '2d':
        ising = TransverseFieldIsing2D(args.size, args.h)
    
    # 2. Instantiate RBM
    n_hidden = args.n_hidden or args.size
    if args.rbm == 'full':
        rbm = FullyConnectedRBM(args.size, n_hidden)
    else:
        rbm = DWaveTopologyRBM(args.size, n_hidden)
    
    # 3. Instantiate sampler
    if args.sampler == 'classical':
        sampler = ClassicalSampler(method=args.sampling_method)
    else:
        sampler = DWaveSampler()
    
    # 4. Build trainer config
    trainer_config = {
        'learning_rate': args.learning_rate,
        'n_iterations': args.iterations,
        'n_samples': args.n_samples,
        'regularization': args.regularization,
    }
    
    # 5. Create trainer and run
    trainer = Trainer(rbm, ising, sampler, trainer_config)
    
    print(f"\nStarting training...")
    history = trainer.train()
    
    # 6. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'config': vars(args),
        'history': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in history.items()},
        'final_energy': history['energy'][-1],
        'exact_energy': ising.exact_ground_energy()
    }
    
    output_file = output_dir / f"result_{args.model}_h{args.h}_rbm{args.rbm}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Final energy: {results['final_energy']:.6f}")
    print(f"Exact energy: {results['exact_energy']:.6f}")
    print(f"Error: {abs(results['final_energy'] - results['exact_energy']):.6f}")
    
    # 7. Plot if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['energy'])
            plt.axhline(results['exact_energy'], color='r', linestyle='--', label='Exact')
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.title('Convergence')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['error'])
            plt.xlabel('Iteration')
            plt.ylabel('Standard Error')
            plt.title('Energy Variance')
            
            plt.tight_layout()
            plot_file = output_dir / f"plot_{args.model}_h{args.h}_rbm{args.rbm}.png"
            plt.savefig(plot_file, dpi=150)
            plt.show()
            print(f"Plot saved to {plot_file}")
            
        except ImportError:
            print("Matplotlib not available, skipping visualization")


if __name__ == '__main__':
    main()

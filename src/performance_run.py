import numpy as np
import itertools

from helpers import save_results
from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D

from argparse import Namespace

sizes = [4, 8, 16, 32]
hs = [0.1, 0.5, 1.0, 2.0]
rbms = ["full", "dwave"]
sampler_methods = [("velox", "velox"), ("dimod", "simulated_annealing")]
iterations = [10, 30, 100]
learning_rates = [0.01, 0.1, 0.2]
regularizations = [1e-6, 1e-5, 1e-4]
n_samples = [100, 1000, 5000]
seeds = [1, 42, 123]

output_dir = "results/"


def run_experiment(args):
    np.random.seed(args.seed)

    # 1. Instantiate Ising model
    ising = TransverseFieldIsing1D(args.size, args.h)

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
    history = trainer.train()
    save_results(args, history, ising)


if __name__ == "__main__":
    for size, h, rbm, (
        sampler,
        sampling_method,
    ), iteration, lr, reg, n_sample, seed in itertools.product(
        sizes,
        hs,
        rbms,
        sampler_methods,
        iterations,
        learning_rates,
        regularizations,
        n_samples,
        seeds,
    ):
        for n_hidden in [size, 2 * size]:
            args = Namespace(
                model="1d",
                size=size,
                h=h,
                rbm=rbm,
                n_hidden=n_hidden,
                sampler=sampler,
                sampling_method=sampling_method,
                iterations=iteration,
                learning_rate=lr,
                regularization=reg,
                n_samples=n_sample,
                output_dir=output_dir,
                seed=seed,
                visualize=False,
            )
            run_experiment(args)

#!/usr/bin/env python3
"""
mcmc_matched_sweep.py — run Metropolis / Gibbs / LSB at the exact
(lr, reg, n_samples, iterations) cell already used for the FPGA/VeloxQ
"sweeps100" campaign, across the same system sizes, so Figure 10
(scripts/viz/paper_figures.py) can compare all five solvers at one
matched hyperparameter point instead of each family's own best cell.

Runs in-process (Trainer/ClassicalSampler directly, same construction as
scripts/main.py's --sampler custom path) — never shells out to main.py.

Writes results to the same location/format main.py would (via
helpers.save_results), so paper_figures.py's existing mcmc_recs() loader
picks these up unchanged once pointed at lr=0.08/reg=0.05/ns=200/iter=100:
    results/tfim_1d/{N}/custom/{method}/result_1d_h{h}_rbmfull_nh{N}_lr0.08_reg0.05_ns200_seed{s}_iter100_cem0_sigma1.0.json.gz

Usage:
    python scripts/exper/mcmc_matched_sweep.py
    python scripts/exper/mcmc_matched_sweep.py --methods lsb --cem   # lsb+CEM variant
    python scripts/exper/mcmc_matched_sweep.py --sizes 8 12 --seeds 3 --smoke-test
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import jax
jax.config.update("jax_enable_x64", True)
from argparse import Namespace

from helpers import save_results
from model import FullyConnectedRBM
from sampler import ClassicalSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D

DEFAULT_SIZES = [8, 12, 16, 24, 32, 64, 128]
DEFAULT_METHODS = ["metropolis", "gibbs", "lsb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                   choices=["metropolis", "gibbs", "lsb"])
    p.add_argument("--seeds", type=int, default=20, help="Number of seeds, 0..seeds-1")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--gibbs-sweeps", type=int, default=10)
    p.add_argument("--cem", action="store_true", default=False,
                   help="Enable CEM beta_eff scheduling for this invocation "
                        "(run once without and once with to get both variants)")
    p.add_argument("--output-dir", type=str, default=str(_ROOT / "results"))
    p.add_argument("--smoke-test", action="store_true",
                   help="1 size, 1 seed, 5 iterations — verify the plumbing only")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip (N, method, seed) combos whose result file already exists")
    return p.parse_args()


def run_one(size, method, seed, args):
    ns_args = Namespace(
        model="1d", size=size, h=args.h, rbm="full", n_hidden=size,
        sampler="custom", sampling_method=method,
        iterations=args.iterations, learning_rate=args.lr,
        regularization=args.reg, n_samples=args.n_samples,
        output_dir=args.output_dir, seed=seed, visualize=False, cem=args.cem,
    )

    out_file = (
        Path(args.output_dir) / "tfim_1d" / str(size) / "custom" / method /
        f"result_1d_h{args.h}_rbmfull_nh{size}_lr{args.lr}_reg{args.reg}"
        f"_ns{args.n_samples}_seed{seed}_iter{args.iterations}_cem{int(args.cem)}_sigma1.0.json.gz"
    )
    if args.skip_existing and out_file.exists():
        print(f"  skip (exists): {out_file}")
        return

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    ising = TransverseFieldIsing1D(size, args.h)
    rbm = FullyConnectedRBM(size, size, model_key)

    sampler = ClassicalSampler(
        method=method,
        n_sweeps=args.gibbs_sweeps if method == "gibbs" else 1,
    )
    key, sampler_key = jax.random.split(key)
    sampler._key = sampler_key

    trainer_config = {
        "learning_rate": args.lr,
        "n_iterations": args.iterations,
        "n_samples": args.n_samples,
        "regularization": args.reg,
        "seed": seed,
        "use_cem": args.cem,
    }
    trainer = Trainer(rbm, ising, sampler, trainer_config, args=ns_args)
    history = trainer.train()
    save_results(ns_args, history, ising, rbm, energy_j=trainer.total_energy_j)


def main():
    args = parse_args()
    if args.smoke_test:
        args.sizes = args.sizes[:1]
        args.seeds = 1
        args.iterations = 5

    for method in args.methods:
        for size in args.sizes:
            for seed in range(args.seeds):
                print(f"=== {method} N={size} seed={seed} ===")
                run_one(size, method, seed, args)


if __name__ == "__main__":
    main()

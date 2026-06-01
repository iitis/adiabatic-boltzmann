#!/usr/bin/env python3
"""
Run VMC with FPGASampler using best hyperparameters found by optuna_sa_sweep.py.

By default picks up both N=16 and N=24 results.  Run without any arguments:

    python run_fpga_best.py

The script constructs the params-file path using the same naming convention as
optuna_sa_sweep.py:

    <optuna-dir>/best_<model>_N<size>_h<h>.json

SA → FPGA parameter mapping applied automatically:
    sa.T_initial  →  FPGASampler(start_temp=...)
    sa.T_final    →  FPGASampler(stop_temp=...)
    sa.n_sweeps   →  FPGASampler(num_steps=...)   # total cooling budget
    vmc.n_samples →  FPGASampler(num_rep=...)     # replicas ≥ n_samples

Override a single file:
    python run_fpga_best.py --params ../optuna_results/best_1d_N16_h0p5.json
"""

import argparse
import json
from pathlib import Path

import jax

from encoder import Trainer
from helpers import save_results
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import FPGASampler

# Must match optuna_sa_sweep.py defaults so paths resolve without arguments.
DEFAULT_SIZES = [16, 24]
DEFAULT_MODEL = "1d"
DEFAULT_H     = 0.5


def _parse_args():
    p = argparse.ArgumentParser(
        description="FPGA VMC run using best SA hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--params",
        default=None,
        help="Explicit JSON file from optuna_sa_sweep.py.  "
             "If given, --sizes / --model / --h are ignored.",
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        metavar="N",
        help="Run FPGA for each size, loading its params JSON automatically.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["1d", "2d"])
    p.add_argument("--h", type=float, default=DEFAULT_H)
    p.add_argument(
        "--optuna-dir",
        default=str(Path(__file__).parent.parent / "optuna_results"),
        help="Directory written by optuna_sa_sweep.py.",
    )
    p.add_argument(
        "--iterations", type=int, default=100,
        help="FPGA training iterations.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "results"),
    )
    return p.parse_args()


def _build_ising(model, size, h):
    if model == "1d":
        return TransverseFieldIsing1D(size, h)
    if model == "2d":
        return TransverseFieldIsing2D(size, h)
    raise ValueError(f"Unknown model: {model!r}")


def _run_one(params_path, iterations, seed, output_dir):
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(
            f"Params file not found: {params_path}\n"
            "Run optuna_sa_sweep.py first to generate it."
        )

    with open(params_path) as f:
        params = json.load(f)

    model = params["model"]
    size  = params["size"]
    h     = params["h"]
    top_trials = params["top_trials"]

    ising = _build_ising(model, size, h)
    n_visible = size if model == "1d" else size ** 2

    # One FPGASampler shared across all 5 trials — avoids repeated Julia server
    # startups. Per-trial SA schedule is passed through trainer_config, which
    # FPGASampler.sample() reads via config keys fpga_start_temp / fpga_stop_temp
    # / fpga_num_steps.  num_rep is set to the largest n_samples across trials.
    max_n_samples = max(e["vmc"]["n_samples"] for e in top_trials)
    fpga_num_rep = max(max_n_samples, 1024)
    sampler = FPGASampler(num_rep=fpga_num_rep)

    for entry in top_trials:
        rank  = entry["rank"]
        sa    = entry["sa"]
        vmc   = entry["vmc"]
        n_hidden       = vmc["n_hidden"]
        learning_rate  = vmc["learning_rate"]
        regularization = vmc["regularization"]
        n_samples      = vmc["n_samples"]

        key = jax.random.PRNGKey(seed + rank)
        _, model_key = jax.random.split(key)
        rbm = FullyConnectedRBM(n_visible, n_hidden, model_key)

        trainer_ns = argparse.Namespace(
            model=model,
            size=size,
            h=h,
            rbm="full",
            sampler="fpga",
            sampling_method="fpga",
            ansatz="rbm",
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            regularization=regularization,
            n_samples=n_samples,
            iterations=iterations,
            seed=seed + rank,
            visualize=True,
            output_dir=output_dir,
            patch_size=2,
            mh_warmup=0,
            mh_sweeps=1,
            ra_s_target=0.45,
            ra_pause_time=10,
            ra_anneal_time=10,
            sigma=1.0,
            cem=False,
            cem_interval=5,
        )

        trainer_config = {
            "learning_rate":   learning_rate,
            "n_iterations":    iterations,
            "n_samples":       n_samples,
            "regularization":  regularization,
            "fpga_start_temp": sa["T_initial"],
            "fpga_stop_temp":  sa["T_final"],
            "fpga_num_steps":  sa["n_sweeps"],
            "fpga_num_rep":    fpga_num_rep,
        }

        print(f"\n{'='*60}")
        print(f"FPGA VMC  N={size}  rank={rank}/5  |  {params_path.name}")
        print(f"  T: {sa['T_initial']:.3f} -> {sa['T_final']:.4f}"
              f"  n_steps={sa['n_sweeps']}")
        print(f"  lr={learning_rate:.4f}  reg={regularization:.2e}"
              f"  nh={n_hidden}  ns={n_samples}")
        print(f"{'='*60}")

        trainer = Trainer(rbm, ising, sampler, trainer_config, args=trainer_ns)
        history = trainer.train()
        save_results(rbm, ising, history, trainer_ns)


def main():
    args = _parse_args()

    if args.params is not None:
        # Explicit file: run just that one.
        _run_one(args.params, args.iterations, args.seed, args.output_dir)
        return

    # Auto-discover one JSON per requested size.
    optuna_dir = Path(args.optuna_dir)
    h_str = str(args.h).replace(".", "p")
    for size in args.sizes:
        params_path = optuna_dir / f"best_{args.model}_N{size}_h{h_str}.json"
        _run_one(str(params_path), args.iterations, args.seed, args.output_dir)


if __name__ == "__main__":
    main()

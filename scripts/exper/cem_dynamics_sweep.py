"""
cem_dynamics_sweep.py -- generate matched CEM on/off pairs for
scripts/viz/plot_cem_dynamics.py.

Trains TFIM 1D with the LSB sampler at N=16 (<= KL_EXACT_MAX_N, so exact
KL to the RBM's own distribution is tracked every iteration) across a
grid of (h, seed), once with --cem and once without, at identical LSB
dynamics settings (steps/delta/gamma left at src/sampler.py's current
defaults for both runs) so the two sides of each pair differ ONLY in the
cem flag.

Follows the direct-Trainer pattern used by scripts/j1j2/j1j2_bench.py
(_run_rbm) rather than shelling out to scripts/main.py, so state (RNG,
model, sampler) is constructed the same way every other batch experiment
runner in this repo does it.

Idempotent: skips any (h, seed, cem) combination whose result file
already exists, so it can be safely re-run to resume after an interrupt.

Usage (from project root):
    python scripts/exper/cem_dynamics_sweep.py
"""

import sys
import time
from argparse import Namespace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)

from encoder import Trainer
from helpers import _model_params_str, _model_subdir, save_results
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import ClassicalSampler

N = 16
H_VALUES = [0.5, 1.0, 2.0]
SEEDS = [1, 2, 3]
LEARNING_RATE = 0.01
ITERATIONS = 300
N_SAMPLES = 1000
REGULARIZATION = 1e-5
SIGMA = 1.0
OUTPUT_DIR = str(_REPO / "results")


def _args(h, seed, cem) -> Namespace:
    return Namespace(
        model="1d", size=N, h=h, ansatz="rbm", rbm="full", n_hidden=N,
        sampler="custom", sampling_method="lsb",
        n_samples=N_SAMPLES, iterations=ITERATIONS, learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION, seed=seed, cem=cem, cem_interval=5,
        sigma=SIGMA, visualize=False, output_dir=OUTPUT_DIR,
    )


def _result_path(args: Namespace) -> Path:
    output_dir = (
        Path(args.output_dir) / _model_subdir(args.model) / str(args.size)
        / args.sampler / args.sampling_method
    )
    fname = (
        f"result_{args.model}{_model_params_str(args)}"
        f"_rbm{args.rbm}_nh{args.n_hidden}"
        f"_lr{args.learning_rate}_reg{args.regularization}_ns{args.n_samples}"
        f"_seed{args.seed}_iter{args.iterations}_cem{int(args.cem)}_sigma{args.sigma}"
        f".json.gz"
    )
    return output_dir / fname


def run_one(h, seed, cem):
    args = _args(h, seed, cem)
    out = _result_path(args)
    label = f"h={h:g}  seed={seed}  cem={int(cem)}"
    if out.exists():
        print(f"  [skip]  {label}")
        return

    key = jax.random.PRNGKey(seed)
    key, model_key, sampler_key = jax.random.split(key, 3)

    wave_fn = FullyConnectedRBM(N, N, model_key)
    ising = TransverseFieldIsing1D(N, h)
    sampler = ClassicalSampler(method="lsb")
    sampler._key = sampler_key

    config = dict(
        learning_rate=LEARNING_RATE, n_iterations=ITERATIONS, n_samples=N_SAMPLES,
        regularization=REGULARIZATION, seed=seed,
        use_cem=cem, cem_interval=5, lsb_sigma=SIGMA,
    )
    t0 = time.perf_counter()
    trainer = Trainer(wave_fn, ising, sampler, config, args=args)
    history = trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"  {label}  E={history['energy'][-1]:.4f}  t={elapsed:.1f}s")
    save_results(args, history, ising, rbm=wave_fn, energy_j=trainer.total_energy_j)


if __name__ == "__main__":
    for h in H_VALUES:
        for seed in SEEDS:
            for cem in (False, True):
                run_one(h, seed, cem)
    print("ALL DONE")

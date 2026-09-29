"""
rerun_cem_fixed_headline.py -- reruns the real D-Wave (+CEM) condition behind
the report's Figs. 4/5/6 and Table II with the FIXED beta_x feedback rule
(src/encoder.py), replacing the buggy-rule results. Mirrors scripts/main.py's
own training path in-process (no subprocess), same pattern as
scripts/ite/ite_run.py, so output lands in the exact same results/ layout and
filename convention that the existing plotting scripts already read.

Scope: N in {8, 16, 32, 64} x device in {pegasus, zephyr} x seed in 0..19,
same hyperparameters as the report (Sec II): h=0.5, rbm=full, lr=0.08,
reg=0.05, n_samples=200, iterations=100, cem=True (cem_interval=5 default).

The OLD buggy-rule cem1 result files for these (N, device) combos were moved
to results/archive/cem_beta_x_ema_bug/ before this script was ever run, so
the before/after comparison is preserved rather than overwritten.

Resumable: skips any (N, device, seed) whose output file already exists, so
this can be safely re-invoked if interrupted partway through -- results are
written with an atomic temp-file + rename (see helpers.save_results), so a
kill mid-run never leaves a corrupt/partial result file.

Usage:
    python scripts/exper/rerun_cem_fixed_headline.py
    python scripts/exper/rerun_cem_fixed_headline.py --sizes 8 16 --devices pegasus
"""
import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import jax
jax.config.update("jax_enable_x64", True)

from helpers import save_results, _model_params_str, _ansatz_str
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from sampler import DimodSampler
from encoder import Trainer

H_FIELD = 0.5
LR = 0.08
REG = 0.05
N_SAMPLES = 200
N_ITER = 100
N_SEEDS = 20
OUTPUT_DIR = _REPO / "results"


def expected_output_path(N, device, seed):
    args = build_args(N, device, seed)
    out_dir = OUTPUT_DIR / "tfim_1d" / str(N) / "dimod" / device
    fname = (
        f"result_1d{_model_params_str(args)}{_ansatz_str(args)}"
        f"_lr{LR}_reg{REG}_ns{N_SAMPLES}_seed{seed}_iter{N_ITER}_cem1_sigma1.0.json.gz"
    )
    return out_dir / fname


def build_args(N, device, seed):
    return argparse.Namespace(
        model="1d", size=N, h=H_FIELD, rbm="full", n_hidden=N,
        sampler="dimod", sampling_method=device,
        iterations=N_ITER, learning_rate=LR, regularization=REG,
        n_samples=N_SAMPLES, cem_interval=5,
        output_dir=str(OUTPUT_DIR), seed=seed, visualize=False, cem=True,
        n_parallel=1,
    )


def run_one(N, device, seed, sampler):
    out_path = expected_output_path(N, device, seed)
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return "skipped"

    args = build_args(N, device, seed)
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    ising = TransverseFieldIsing1D(N, H_FIELD)
    wave_fn = FullyConnectedRBM(N, N, model_key)

    trainer_config = {
        "learning_rate": LR,
        "n_iterations": N_ITER,
        "n_samples": N_SAMPLES,
        "regularization": REG,
        "save_checkpoints": True,
        "checkpoint_interval": 10,
        "use_cem": True,
        "cem_interval": 5,
        "seed": seed,
        "n_parallel": 1,
    }
    trainer = Trainer(wave_fn, ising, sampler, trainer_config, args=args)
    t0 = time.time()
    history = trainer.train()
    wall_s = time.time() - t0
    save_results(args, history, ising, wave_fn, energy_j=trainer.total_energy_j, sampler=sampler)
    device_s = float(sum(history.get("total_sampling_time_s", [])))
    print(f"  [done] N={N} device={device} seed={seed}  wall={wall_s:.1f}s  "
          f"device_time={device_s:.2f}s  final_energy={history['energy'][-1]:.4f}")
    return device_s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--devices", type=str, nargs="+", default=["pegasus", "zephyr"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    cli = p.parse_args()

    total_device_s = 0.0
    t_start = time.time()
    for N in cli.sizes:
        for device in cli.devices:
            sampler = DimodSampler(method=device)  # one embedding cache reused across all seeds
            for seed in cli.seeds:
                result = run_one(N, device, seed, sampler)
                if isinstance(result, float):
                    total_device_s += result
                print(f"    cumulative device_time so far: {total_device_s:.1f}s "
                      f"(elapsed wall: {time.time()-t_start:.0f}s)")

    print(f"\nTOTAL device time this run: {total_device_s:.2f}s")
    print(f"TOTAL wall time this run: {time.time()-t_start:.1f}s")

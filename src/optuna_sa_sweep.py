#!/usr/bin/env python3
"""
Hyperparameter search for the SA-sampler VMC using Optuna.

By default sweeps over N=16 and N=24.  Run without any arguments:

    python optuna_sa_sweep.py

Searched parameters: T_initial, T_final, n_sweeps, n_warmup,
                     learning_rate, regularization, n_hidden.

Saves one JSON per size to <output-dir>/best_<model>_N<size>_h<h>.json.
These files are read by run_fpga_best.py.

──────────────────────────────────────────────────────────────────────────────
CUSTOM SA SAMPLER — how to plug in your own implementation
──────────────────────────────────────────────────────────────────────────────
Find the function _build_sampler() below.  Replace the body with your own
sampler construction.  Your sampler must satisfy this contract:

    class MySASampler:
        def sample(self, rbm, n_samples: int, config: dict, **_) -> np.ndarray:
            # Read the annealing schedule from config:
            T_initial = config.get("T_initial", 5.0)   # start temperature
            T_final   = config.get("T_final",   0.5)   # end temperature
            n_sweeps  = config.get("n_sweeps",  20)    # cooling sweep count
            n_warmup  = config.get("n_warmup",  5)     # warmup sweeps at T_initial

            # ... run your SA here ...

            # Return spin configurations: shape (n_samples, n_visible), values ±1
            return samples  # np.ndarray, dtype float or int

Optuna will vary T_initial / T_final / n_sweeps / n_warmup across trials and
pass them through config so your sampler picks them up automatically.

If your sampler uses different parameter names, add a thin adapter in
_build_sampler() that translates config keys before forwarding.
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import math
from pathlib import Path

import jax
import numpy as np
import optuna

from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import ClassicalSampler

# ── default sweep targets ─────────────────────────────────────────────────────
DEFAULT_SIZES = [16, 24]
DEFAULT_MODEL = "1d"
DEFAULT_H     = 0.5


# ── CUSTOM SAMPLER PLUG-IN ────────────────────────────────────────────────────
def _build_sampler(T_initial, T_final, n_sweeps, n_warmup, seed, trial_number):
    """
    Build and return the SA sampler for one Optuna trial.

    TO USE YOUR OWN SAMPLER: replace the body of this function.
    The only requirement is that the returned object has a .sample() method
    matching the contract described in the module docstring above.

    Example with a custom sampler:

        from my_sa_module import MySASampler
        sampler = MySASampler(
            start_temp=T_initial,
            end_temp=T_final,
            sweeps=n_sweeps,
        )
        return sampler

    The built-in ClassicalSampler reads T_initial / T_final / n_sweeps /
    n_warmup from the config dict at sample time, so those values are
    forwarded automatically through trainer_config in _objective().
    """
    sampler = ClassicalSampler(
        method="simulated_annealing",
        T_initial=T_initial,
        T_final=T_final,
        n_sweeps=n_sweeps,
        n_warmup=n_warmup,
    )
    sampler._key = jax.random.PRNGKey(seed + trial_number + 100_000)
    return sampler
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="Optuna SA hyperparameter sweep for VMC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["1d", "2d"])
    p.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        metavar="N",
        help="Lattice sizes to sweep (one Optuna study per size)",
    )
    p.add_argument("--h", type=float, default=DEFAULT_H, help="Transverse field strength")
    p.add_argument("--n-trials", type=int, default=50, help="Optuna trials per size")
    p.add_argument("--iterations", type=int, default=30, help="VMC iterations per trial")
    p.add_argument("--n-samples", type=int, default=200, help="Samples per VMC iteration")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "optuna_results"),
        help="Directory for best-params JSON files",
    )
    p.add_argument("--study-name", default=None, help="Optuna study name prefix")
    p.add_argument(
        "--verbosity",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Optuna log level",
    )
    return p.parse_args()


def _build_ising(model, size, h):
    if model == "1d":
        return TransverseFieldIsing1D(size, h)
    if model == "2d":
        return TransverseFieldIsing2D(size, h)
    raise ValueError(f"Unknown model: {model!r}")


def _make_trainer_ns(model, size, h, n_hidden, learning_rate, regularization,
                     n_samples, iterations, seed, output_dir):
    return argparse.Namespace(
        model=model,
        size=size,
        h=h,
        rbm="full",
        sampler="custom",
        sampling_method="simulated_annealing",
        ansatz="rbm",
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        regularization=regularization,
        n_samples=n_samples,
        iterations=iterations,
        seed=seed,
        visualize=False,
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


def _objective(trial, args, size, ising, exact_energy):
    T_initial      = trial.suggest_float("T_initial",      1.0,  20.0,  log=True)
    T_final        = trial.suggest_float("T_final",        0.01,  2.0,  log=True)
    n_sweeps       = trial.suggest_int(  "n_sweeps",       5,    100)
    n_warmup       = trial.suggest_int(  "n_warmup",       1,     20)
    learning_rate  = trial.suggest_float("learning_rate",  1e-3,  0.3,  log=True)
    regularization = trial.suggest_float("regularization", 1e-6,  1e-2, log=True)
    n_hidden       = trial.suggest_int(  "n_hidden",       size,  4 * size)

    n_visible = size if args.model == "1d" else size ** 2
    key = jax.random.PRNGKey(args.seed + trial.number)
    _, model_key = jax.random.split(key)
    rbm = FullyConnectedRBM(n_visible, n_hidden, model_key)

    # ── sampler: swap your implementation here via _build_sampler() ──────────
    sampler = _build_sampler(T_initial, T_final, n_sweeps, n_warmup,
                             args.seed, trial.number)

    trainer_config = {
        "learning_rate":  learning_rate,
        "n_iterations":   args.iterations,
        "n_samples":      args.n_samples,
        "regularization": regularization,
        # These keys are forwarded to sampler.sample(config=...) each iteration.
        # Built-in ClassicalSampler reads them; your custom sampler should too.
        "T_initial": T_initial,
        "T_final":   T_final,
        "n_sweeps":  n_sweeps,
        "n_warmup":  n_warmup,
    }
    ns = _make_trainer_ns(
        args.model, size, args.h,
        n_hidden, learning_rate, regularization,
        args.n_samples, args.iterations,
        args.seed + trial.number,
        str(Path(__file__).parent.parent / "results"),
    )

    trainer = Trainer(rbm, ising, sampler, trainer_config, args=ns)
    history = trainer.train()

    tail = history["energy"][-5:]
    if not tail or not all(math.isfinite(e) for e in tail):
        return float("inf")

    # Minimize the variational energy gap above the exact ground state.
    return float(np.mean(tail)) - exact_energy


def _run_for_size(args, size, output_dir):
    print(f"\n{'='*60}")
    print(f"N={size}  model={args.model}  h={args.h}  trials={args.n_trials}")
    print(f"{'='*60}")

    ising = _build_ising(args.model, size, args.h)
    exact_energy = ising.exact_ground_energy()
    print(f"Exact ground energy: {exact_energy:.6f}")

    study_name = (args.study_name or "sa") + f"_{args.model}_N{size}_h{args.h}"
    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(
        lambda trial: _objective(trial, args, size, ising, exact_energy),
        n_trials=args.n_trials,
    )

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))[:5]
    h_str = str(args.h).replace(".", "p")
    output_path = output_dir / f"best_{args.model}_N{size}_h{h_str}.json"

    result = {
        "model": args.model,
        "size":  size,
        "h":     args.h,
        "exact_energy": exact_energy,
        "n_trials":     args.n_trials,
        "top_trials": [
            {
                "rank":              rank + 1,
                "trial_number":      t.number,
                "variational_error": t.value,
                "sa": {
                    "T_initial": t.params["T_initial"],
                    "T_final":   t.params["T_final"],
                    "n_sweeps":  t.params["n_sweeps"],
                    "n_warmup":  t.params["n_warmup"],
                },
                "vmc": {
                    "learning_rate":  t.params["learning_rate"],
                    "regularization": t.params["regularization"],
                    "n_hidden":       t.params["n_hidden"],
                    "n_samples":      args.n_samples,
                },
            }
            for rank, t in enumerate(top_trials)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Top 5 trials saved (best: #{top_trials[0].number}  error={top_trials[0].value:.6f})")
    print(f"Saved:  {output_path}")
    return output_path


def main():
    args = _parse_args()

    optuna.logging.set_verbosity(getattr(optuna.logging, args.verbosity))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for size in args.sizes:
        path = _run_for_size(args, size, output_dir)
        saved.append(path)

    print(f"\nDone. {len(saved)} file(s) written:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()

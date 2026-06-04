#!/usr/bin/env python3
"""
Hyperparameter search for the VeloxQstandard SA-sampler VMC using Optuna.

This variant drives the *VeloxQstandard* ``SimulatedAnnealing`` solver
(``../veloxQstandard``) as the VMC sampler and tunes it to draw samples from a
fixed-temperature Gibbs distribution of |Ψ(v)|², rather than running a true
annealing schedule.

To sample from a Gibbs distribution we pin the annealing setup:

    * ``num_steps = 1``            → a single schedule point (no cooling)
    * ``schedule_type = geometric``→ that single point equals ``start_temp``
    * ``stop_temp < start_temp``   → only there to satisfy the solver's strict
                                     ``T_min < T_max`` assertion; with one step
                                     it does not affect the temperature
    * ``num_sweeps_per_step``      → the only knob controlling equilibration

Optuna therefore searches over:

    * ``start_temp``          (the Gibbs temperature; free continuous range —
                              currently PINNED to ``FIXED_START_TEMP`` for the
                              quick experiment, set it to ``None`` to search)
    * ``num_sweeps_per_step`` (categorical, from a fixed ladder)
    * ``learning_rate`` / ``regularization`` / ``n_hidden``  (VMC ansatz)

The VeloxQstandard ``ComputationModel`` is built with ``scale_model`` and
``compress`` taken from the ``SCALE_ISING`` / ``COMPRESS`` module constants
(both ``False`` here ⇒ the SA solver sees the raw, unscaled Ising model).

By default sweeps over N=16 and N=24.  Run without any arguments:

    cd src
    python optuna_sa_sweep.py

Saves one JSON per size to <output-dir>/best_<model>_N<size>_h<h>.json.

──────────────────────────────────────────────────────────────────────────────
SA SAMPLER — VeloxQstandard SimulatedAnnealing
──────────────────────────────────────────────────────────────────────────────
The sampler is ``VeloxQStandardSASampler`` (see ``sampler.py``), which talks to
a long-lived Julia server (``scripts/veloxq_sa_server.jl``) over a unix socket.
One server is started per Optuna study (per size) and reused across all trials;
each trial only changes the SA config keys forwarded through ``trainer_config``:

    veloxq_num_steps    = 1                 (fixed)
    veloxq_num_sweeps   = num_sweeps_per_step
    veloxq_start_temp   = start_temp        (pinned to FIXED_START_TEMP)
    veloxq_stop_temp    = 0.5 * start_temp  (irrelevant with one step)
    veloxq_schedule     = geometric         (single point == start_temp)
    veloxq_scale_model  = SCALE_ISING       (→ VELOXQ_SCALE_MODEL)
    veloxq_compress     = COMPRESS          (→ VELOXQ_COMPRESS)

The Julia server maps ``veloxq_num_sweeps`` → ``num_sweeps_per_step`` and
``veloxq_start_temp`` / ``veloxq_stop_temp`` → ``start_temp`` / ``stop_temp`` on
the ``SimulatedAnnealing`` struct.

The ``--julia-project`` defaults to ``scripts/julia_local``, an environment that
``dev``-depends on the local ``../veloxQstandard`` checkout.
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import math
import os
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)  # match main.py/test_e2e.py: SR/CG run in float64

import numpy as np
import optuna

from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import VeloxQStandardSASampler

# ── default sweep targets ─────────────────────────────────────────────────────
DEFAULT_SIZES = [16, 24]
DEFAULT_MODEL = "1d"
DEFAULT_H     = 0.5

# Fixed SA setup for Gibbs sampling (no annealing).
NUM_STEPS = 1

# Ladder of sweep counts Optuna chooses from for num_sweeps_per_step.
NUM_SWEEPS_CHOICES = [10, 100, 500, 1000, 2000, 5000, 10000, 50000]

# ── QUICK-EXPERIMENT KNOBS ─────────────────────────────────────────────────────
# Pin the Gibbs temperature instead of letting Optuna search it.
# Set to None to restore the log-uniform search over [--temp-low, --temp-high].
FIXED_START_TEMP = 1.0

# VeloxQstandard ComputationModel build flags, forwarded to the Julia server via
# VELOXQ_SCALE_MODEL / VELOXQ_COMPRESS so the SA solver sees the raw Ising model.
SCALE_ISING = False   # ComputationModel(scale_model=...)
COMPRESS    = False   # ComputationModel(compress=...)
# ────────────────────────────────────────────────────────────────────────────────

# Default Julia environment that dev-depends on ../veloxQstandard.
DEFAULT_JULIA_PROJECT = str(Path(__file__).parent.parent / "scripts" / "julia_local")


# ── SAMPLER CONSTRUCTION ──────────────────────────────────────────────────────
def _build_sampler(args, n_samples):
    """
    Build the shared VeloxQstandard SA sampler for one Optuna study.

    The sampler launches a persistent Julia server on first use and reuses it
    across every trial in the study. Per-trial SA parameters (start_temp,
    num_sweeps_per_step) are forwarded later via the trainer config, not here.
    """
    sampler = VeloxQStandardSASampler(
        project_path=args.julia_project,
        num_rep=max(args.num_rep, n_samples),
        num_steps=NUM_STEPS,
        num_sweeps=NUM_SWEEPS_CHOICES[0],
        start_temp=1.0,
        stop_temp=0.5,
        schedule_type="geometric",
        server_ready_timeout_s=args.server_timeout,
    )
    return sampler
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="Optuna VeloxQstandard-SA Gibbs sweep for VMC",
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
        "--temp-low", type=float, default=0.05,
        help="Lower bound for the Gibbs temperature search (log-uniform)",
    )
    p.add_argument(
        "--temp-high", type=float, default=50.0,
        help="Upper bound for the Gibbs temperature search (log-uniform)",
    )
    p.add_argument(
        "--num-rep", type=int, default=1024,
        help="VeloxQ SA replica count (must be >= n-samples)",
    )
    p.add_argument(
        "--backend", default="cuda", choices=["cuda", "gpu", "cpu"],
        help="VeloxQstandard simulation backend",
    )
    p.add_argument(
        "--julia-project", default=DEFAULT_JULIA_PROJECT,
        help="Julia project that dev-depends on ../veloxQstandard",
    )
    p.add_argument(
        "--server-timeout", type=float, default=600.0,
        help="Seconds to wait for the Julia SA server to become ready",
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "optuna_results"),
        help="Directory for best-params JSON files",
    )
    p.add_argument("--study-name", default=None, help="Optuna study name prefix")
    p.add_argument(
        "--storage", default=None,
        help="Optuna storage URL (default: sqlite:///<output-dir>/optuna_sa_studies.db). "
             "Trials are persisted as they complete and the study is resumable — "
             "relaunching with the same study tops up to --n-trials.",
    )
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
        sampler="velox",
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


def _objective(trial, args, size, ising, exact_energy, sampler):
    # ── SA sampling params (the Gibbs knobs) ─────────────────────────────────
    if FIXED_START_TEMP is None:
        start_temp = trial.suggest_float("start_temp", args.temp_low,
                                         args.temp_high, log=True)
    else:
        # Pinned for the quick experiment. Still registered as a trial param
        # (degenerate single-point range) so the results JSON, which reads
        # t.params["start_temp"], stays consistent across trials.
        start_temp = trial.suggest_float("start_temp",
                                         FIXED_START_TEMP, FIXED_START_TEMP)
    num_sweeps_per_step = trial.suggest_categorical("num_sweeps_per_step",
                                                    NUM_SWEEPS_CHOICES)

    # ── VMC ansatz params ────────────────────────────────────────────────────
    learning_rate  = trial.suggest_float("learning_rate",  1e-3,  0.3,  log=True)
    regularization = trial.suggest_float("regularization", 1e-6,  1e-2, log=True)
    n_hidden       = trial.suggest_int(  "n_hidden",       size,  4 * size)

    n_visible = size if args.model == "1d" else size ** 2
    key = jax.random.PRNGKey(args.seed + trial.number)
    _, model_key = jax.random.split(key)
    rbm = FullyConnectedRBM(n_visible, n_hidden, model_key)

    trainer_config = {
        "learning_rate":  learning_rate,
        "n_iterations":   args.iterations,
        "n_samples":      args.n_samples,
        "regularization": regularization,
        # ── Gibbs SA setup forwarded to sampler.sample(config=...) ───────────
        # num_steps=1 + geometric schedule ⇒ a single, fixed-temperature pass
        # at `start_temp`, i.e. sampling the Gibbs distribution at start_temp.
        "veloxq_num_steps":  NUM_STEPS,
        "veloxq_num_sweeps": num_sweeps_per_step,
        "veloxq_start_temp": start_temp,
        # With num_steps=1 the geometric schedule has a single point equal to
        # start_temp (x**0 == 1), so stop_temp never affects the sampling
        # temperature — it only has to satisfy make_schedule's strict
        # `T_min < T_max` assertion. Hence a value just below start_temp.
        "veloxq_stop_temp":  0.5 * start_temp,
        "veloxq_schedule":   "geometric",
        "veloxq_num_rep":    max(args.num_rep, args.n_samples),
        # ── VeloxQstandard ComputationModel build flags ──────────────────────
        # These bind when the Julia server launches (first sample() call) and
        # stay fixed for the whole study. False ⇒ the SA solver sees the raw
        # Ising model (no scaling, no compression).
        "veloxq_scale_model": SCALE_ISING,
        "veloxq_compress":    COMPRESS,
        # Unbiased random subsample of the energy-sorted SA replicas (seeded for
        # reproducibility); see VeloxQStandardSASampler.sample.
        "veloxq_subsample_seed": args.seed,
        # ── pin beta_x so the Gibbs temperature is controlled solely by
        #    start_temp (no adaptive coupling rescaling confounding it) ───────
        "beta_x_init": 1.0,
        "beta_min":    1.0,
        "beta_max":    1.0,
        "use_cem":     False,
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

    mean_energy = float(np.mean(tail))
    signed_gap  = mean_energy - exact_energy
    # Store the *signed* quantities so the variational bound (⟨E⟩ ≥ E_exact) can
    # be checked directly from the study/JSON — the objective below drops the
    # sign. (Recovering these from stdout is unreliable: the per-iteration prints,
    # the Julia server's drained output, and Optuna's logging interleave racily.)
    trial.set_user_attr("mean_energy", mean_energy)
    trial.set_user_attr("signed_gap", signed_gap)

    # Minimize the DISTANCE to the exact ground energy. With a faithful |Ψ|²
    # sampler the variational principle gives ⟨E⟩ ≥ E_exact, so this is just the
    # variational gap; the abs() also guards against any residual sub-variational
    # (sampling-bias) excursion being *rewarded*. Previously this returned the
    # signed gap, which rewarded the SA sort-and-skim bias now fixed in
    # sampler.py (VeloxQStandardSASampler unbiased subsampling).
    return abs(signed_gap)


def _run_for_size(args, size, output_dir):
    print(f"\n{'='*60}")
    print(f"N={size}  model={args.model}  h={args.h}  trials={args.n_trials}")
    print(f"{'='*60}")

    ising = _build_ising(args.model, size, args.h)
    exact_energy = ising.exact_ground_energy()
    print(f"Exact ground energy: {exact_energy:.6f}")

    sampler = _build_sampler(args, args.n_samples)
    study_name = (args.study_name or "veloxsa") + f"_{args.model}_N{size}_h{args.h}"
    # Persist trials to disk so an interrupted run is not lost and can be resumed.
    storage = args.storage or f"sqlite:///{output_dir / 'optuna_sa_studies.db'}"
    study = optuna.create_study(
        direction="minimize", study_name=study_name,
        storage=storage, load_if_exists=True,
    )
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.n_trials - n_done)
    if n_done:
        print(f"Resuming {study_name!r}: {n_done} completed trial(s) found in "
              f"{storage}; running {remaining} more (target {args.n_trials}).")
    try:
        if remaining:
            study.optimize(
                lambda trial: _objective(trial, args, size, ising, exact_energy, sampler),
                n_trials=remaining,
            )
    finally:
        sampler.close()

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))[:5]
    h_str = str(args.h).replace(".", "p")
    output_path = output_dir / f"best_{args.model}_N{size}_h{h_str}.json"

    result = {
        "model": args.model,
        "size":  size,
        "h":     args.h,
        "exact_energy": exact_energy,
        "n_trials":     args.n_trials,
        "sa_setup": {
            "num_steps": NUM_STEPS,
            "schedule": "geometric, single point == start_temp → Gibbs sampling",
            "num_sweeps_choices": NUM_SWEEPS_CHOICES,
        },
        "top_trials": [
            {
                "rank":              rank + 1,
                "trial_number":      t.number,
                "variational_error": t.value,              # |⟨E⟩ − E_exact|
                "mean_energy":       t.user_attrs.get("mean_energy"),
                "signed_gap":        t.user_attrs.get("signed_gap"),  # ⟨E⟩ − E_exact
                "sa": {
                    "num_steps":           NUM_STEPS,
                    "start_temp":          t.params["start_temp"],
                    "num_sweeps_per_step": t.params["num_sweeps_per_step"],
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

    best = top_trials[0]
    best_err = best.value if best.value is not None else float("inf")
    print(f"Top 5 trials saved (best: #{best.number}  error={best_err:.6f})")
    print(f"Saved:  {output_path}")
    return output_path


def main():
    args = _parse_args()

    if args.num_rep < args.n_samples:
        raise ValueError(
            f"--num-rep ({args.num_rep}) must be >= --n-samples ({args.n_samples})."
        )

    # The Julia server reads VELOXQ_BACKEND from its environment at startup.
    os.environ["VELOXQ_BACKEND"] = args.backend

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

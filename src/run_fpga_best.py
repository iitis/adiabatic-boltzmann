#!/usr/bin/env python3
"""
Run VMC seed sweeps using best hyperparameters found by optuna_sa_sweep.py.

Loads best_<model>_N<size>_h<h>.json from optuna_results/ and re-runs the
top-K trials across --n-seeds random seeds for one or both backends:

  veloxq_sa : VeloxQStandardSASampler — the SA solver used during the Optuna search
  fpga      : FPGASampler             — VeloxQFPGA FPGA annealer

Both backends run in Gibbs mode (num_steps=1, geometric schedule) matching the
optuna sweep.  SA config is forwarded to the sampler via trainer_config so the
same parameters are used on both backends.

SA → FPGA parameter mapping:
  sa.start_temp          →  fpga_start_temp
  0.5 * sa.start_temp    →  fpga_stop_temp   (only needs T_min < T_max)
  sa.num_steps = 1       →  fpga_num_steps   (single point = Gibbs mode)
  sa.num_sweeps_per_step →  fpga_num_sweeps

By default picks up both N=16 and N=24 results.  Run without any arguments:

    cd src
    python run_fpga_best.py

Override a single file:
    python run_fpga_best.py --params ../optuna_results/best_1d_N16_h0p5.json

Run only one backend, more seeds, more iterations:
    python run_fpga_best.py --backends veloxq_sa --n-seeds 30 --iterations 200
"""

import argparse
import json
import math
import os
from pathlib import Path

import jax

from encoder import Trainer
from helpers import save_results
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import FPGASampler, VeloxQStandardSASampler

# Must match optuna_sa_sweep.py defaults so paths resolve without arguments.
DEFAULT_SIZES = [16, 24]
DEFAULT_MODEL = "1d"
DEFAULT_H = 0.5

# num_steps=1 + geometric schedule = single temperature point = Gibbs sampling.
# Hard-coded as NUM_STEPS=1 in optuna_sa_sweep.py; must match here.
_GIBBS_NUM_STEPS = 1

DEFAULT_JULIA_PROJECT = str(Path(__file__).parent.parent / "scripts" / "julia_local")


def _parse_args():
    p = argparse.ArgumentParser(
        description="FPGA/VeloxQ SA seed sweep using best optuna_sa_sweep.py params",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--params",
        default=None,
        help="Explicit JSON file from optuna_sa_sweep.py. "
        "If given, --sizes / --model / --h are ignored.",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="Run for each size, loading its params JSON automatically.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["1d", "2d"])
    p.add_argument("--h", type=float, default=DEFAULT_H)
    p.add_argument(
        "--optuna-dir",
        default=str(Path(__file__).parent.parent / "optuna_results"),
        help="Directory written by optuna_sa_sweep.py.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Use top-K trials from the JSON (ranked by variational_error).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Random seeds per trial.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="SR training iterations per run.",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["veloxq_sa", "fpga"],
        choices=["veloxq_sa", "fpga"],
        help="Backends to run.",
    )
    p.add_argument(
        "--num-rep",
        type=int,
        default=1024,
        help="Minimum replica count (clipped up to n_samples if needed).",
    )
    p.add_argument(
        "--julia-project",
        default=DEFAULT_JULIA_PROJECT,
        help="Julia project for VeloxQ SA (dev-depends on ../veloxQstandard).",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for Julia server readiness.",
    )
    p.add_argument(
        "--veloxq-backend",
        default="cuda",
        choices=["cuda", "gpu", "cpu"],
        help="VeloxQstandard simulation backend (SA only).",
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "results"),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run grid without executing.",
    )
    return p.parse_args()


def _build_ising(model, size, h):
    if model == "1d":
        return TransverseFieldIsing1D(size, h)
    if model == "2d":
        return TransverseFieldIsing2D(size, h)
    raise ValueError(f"Unknown model: {model!r}")


def _make_args_ns(
    *,
    model,
    size,
    h,
    n_hidden,
    learning_rate,
    regularization,
    n_samples,
    iterations,
    seed,
    sampler_name,
    sampling_method,
    output_dir,
):
    return argparse.Namespace(
        model=model,
        size=size,
        h=h,
        rbm="full",
        sampler=sampler_name,
        sampling_method=sampling_method,
        ansatz="rbm",
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        regularization=regularization,
        n_samples=n_samples,
        iterations=iterations,
        seed=seed,
        visualize=False,
        output_dir=str(output_dir),
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


def _run_seed(
    *,
    model,
    size,
    h,
    ising,
    trial_entry,
    seed,
    iterations,
    sampler_obj,
    sampler_name,
    sampling_method,
    num_rep,
    output_dir,
):
    sa = trial_entry["sa"]
    vmc = trial_entry["vmc"]
    n_hidden = vmc["n_hidden"]
    learning_rate = vmc["learning_rate"]
    regularization = vmc["regularization"]
    n_samples = vmc["n_samples"]
    start_temp = sa["start_temp"]
    stop_temp = 0.5 * start_temp  # T_min < T_max; irrelevant with num_steps=1
    num_sweeps = sa["num_sweeps_per_step"]

    n_visible = size if model == "1d" else size**2
    key = jax.random.PRNGKey(seed)
    _, model_key = jax.random.split(key)
    rbm = FullyConnectedRBM(n_visible, n_hidden, model_key)

    if sampler_name == "velox":
        trainer_config = {
            "learning_rate": learning_rate,
            "n_iterations": iterations,
            "n_samples": n_samples,
            "regularization": regularization,
            "veloxq_num_steps": _GIBBS_NUM_STEPS,
            "veloxq_num_sweeps": num_sweeps,
            "veloxq_start_temp": start_temp,
            "veloxq_stop_temp": stop_temp,
            "veloxq_schedule": "geometric",
            "veloxq_num_rep": max(num_rep, n_samples),
            "veloxq_scale_model": False,
            "veloxq_compress": False,
            "veloxq_subsample_seed": seed,
            "beta_x_init": 1.0,
            "beta_min": 1.0,
            "beta_max": 1.0,
            "use_cem": False,
        }
    else:  # fpga
        trainer_config = {
            "learning_rate": learning_rate,
            "n_iterations": iterations,
            "n_samples": n_samples,
            "regularization": regularization,
            "fpga_num_steps": _GIBBS_NUM_STEPS,
            "fpga_num_sweeps": num_sweeps,
            "fpga_start_temp": start_temp,
            "fpga_stop_temp": stop_temp,
            "fpga_schedule": "geometric",
            "fpga_num_rep": max(num_rep, n_samples),
            "fpga_subsample_seed": seed,
            "beta_x_init": 1.0,
            "beta_min": 1.0,
            "beta_max": 1.0,
            "use_cem": False,
        }

    args_ns = _make_args_ns(
        model=model,
        size=size,
        h=h,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        regularization=regularization,
        n_samples=n_samples,
        iterations=iterations,
        seed=seed,
        sampler_name=sampler_name,
        sampling_method=sampling_method,
        output_dir=output_dir,
    )

    trainer = Trainer(rbm, ising, sampler_obj, trainer_config, args=args_ns)
    history = trainer.train()
    save_results(args_ns, history, ising, rbm=rbm)

    energies = history["energy"]
    tail_mean = float(
        sum(energies[max(0, int(0.8 * len(energies))) :])
        / max(1, len(energies) - max(0, int(0.8 * len(energies))))
    )
    exact = float(ising.exact_ground_energy())
    rel_error = abs(tail_mean - exact) / abs(exact)
    diverged = any(not math.isfinite(e) for e in energies)
    return {"tail_mean": tail_mean, "rel_error": rel_error, "diverged": diverged}


def _run_one(params_path, args):
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(
            f"Params file not found: {params_path}\n"
            "Run optuna_sa_sweep.py first to generate it."
        )

    with open(params_path) as f:
        params = json.load(f)

    model = params["model"]
    size = params["size"]
    h = params["h"]
    top_trials = params["top_trials"][: args.top_k]

    ising = _build_ising(model, size, h)
    output_dir = Path(args.output_dir)
    num_rep = max(args.num_rep, max(e["vmc"]["n_samples"] for e in top_trials))

    print(f"\n{'=' * 60}")
    print(f"N={size}  model={model}  h={h}  exact={params['exact_energy']:.6f}")
    print(
        f"Top-{len(top_trials)} trial(s)  ×  {args.n_seeds} seeds  ×  {args.backends}"
    )
    print(f"{'=' * 60}")

    if args.dry_run:
        print(
            f"\n{'Rank':>5}  {'err':>10}  {'nh':>4}  {'lr':>8}  {'ns':>5}  {'sweeps':>8}  {'T':>5}"
        )
        print(f"  {'-' * 55}")
        for t in top_trials:
            vmc = t["vmc"]
            sa = t["sa"]
            for backend in args.backends:
                for seed in range(args.n_seeds):
                    print(
                        f"  {t['rank']:>3}  {t['variational_error']:>10.6f}"
                        f"  {vmc['n_hidden']:>4}  {vmc['learning_rate']:>8.4f}"
                        f"  {vmc['n_samples']:>5}  {sa['num_sweeps_per_step']:>8}"
                        f"  {sa['start_temp']:>5.2f}"
                        f"  {backend}  seed={seed}"
                    )
        total = len(top_trials) * len(args.backends) * args.n_seeds
        print(f"\n  Total: {total} runs")
        return

    for backend in args.backends:
        print(f"\n--- Backend: {backend} ---")

        if backend == "veloxq_sa":
            os.environ["VELOXQ_BACKEND"] = args.veloxq_backend
            # All trials share one Julia server; per-trial SA params are forwarded
            # via trainer_config, not the constructor.
            sampler_obj = VeloxQStandardSASampler(
                project_path=args.julia_project,
                num_rep=num_rep,
                num_steps=_GIBBS_NUM_STEPS,
                num_sweeps=top_trials[0]["sa"]["num_sweeps_per_step"],
                start_temp=top_trials[0]["sa"]["start_temp"],
                stop_temp=0.5 * top_trials[0]["sa"]["start_temp"],
                schedule_type="geometric",
                server_ready_timeout_s=args.server_timeout,
            )
            sampler_name = "velox"
            sampling_method = "simulated_annealing"
        else:
            sampler_obj = FPGASampler(num_rep=num_rep)
            sampler_name = "fpga"
            sampling_method = "fpga"

        try:
            for trial_entry in top_trials:
                rank = trial_entry["rank"]
                vmc = trial_entry["vmc"]
                sa = trial_entry["sa"]
                print(
                    f"\n  Rank {rank}"
                    f"  err={trial_entry['variational_error']:.6f}"
                    f"  nh={vmc['n_hidden']}"
                    f"  lr={vmc['learning_rate']:.4f}"
                    f"  ns={vmc['n_samples']}"
                    f"  sweeps={sa['num_sweeps_per_step']}"
                    f"  T={sa['start_temp']:.2f}"
                )

                results = []
                for seed in range(args.n_seeds):
                    print(
                        f"    seed {seed + 1}/{args.n_seeds} ...", end="\r", flush=True
                    )
                    try:
                        m = _run_seed(
                            model=model,
                            size=size,
                            h=h,
                            ising=ising,
                            trial_entry=trial_entry,
                            seed=seed,
                            iterations=args.iterations,
                            sampler_obj=sampler_obj,
                            sampler_name=sampler_name,
                            sampling_method=sampling_method,
                            num_rep=num_rep,
                            output_dir=output_dir,
                        )
                        results.append(m)
                    except Exception as exc:
                        print(f"\n    seed {seed} FAILED: {exc}")

                n_ok = sum(1 for m in results if not m["diverged"])
                errors = [
                    m["rel_error"]
                    for m in results
                    if not m["diverged"] and math.isfinite(m["rel_error"])
                ]
                mean_err = sum(errors) / len(errors) if errors else float("nan")
                print(
                    f"    {n_ok}/{len(results)} converged"
                    f"  mean_rel_err={mean_err:.6f}"
                    f"                    "
                )
        finally:
            if hasattr(sampler_obj, "close"):
                sampler_obj.close()


def main():
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.params is not None:
        _run_one(args.params, args)
        return

    optuna_dir = Path(args.optuna_dir)
    h_str = str(args.h).replace(".", "p")
    for size in args.sizes:
        params_path = optuna_dir / f"best_{args.model}_N{size}_h{h_str}.json"
        _run_one(str(params_path), args)


if __name__ == "__main__":
    main()

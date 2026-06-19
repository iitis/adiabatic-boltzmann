#!/usr/bin/env python3
"""
Run VMC seed sweeps using best hyperparameters found by hparam_optuna.py.

Two modes:

  --generalize (default)
      Fixed VMC params (lr=0.13, reg=0.005, ns=200, nh=round(3*N)) across TFIM
      sizes and J1J2 Heisenberg.  Does not require any prior hparam search.

  --no-generalize
      Loads the top-K trials from results/hparam_search/tfim_1d/ (written by
      scripts/hparam_optuna.py --hamiltonian tfim_1d) and re-runs them across
      --n-seeds random seeds for one or both backends:

        veloxq_sa : VeloxQStandardSASampler
        fpga      : FPGASampler

      SA → FPGA parameter mapping:
        T_initial              →  fpga_start_temp
        0.5 * T_initial        →  fpga_stop_temp
        num_steps=1 (Gibbs)    →  fpga_num_steps
        --num-sweeps[0]        →  fpga_num_sweeps

Usage:

    cd src
    python run_fpga_best.py                          # generalization sweep
    python run_fpga_best.py --no-generalize          # best hparam trials for N=16,24
    python run_fpga_best.py --backends veloxq_sa --n-seeds 30 --iterations 200
    python run_fpga_best.py --generalize --tfim-sizes 8 12 16 24 32 \\
        --j1j2-sizes 8 12 16 --j2 0.1 0.3 0.5 --backends fpga
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import jax

from encoder import Trainer
from helpers import (
    _ansatz_str,
    _model_params_str,
    _model_subdir,
    save_results,
)
from ising import J1J2HeisenbergXXZ1D, TransverseFieldIsing1D, TransverseFieldIsing2D
from model import FullyConnectedRBM
from sampler import FPGASampler, VeloxQStandardSASampler

DEFAULT_SIZES = [16, 24]
DEFAULT_MODEL = "1d"
DEFAULT_H = 0.5

# Fixed hyperparameters for the generalization sweep (from TFIM Optuna optima).
_GEN_LR = 0.13
_GEN_REG = 0.005
_GEN_N_SAMPLES = 200
_GEN_START_TEMP = 1.0
_GEN_NH_ALPHA = 3.0   # n_hidden = round(NH_ALPHA * N)


def _result_exists(args_ns) -> bool:
    """Return True if save_results would find an existing file for this run."""
    output_dir = Path(
        f"{args_ns.output_dir}/{_model_subdir(args_ns.model)}"
        f"/{args_ns.size}/{args_ns.sampler}/{args_ns.sampling_method}"
    )
    use_cem = getattr(args_ns, "cem", False)
    fname = (
        f"result"
        f"_{args_ns.model}"
        f"{_model_params_str(args_ns)}"
        f"{_ansatz_str(args_ns)}"
        f"_lr{args_ns.learning_rate}"
        f"_reg{args_ns.regularization}"
        f"_ns{args_ns.n_samples}"
        f"_seed{args_ns.seed}"
        f"_iter{args_ns.iterations}"
        f"_cem{int(use_cem)}"
        f"_sigma{float(getattr(args_ns, 'sigma', 1.0))}"
        f".json"
    )
    return (output_dir / fname).exists()

# num_steps=1 + geometric schedule = single temperature point = Gibbs sampling.
_GIBBS_NUM_STEPS = 1

DEFAULT_JULIA_PROJECT = str(Path(__file__).parent.parent / "scripts" / "julia_local")


def _parse_args():
    p = argparse.ArgumentParser(
        description="FPGA/VeloxQ SA seed sweep using best params from hparam_optuna.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="Run for each size (--no-generalize mode only).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["1d", "2d"])
    p.add_argument("--h", type=float, default=DEFAULT_H)
    p.add_argument(
        "--hparam-dir",
        default=str(Path(__file__).parent.parent / "results" / "hparam_search"),
        help="Directory written by scripts/hparam_optuna.py (contains tfim_1d/ subdirectory).",
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
        default=["fpga", "veloxq_sa"],
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
    # ── Generalization sweep ──────────────────────────────────────────────
    p.add_argument(
        "--generalize",
        action="store_true",
        default=True,
        help="Run the generalization sweep (default). "
             "Uses fixed VMC params (lr=0.13, reg=0.005, ns=200, nh=round(3*N)).",
    )
    p.add_argument(
        "--no-generalize",
        dest="generalize",
        action="store_false",
        help="Load best trials from results/hparam_search/tfim_1d/ instead of using fixed params.",
    )
    p.add_argument(
        "--tfim-sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16, 24, 32],
        metavar="N",
        help="TFIM system sizes for --generalize.",
    )
    p.add_argument(
        "--j1j2-sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16],
        metavar="N",
        help="J1J2 Heisenberg system sizes for --generalize.",
    )
    p.add_argument(
        "--j2",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        metavar="J2",
        help="J2/J1 values for J1J2 Heisenberg in --generalize.",
    )
    p.add_argument(
        "--num-sweeps",
        type=int,
        nargs="+",
        default=[100, 2000],
        metavar="S",
        help="FPGA sweeps per step to test in --generalize. Multiple values sweep the axis.",
    )
    return p.parse_args()


def _build_ising(model, size, h, J1=1.0, J2=0.0, delta=1.0):
    if model == "1d":
        return TransverseFieldIsing1D(size, h)
    if model == "2d":
        return TransverseFieldIsing2D(size, h)
    if model == "heisenberg_j1j2_1d":
        return J1J2HeisenbergXXZ1D(size, J1=J1, J2=J2, delta=delta)
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
    J1=1.0,
    J2=0.0,
    delta=1.0,
):
    return argparse.Namespace(
        model=model,
        size=size,
        h=h,
        J1=J1,
        J2=J2,
        J=J1,
        delta=delta,
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
    J1=1.0,
    J2=0.0,
    delta=1.0,
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

    # 2D models use size² spins; 1D models (TFIM and J1J2 Heisenberg) use size.
    n_visible = size**2 if model == "2d" else size
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
        J1=J1,
        J2=J2,
        delta=delta,
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


def _load_best_trials(hparam_dir, model, size, h, top_k, num_sweeps_per_step):
    """Return top-K trial_entry dicts from index.jsonl files matching (model, size, h).

    Reads from hparam_dir/tfim_1d/**/index.jsonl, filters to simulated_annealing
    entries for the given system size and transverse field, and maps the recorded
    params to the trial_entry format expected by _run_seed.
    """
    hamiltonian = "tfim_1d" if model == "1d" else f"tfim_{model}"
    search_root = Path(hparam_dir) / hamiltonian
    if not search_root.exists():
        raise FileNotFoundError(
            f"No hparam data for {hamiltonian!r}: {search_root}\n"
            "Run: python scripts/hparam_optuna.py --hamiltonian tfim_1d"
        )

    records = []
    for index_file in search_root.rglob("index.jsonl"):
        with open(index_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("N") != size:
                    continue
                if abs(rec.get("phys_params", {}).get("h", float("nan")) - h) > 1e-9:
                    continue
                if rec.get("params", {}).get("sampling_method") != "simulated_annealing":
                    continue
                if not math.isfinite(rec.get("rel_error", float("nan"))):
                    continue
                records.append(rec)

    if not records:
        raise ValueError(
            f"No simulated_annealing trials found for N={size}, h={h} in {search_root}"
        )

    records.sort(key=lambda r: r.get("rel_error", float("inf")))
    return [
        {
            "rank": rank,
            "variational_error": rec["rel_error"],
            "sa": {
                "num_steps": _GIBBS_NUM_STEPS,
                "start_temp": rec["params"].get("T_initial", 1.0),
                "num_sweeps_per_step": num_sweeps_per_step,
            },
            "vmc": {
                "n_hidden": rec["n_hidden"],
                "learning_rate": rec["params"]["learning_rate"],
                "regularization": rec["params"]["regularization"],
                "n_samples": rec["params"]["n_samples"],
            },
        }
        for rank, rec in enumerate(records[:top_k], start=1)
    ]


def _run_one(hparam_dir, model, size, h, args):
    num_sweeps_per_step = args.num_sweeps[0] if args.num_sweeps else 2000
    top_trials = _load_best_trials(
        hparam_dir, model, size, h, args.top_k, num_sweeps_per_step
    )

    ising = _build_ising(model, size, h)
    output_dir = Path(args.output_dir)
    num_rep = max(args.num_rep, max(e["vmc"]["n_samples"] for e in top_trials))

    try:
        exact_str = f"{float(ising.exact_ground_energy()):.6f}"
    except NotImplementedError:
        exact_str = "N/A"

    print(f"\n{'=' * 60}")
    print(f"N={size}  model={model}  h={h}  exact={exact_str}")
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
            sampler_obj = FPGASampler(num_rep=num_rep, transport="pcie")
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


def _run_generalize(args):
    """Generalization sweep: fixed VMC params across TFIM sizes + J1J2 Heisenberg."""
    configs = []
    for N in args.tfim_sizes:
        configs.append({"model": "1d", "size": N, "h": args.h, "J1": 1.0, "J2": 0.0, "delta": 1.0,
                        "label": f"TFIM N={N} h={args.h}"})
    for N in args.j1j2_sizes:
        for J2 in args.j2:
            configs.append({"model": "heisenberg_j1j2_1d", "size": N, "h": 0.0,
                            "J1": 1.0, "J2": J2, "delta": 1.0,
                            "label": f"J1J2 N={N} J2={J2}"})

    print(f"\n{'=' * 60}")
    print(f"Generalization sweep — {len(configs)} configs  ×  {args.n_seeds} seeds  ×  {args.backends}")
    print(f"lr={_GEN_LR}  reg={_GEN_REG}  ns={_GEN_N_SAMPLES}  T={_GEN_START_TEMP}  nh=round({_GEN_NH_ALPHA}*N)")
    print(f"{'=' * 60}")

    if args.dry_run:
        print(f"\n{'Label':<35}  {'nh':>4}  {'sweeps':>8}  backend  seeds")
        print(f"  {'-' * 63}")
        for cfg in configs:
            nh = max(1, round(_GEN_NH_ALPHA * cfg["size"]))
            for num_sweeps in args.num_sweeps:
                for backend in args.backends:
                    print(f"  {cfg['label']:<33}  {nh:>4}  {num_sweeps:>8}  {backend:<10}  {args.n_seeds}")
        total = len(configs) * len(args.num_sweeps) * len(args.backends) * args.n_seeds
        print(f"\n  Total: {total} runs")
        return

    output_dir = Path(args.output_dir)
    num_rep = max(args.num_rep, _GEN_N_SAMPLES)

    for backend in args.backends:
        print(f"\n--- Backend: {backend} ---")

        if backend == "veloxq_sa":
            os.environ["VELOXQ_BACKEND"] = args.veloxq_backend
            sampler_obj = VeloxQStandardSASampler(
                project_path=args.julia_project,
                num_rep=num_rep,
                num_steps=_GIBBS_NUM_STEPS,
                num_sweeps=configs[0]["size"],  # will be overridden via trainer_config
                start_temp=_GEN_START_TEMP,
                stop_temp=0.5 * _GEN_START_TEMP,
                schedule_type="geometric",
                server_ready_timeout_s=args.server_timeout,
            )
            sampler_name = "velox"
            sampling_method = "simulated_annealing"
        else:
            sampler_obj = FPGASampler(num_rep=num_rep, transport="pcie")
            sampler_name = "fpga"
            sampling_method = "fpga"

        try:
            for num_sweeps in args.num_sweeps:
                # Per-sweep subtree: save_results' filename does not include
                # num_sweeps, so we must namespace by directory to avoid
                # overwriting num_sweeps=100 results with num_sweeps=2000.
                sweep_output_dir = output_dir / f"sweeps{num_sweeps}"
                for cfg in configs:
                    N = cfg["size"]
                    n_hidden = max(1, round(_GEN_NH_ALPHA * N))
                    ising = _build_ising(cfg["model"], N, cfg["h"],
                                         J1=cfg["J1"], J2=cfg["J2"], delta=cfg["delta"])

                    try:
                        exact_e = ising.exact_ground_energy()
                        exact_str = f"{exact_e:.6f}"
                    except NotImplementedError:
                        exact_str = "N/A"

                    print(f"\n  {cfg['label']}  nh={n_hidden}  sweeps={num_sweeps}  exact={exact_str}")

                    trial_entry = {
                        "sa": {
                            "num_steps": _GIBBS_NUM_STEPS,
                            "start_temp": _GEN_START_TEMP,
                            "num_sweeps_per_step": num_sweeps,
                        },
                        "vmc": {
                            "n_hidden": n_hidden,
                            "learning_rate": _GEN_LR,
                            "regularization": _GEN_REG,
                            "n_samples": _GEN_N_SAMPLES,
                        },
                    }

                    results = []
                    skipped = 0
                    for seed in range(args.n_seeds):
                        probe = _make_args_ns(
                            model=cfg["model"], size=N, h=cfg["h"],
                            n_hidden=n_hidden,
                            learning_rate=_GEN_LR, regularization=_GEN_REG,
                            n_samples=_GEN_N_SAMPLES, iterations=args.iterations,
                            seed=seed, sampler_name=sampler_name,
                            sampling_method=sampling_method, output_dir=sweep_output_dir,
                            J1=cfg["J1"], J2=cfg["J2"], delta=cfg["delta"],
                        )
                        if _result_exists(probe):
                            skipped += 1
                            continue
                        print(f"    seed {seed + 1}/{args.n_seeds} ...", end="\r", flush=True)
                        try:
                            m = _run_seed(
                                model=cfg["model"],
                                size=N,
                                h=cfg["h"],
                                ising=ising,
                                trial_entry=trial_entry,
                                seed=seed,
                                iterations=args.iterations,
                                sampler_obj=sampler_obj,
                                sampler_name=sampler_name,
                                sampling_method=sampling_method,
                                num_rep=num_rep,
                                output_dir=sweep_output_dir,
                                J1=cfg["J1"],
                                J2=cfg["J2"],
                                delta=cfg["delta"],
                            )
                            results.append(m)
                        except Exception as exc:
                            print(f"\n    seed {seed} FAILED: {exc}")

                    n_ok = sum(1 for m in results if not m["diverged"])
                    errors = [m["rel_error"] for m in results
                              if not m["diverged"] and math.isfinite(m["rel_error"])]
                    mean_err = sum(errors) / len(errors) if errors else float("nan")
                    skip_str = f"  ({skipped} skipped)" if skipped else ""
                    print(
                        f"    {n_ok}/{len(results)} converged"
                        f"  mean_rel_err={mean_err:.6f}"
                        f"{skip_str}"
                        f"                    "
                    )
        finally:
            if hasattr(sampler_obj, "close"):
                sampler_obj.close()


def main():
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.generalize:
        _run_generalize(args)
        return

    for size in args.sizes:
        _run_one(args.hparam_dir, args.model, size, args.h, args)


if __name__ == "__main__":
    main()

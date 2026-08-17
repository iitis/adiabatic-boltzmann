import gzip
import json
import subprocess
from pathlib import Path
import numpy as np
import jax
import pickle


def _git_sha() -> str | None:
    """Short SHA of HEAD, so a result file can be traced to the code that produced it.

    Best-effort only (not a shared resource/budget check) -- on failure this
    logs and returns None rather than raising, since a missing SHA shouldn't
    block saving a training result.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except Exception as e:
        print(f"  [save_results] could not resolve git SHA: {e}")
        return None


def _sampler_config(sampler) -> dict | None:
    """Snapshot the mixing hyperparameters that decide a ClassicalSampler run's
    output but aren't otherwise recoverable from `args`/the filename -- without
    this, two files with identical (h, lr, reg, ns, seed, iterations) can have
    been produced by different n_warmup/n_sweeps/T_initial/T_final and there
    would be no way to tell after the fact (see results/tfim_1d git history:
    commit a53d81da7 added simulated_annealing wholesale, and different sizes
    of ClassicalSampler.method sweep have been going through
    scripts/exper/mcmc_matched_sweep.py's --sa-sweeps/--n-warmup un-recorded).
    """
    if sampler is None or not hasattr(sampler, "method"):
        return None
    return {
        "method": sampler.method,
        "n_warmup": getattr(sampler, "n_warmup", None),
        "n_sweeps": getattr(sampler, "n_sweeps", None),
        "T_initial": getattr(sampler, "T_initial", None),
        "T_final": getattr(sampler, "T_final", None),
    }

# Unknown model names pass through unchanged.
_MODEL_SUBDIR: dict[str, str] = {
    "1d": "tfim_1d",
    "2d": "tfim_2d",
    "heisenberg_j1j2_1d": "heisenberg_j1j2_1d",
}


def _model_subdir(model: str) -> str:
    return _MODEL_SUBDIR.get(model, model)


def load_result_json(path) -> dict:
    """Load a result file that is either .json or .json.gz."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _model_params_str(args) -> str:
    """Return the model-parameter component for result filenames."""
    if args.model == "heisenberg_j1j2_1d":
        J1 = getattr(args, "J1", 1.0)
        J2 = getattr(args, "J2", 0.3)
        delta = getattr(args, "delta", 1.0)
        return f"_J1{J1}_J2{J2}_delta{delta}"
    return f"_h{args.h}"


def save_rbm_checkpoint(rbm, args, iteration):
    """
    Save RBM parameters (weights, biases) to a checkpoint file.

    Called from encoder.Trainer when config["save_checkpoints"] is set.
    """
    checkpoint_dir = Path(
        f"{args.output_dir.replace('results', 'checkpoints')}"
        f"/{_model_subdir(args.model)}/{args.size}/{args.sampler}/{args.sampling_method}/{args.rbm}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "config": vars(args),
        "rbm_state": {
            "a": np.array(rbm.a).tolist(),
            "b": np.array(rbm.b).tolist(),
            "W": np.array(rbm.W).tolist(),
            "n_visible": rbm.n_visible,
            "n_hidden": rbm.n_hidden,
        },
    }

    _n_parallel = getattr(args, "n_parallel", 1)
    checkpoint_file = checkpoint_dir / (
        f"checkpoint"
        f"_{args.model}"
        f"{_model_params_str(args)}"
        f"_rbm{args.rbm}"
        f"_nh{args.n_hidden}"
        f"_lr{args.learning_rate}"
        f"_iter{iteration:04d}"
        + (f"_np{_n_parallel}" if _n_parallel and _n_parallel != 1 else "")
        + f".pkl"
    )

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)

    return checkpoint_file


def _safe_exact_energy(ising):
    try:
        return ising.exact_ground_energy()
    except NotImplementedError:
        return None


def _safe_rel_error(final_energy, ising):
    try:
        exact = ising.exact_ground_energy()
        return abs(final_energy - exact)
    except NotImplementedError:
        return None


def _ansatz_str(args) -> str:
    """Return the ansatz component of the result filename."""
    rbm     = getattr(args, "rbm", "full")
    n_hidden = getattr(args, "n_hidden", None)
    return f"_rbm{rbm}_nh{n_hidden}"


def save_results(args, history, ising, rbm=None, energy_j=None, num_sweeps=None, sampler=None):
    """
    num_sweeps: optional SA/annealing sweep count. Not derivable from `args`
    (main.py's Metropolis path has no such concept), so callers that have it
    (run_fpga_best.py's fpga/veloxq_sa modes) pass it explicitly. When given,
    it's appended to the filename and stored in the result JSON; when None,
    filenames are unchanged from before this parameter existed. This matters
    because num_sweeps otherwise isn't encoded anywhere the filename/resume
    logic can see it: rerunning with a different num_sweeps but identical
    lr/reg/n_samples/seed would otherwise collide with a prior run's file.

    sampler: optional Sampler instance. When it's a DimodSampler that hit a
    real/trivial embedding this run, its `last_embedding_info` (chain stats
    for the embedding actually used on the solver) is stored in the result
    JSON as "embedding_info" -- otherwise this is unverifiable after the
    fact since the embedding search is only logged to stdout.
    """
    output_dir = Path(
        f"{args.output_dir}/{_model_subdir(args.model)}/{args.size}/{args.sampler}/{args.sampling_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _jax_devices = jax.devices()
    _jax_backend = _jax_devices[0].platform if _jax_devices else "cpu"
    _jax_device_strs = [str(d) for d in _jax_devices]

    use_cem = getattr(args, "cem", False)
    results = {
        "config": vars(args),
        "history": {
            k: [float(v) if v is not None else None for v in vals]
            for k, vals in history.items()
        },
        "final_energy": history["energy"][-1],
        "exact_energy": _safe_exact_energy(ising),
        "error": _safe_rel_error(history["energy"][-1], ising),
        "sparsity": float(rbm.sparsity()) if rbm is not None else None,
        "sampling_time_s": float(sum(history.get("sampling_time_s", []))),
        "cem_time_s": float(sum(history.get("cem_time_s", []))),
        "total_sampling_time_s": float(sum(history.get("total_sampling_time_s", []))),
        # solver (sampling) GPU energy only — excludes SR/CG/gradient work, see energy.py
        "gpu_energy_wh": (energy_j / 3600.0) if energy_j is not None else None,
        "jax_devices": {
            "backend": _jax_backend,
            "devices": _jax_device_strs,
        },
        "final_ess": history["ess"][-1] if history.get("ess") else None,
        "mean_ess": float(np.mean(history["ess"])) if history.get("ess") else None,
        "final_kl_exact": history["kl_exact"][-1] if history.get("kl_exact") else None,
        "final_n_unique_ratio": history["n_unique_ratio"][-1]
        if history.get("n_unique_ratio")
        else None,
        "mean_n_unique_ratio": float(
            np.mean([x for x in history["n_unique_ratio"] if x is not None])
        )
        if history.get("n_unique_ratio")
        else None,
        "mean_mh_acceptance_rate": float(
            np.mean([x for x in history["mh_acceptance_rate"] if x is not None])
        )
        if history.get("mh_acceptance_rate") and any(
            x is not None for x in history["mh_acceptance_rate"]
        )
        else None,
        "n_parallel": getattr(args, "n_parallel", None),
        "num_sweeps": num_sweeps,
        "embedding_info": getattr(sampler, "last_embedding_info", None),
        "sampler_config": _sampler_config(sampler),
        "git_sha": _git_sha(),
    }

    # Filename encodes every axis that varies in the sweep
    _n_parallel = getattr(args, "n_parallel", 1)
    output_file = output_dir / (
        f"result"
        f"_{args.model}"
        f"{_model_params_str(args)}"
        f"{_ansatz_str(args)}"
        f"_lr{args.learning_rate}"
        f"_reg{args.regularization}"
        f"_ns{args.n_samples}"
        f"_seed{args.seed}"
        f"_iter{args.iterations}"
        f"_cem{int(use_cem)}"
        f"_sigma{float(getattr(args, 'sigma', 1.0))}"
        + (f"_np{_n_parallel}" if _n_parallel and _n_parallel != 1 else "")
        + (f"_sw{num_sweeps}" if num_sweeps is not None else "")
        + f".json.gz"
    )

    # atomic write via temp file + rename
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with gzip.open(tmp_file, "wt") as f:
        json.dump(results, f, indent=2)
    tmp_file.replace(output_file)

    def _fmt(v):
        return f"{v:.6f}" if v is not None else "N/A"

    print(f"Saved  → {output_file}")
    print(f"  Final energy : {_fmt(results['final_energy'])}")
    print(f"  Exact energy : {_fmt(results['exact_energy'])}")
    print(f"  Error        : {_fmt(results['error'])}")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 7. Plot if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history["energy"])
            if results["exact_energy"] is not None:
                plt.axhline(
                    results["exact_energy"], color="r", linestyle="--", label="Exact"
                )
            plt.xlabel("Iteration")
            plt.ylabel("Energy")
            plt.title("Convergence")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history["error"])
            plt.xlabel("Iteration")
            plt.ylabel("Standard Error")
            plt.title("Energy Variance")

            plt.tight_layout()
            plot_file = plot_dir / f"plot_{args.model}{_model_params_str(args)}_rbm{args.rbm}.png"
            plt.savefig(plot_file, dpi=150)
            plt.show()
            print(f"Plot saved to {plot_file}")

        except ImportError:
            print("Matplotlib not available, skipping visualization")


def save_dwave_samples(V: np.ndarray, args, iteration: int, sampleset=None) -> Path:
    """
    Save raw D-Wave samples for one training iteration.

    Stored as gzip-compressed pickle under:
        dwave_samples/{n_hidden}/{sampler}/{method}/
            samples_{model}{model_params}_rbm{rbm}_nh{n_hidden}_lr{lr}
            _reg{reg}_ns{ns}_seed{seed}_iter{IIII}.pkl.gz

    Content always includes:
        "v"         : ndarray(ns, N)  visible-unit spin configs ±1
        "iteration" : int
        "config"    : dict

    When sampleset is provided (dimod SampleSet from a QPU call):
        "energies"         : ndarray(n_reads,)  Ising energies per sample
        "num_occurrences"  : ndarray(n_reads,)  read counts (usually all 1 for raw mode)
        "timing"           : dict from sampleset.info["timing"]
    """
    out_dir = Path(
        f"dwave_samples/{args.n_hidden}/{args.sampler}/{args.sampling_method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _n_parallel = getattr(args, "n_parallel", 1)
    fname = (
        f"samples_{args.model}"
        f"{_model_params_str(args)}"
        f"_rbm{args.rbm}"
        f"_nh{args.n_hidden}"
        f"_lr{args.learning_rate}"
        f"_reg{args.regularization}"
        f"_ns{args.n_samples}"
        f"_seed{args.seed}"
        f"_iter{iteration:04d}"
        + (f"_np{_n_parallel}" if _n_parallel and _n_parallel != 1 else "")
        + f".pkl.gz"
    )
    path = out_dir / fname

    payload = {"v": V, "iteration": iteration, "config": vars(args)}
    if sampleset is not None:
        payload["energies"]        = sampleset.record.energy.copy()
        payload["num_occurrences"] = sampleset.record.num_occurrences.copy()
        payload["timing"]          = dict(sampleset.info.get("timing", {}))

    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=5)
    return path


def read_qpu_time_ms(time_path: Path = Path("time.json"), key: str = "time_ms") -> float:
    """
    Return accumulated QPU access time in milliseconds from time.json.

    Returns 0.0 if the file does not exist (valid initial state).
    Raises OSError / json.JSONDecodeError / KeyError if the file exists but
    cannot be read or parsed — a silent 0 fallback would make budget checks
    always pass and silently burn QPU time.
    """
    if not time_path.exists():
        return 0.0
    with time_path.open("r") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"'{key}' missing from {time_path}")
    return float(data[key])


def get_solver_name(architecture="pegasus"):
    if architecture == "pegasus":
        return "Advantage_system6"
    elif architecture == "zephyr":
        return "Advantage2_system1"

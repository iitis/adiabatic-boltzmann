import fcntl
import gzip
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pickle

# Maps CLI --model value to its results subdirectory.
# New models should be added here; unknown names pass through unchanged.
_MODEL_SUBDIR: dict[str, str] = {
    "1d": "tfim_1d",
    "2d": "tfim_2d",
    "lr1d": "lr_tfim_1d",
    "j1j2_1d": "j1j2_1d",
    "heisenberg_xy_1d": "heisenberg_xy_1d",
    "heisenberg_xxz_2d": "heisenberg_xxz_2d",
}


def _model_subdir(model: str) -> str:
    return _MODEL_SUBDIR.get(model, model)


def _model_params_str(args) -> str:
    """Return the model-parameter component for result filenames."""
    if args.model == "heisenberg_xxz_1d":
        J = getattr(args, "J", 1.0)
        delta = getattr(args, "delta", 1.0)
        return f"_J{J}_delta{delta}"
    if args.model == "lr1d":
        alpha = getattr(args, "alpha", 2.0)
        return f"_h{args.h}_alpha{alpha}"
    if args.model == "j1j2_1d":
        J1 = getattr(args, "J1", 1.0)
        J2 = getattr(args, "J2", 0.5)
        return f"_J1{J1}_J2{J2}_h{args.h}"
    if args.model == "heisenberg_xy_1d":
        J = getattr(args, "J", 1.0)
        return f"_J{J}"
    if args.model == "heisenberg_xxz_2d":
        J = getattr(args, "J", 1.0)
        delta = getattr(args, "delta", 1.0)
        return f"_J{J}_delta{delta}"
    return f"_h{args.h}"


def save_rbm_checkpoint(rbm, args, iteration):
    """
    Save RBM parameters (weights, biases) to a checkpoint file.

    Args:
        rbm: RBM model instance
        args: argparse Namespace with training config
        iteration: current iteration number

    Returns:
        Path to saved checkpoint
    """
    # Directory structure: checkpoints/{model}/{size}/{sampler}/{method}/{rbm}/
    checkpoint_dir = Path(
        f"{args.output_dir.replace('results', 'checkpoints')}"
        f"/{_model_subdir(args.model)}/{args.size}/{args.sampler}/{args.sampling_method}/{args.rbm}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint data
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

    checkpoint_file = checkpoint_dir / (
        f"checkpoint"
        f"_{args.model}"
        f"{_model_params_str(args)}"
        f"_rbm{args.rbm}"
        f"_nh{args.n_hidden}"
        f"_lr{args.learning_rate}"
        f"_iter{iteration:04d}"
        f".pkl"
    )

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)

    return checkpoint_file


def load_rbm_checkpoint(checkpoint_path):
    """
    Load RBM parameters from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (rbm_state_dict, config, iteration)
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint["rbm_state"], checkpoint["config"], checkpoint["iteration"]


def restore_rbm_from_checkpoint(rbm, checkpoint_path):
    """
    Restore RBM parameters from checkpoint into an RBM instance.

    Args:
        rbm: RBM model instance to update
        checkpoint_path: Path to checkpoint file

    Returns:
        iteration number from checkpoint
    """
    rbm_state, config, iteration = load_rbm_checkpoint(checkpoint_path)

    rbm.a = jnp.array(rbm_state["a"])
    rbm.b = jnp.array(rbm_state["b"])
    rbm.W = jnp.array(rbm_state["W"])

    print(f"Restored RBM from checkpoint: {checkpoint_path}")
    print(f"  Starting from iteration {iteration}")

    return iteration


def find_latest_checkpoint(args) -> Path | None:
    """
    Return the highest-iteration checkpoint file that matches args, or None.

    Scans the checkpoint directory for files whose name prefix matches the
    run configuration (model, h, rbm type, n_hidden, lr).  The iteration
    number is parsed from the filename so the result is correct even if the
    filesystem returns files in an arbitrary order.
    """
    checkpoint_dir = Path(
        f"{args.output_dir.replace('results', 'checkpoints')}"
        f"/{_model_subdir(args.model)}/{args.size}/{args.sampler}/{args.sampling_method}/{args.rbm}"
    )
    if not checkpoint_dir.exists():
        return None

    prefix = (
        f"checkpoint"
        f"_{args.model}"
        f"{_model_params_str(args)}"
        f"_rbm{args.rbm}"
        f"_nh{args.n_hidden}"
        f"_lr{args.learning_rate}"
        f"_iter"
    )

    best: Path | None = None
    best_iter = -1
    for p in checkpoint_dir.glob(f"{prefix}*.pkl"):
        stem = p.stem
        marker = "_iter"
        idx = stem.rfind(marker)
        if idx == -1:
            continue
        try:
            it = int(stem[idx + len(marker):])
        except ValueError:
            continue
        if it > best_iter:
            best_iter = it
            best = p
    return best


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


def save_results(args, history, ising, rbm=None):
    # Directory structure: results/{model}/{size}/{sampler}/{method}/
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
    }

    # Filename encodes every axis that varies in the sweep
    output_file = output_dir / (
        f"result"
        f"_{args.model}"
        f"{_model_params_str(args)}"
        f"_rbm{args.rbm}"
        f"_nh{args.n_hidden}"
        f"_lr{args.learning_rate}"
        f"_reg{args.regularization}"
        f"_ns{args.n_samples}"
        f"_seed{args.seed}"
        f"_iter{args.iterations}"
        f"_cem{int(use_cem)}"
        f"_sigma{float(args.sigma)}"
        f".json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(v):
        return f"{v:.6f}" if v is not None else "N/A"

    print(f"Saved  → {output_file}")
    print(f"  Final energy : {_fmt(results['final_energy'])}")
    print(f"  Exact energy : {_fmt(results['exact_energy'])}")
    print(f"  Error        : {_fmt(results['error'])}")
    # For plots
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 7. Plot if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history["energy"])
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


def save_dwave_samples(V: np.ndarray, args, iteration: int) -> Path:
    """
    Save raw D-Wave visible-unit samples for one training iteration.

    Stored as gzip-compressed pickle under:
        dwave_samples/{n_hidden}/{sampler}/{method}/
            samples_{model}_h{h}_rbm{rbm}_nh{n_hidden}_lr{lr}
            _reg{reg}_ns{ns}_seed{seed}_iter{IIII}.pkl.gz

    Content: {"v": ndarray(ns, N), "iteration": int, "config": dict}

    The .pkl.gz is intentionally small: only visible-unit spin configs are
    stored.  Hidden units and RBM weights can be reconstructed from the
    config and checkpoint files.  The goal is to avoid re-querying the QPU
    when new metrics need to be computed post-hoc.
    """
    out_dir = Path(
        f"dwave_samples/{args.n_hidden}/{args.sampler}/{args.sampling_method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"samples_{args.model}"
        f"_h{args.h}"
        f"_rbm{args.rbm}"
        f"_nh{args.n_hidden}"
        f"_lr{args.learning_rate}"
        f"_reg{args.regularization}"
        f"_ns{args.n_samples}"
        f"_seed{args.seed}"
        f"_iter{iteration:04d}"
        f".pkl.gz"
    )
    path = out_dir / fname
    with gzip.open(path, "wb") as f:
        pickle.dump(
            {"v": V, "iteration": iteration, "config": vars(args)}, f, protocol=5
        )
    return path


def log_solver_time_ms(
    elapsed_ms: float, time_path: Path = Path("time.json"), key="time_ms"
):
    """
    Thread/process-safe append of solver elapsed time (ms) to time.json.

    Uses the same exclusive-flock + atomic-rename pattern as
    DimodSampler._log_access_time so all solvers share a single file safely.

    key         : e.g. "time_ms" (D-Wave QPU), "velox_time_ms" (VeloxQ)
    elapsed_ms  : wall time in milliseconds to add
    time_path   : path to the shared JSON counter file
    """
    if not time_path.exists():
        with time_path.open("w") as f:
            json.dump({}, f)
    with time_path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
            data[key] = data.get(key, 0.0) + elapsed_ms
            tmp = time_path.with_suffix(".tmp")
            with tmp.open("w") as tf:
                json.dump(data, tf)
            tmp.rename(time_path)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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
        return "Advantage_system6.4"
    elif architecture == "zephyr":
        return "Advantage2_system1"

import fcntl
import gzip
import json
from pathlib import Path
import numpy as np
import pickle


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
    # Directory structure: checkpoints/size/sampler/sampling_method/rbm/
    checkpoint_dir = Path(
        f"{args.output_dir.replace('results', 'checkpoints')}/{args.size}/{args.sampler}/{args.sampling_method}/{args.rbm}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint data
    checkpoint = {
        "iteration": iteration,
        "config": vars(args),
        "rbm_state": {
            "a": rbm.a.tolist(),
            "b": rbm.b.tolist(),
            "W": rbm.W.tolist(),
            "n_visible": rbm.n_visible,
            "n_hidden": rbm.n_hidden,
        },
    }

    checkpoint_file = checkpoint_dir / (
        f"checkpoint"
        f"_{args.model}"
        f"_h{args.h}"
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

    rbm.a = np.array(rbm_state["a"])
    rbm.b = np.array(rbm_state["b"])
    rbm.W = np.array(rbm_state["W"])

    print(f"Restored RBM from checkpoint: {checkpoint_path}")
    print(f"  Starting from iteration {iteration}")

    return iteration


def save_results(args, history, ising, rbm=None):
    # Directory structure: results/size/sampler/sampling_method/
    output_dir = Path(
        f"{args.output_dir}/{args.size}/{args.sampler}/{args.sampling_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cem = getattr(args, "cem", False)
    results = {
        "config": vars(args),
        "history": {
            k: [float(v) if v is not None else None for v in vals]
            for k, vals in history.items()
        },
        "final_energy": history["energy"][-1],
        "exact_energy": ising.exact_ground_energy(),
        "error": abs(history["energy"][-1] - ising.exact_ground_energy()),
        "sparsity": float(rbm.sparsity()) if rbm is not None else None,
        "sampling_time_s": float(sum(history.get("sampling_time_s", []))),
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
        f"_h{args.h}"
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

    print(f"Saved  → {output_file}")
    print(f"  Final energy : {results['final_energy']:.6f}")
    print(f"  Exact energy : {results['exact_energy']:.6f}")
    print(f"  Error        : {results['error']:.6f}")
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
            plot_file = plot_dir / f"plot_{args.model}_h{args.h}_rbm{args.rbm}.png"
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


def get_solver_name(architecture="pegasus"):
    if architecture == "pegasus":
        return "Advantage_system6.4"
    elif architecture == "zephyr":
        return "Advantage2_system1.13"

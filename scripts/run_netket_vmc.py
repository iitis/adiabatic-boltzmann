#!/usr/bin/env python3
"""
NetKet VMC for the Transverse Field Ising Model,
using any of our sampler backends (custom / dimod / velox) as the sampling backend.

How the bridge works
--------------------
1. GardasRBM — a Flax nn.Module with parameters (a, b, W) matching our
   FullyConnectedRBM convention exactly:
       log Ψ(v) = -a·v/2 + (1/2) Σ_j log(2 cosh(b_j + W[:,j]·v))
   NetKet's VMC driver optimises these Flax parameters via SR (QGT + CG).

2. OurNetKetSampler — subclasses nk.sampler.Sampler.  Its _sample_chain
   overrides the default MCMC step.  At each VMC iteration it:
     a) extracts (a, b, W) from the current Flax parameter tree,
     b) copies them into a temporary RBM (our NumPy class),
     c) calls the configured sampler backend's .sample(...),
     d) returns the result as a JAX array.
   The hand-off between JAX and NumPy/PyTorch is done via jax.pure_callback,
   which lets non-JAX code run inside a jax.jit-compiled function.

Key NetKet concepts used here
------------------------------
nk.sampler.Sampler     — base class; subclasses must implement
                         _init_state / _reset / _sample_chain.
                         Fields declared as struct.field(pytree_node=False)
                         are compile-time constants embedded in the JIT key.
nk.vqs.MCState         — variational state; owns the sampler and model,
                         handles n_samples / chain_length accounting.
nk.driver.VMC_SR       — VMC with built-in Stochastic Reconfiguration (NGD).
jax.pure_callback      — bridge: call arbitrary Python (numpy/torch) from
                         inside a jax.jit-compiled function.

Supported samplers
------------------
  --sampler custom  --sampling-method metropolis        (Numba MH)
  --sampler custom  --sampling-method simulated_annealing
  --sampler custom  --sampling-method gibbs
  --sampler custom  --sampling-method lsb               (default; PyTorch SB)
  --sampler custom  --sampling-method sbm
  --sampler dimod   --sampling-method simulated_annealing
  --sampler dimod   --sampling-method tabu
  --sampler dimod   --sampling-method pegasus            (D-Wave QPU)
  --sampler dimod   --sampling-method zephyr             (D-Wave QPU)
  --sampler velox   --sampling-method velox
  --sampler velox   --sampling-method sbm

Run from src/
    python ../scripts/netket_lsb_vmc.py
    python ../scripts/netket_lsb_vmc.py --sampler custom --sampling-method metropolis --size 8 --h 0.5
    python ../scripts/netket_lsb_vmc.py --sampler dimod  --sampling-method pegasus --rbm pegasus --size 8
"""

import sys
import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

# ── JAX must be in x64 mode before any other JAX import ──────────────────────
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from netket.sampler import Sampler, SamplerState
from netket.utils import struct

# ── make src/ importable ──────────────────────────────────────────────────────
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))
os.chdir(SRC)

from model import FullyConnectedRBM, DWaveTopologyRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler


# =============================================================================
# 1. RBM ansatz as a Flax module  (Gardas convention)
# =============================================================================


class GardasRBM(nn.Module):
    """
    log Ψ(v) = -a·v/2 + (1/2) Σ_j log(2 cosh(b_j + W[:,j]·v))

    Parameter names (a, b, W) deliberately match FullyConnectedRBM so the
    sampler callback can copy them in and out without any re-mapping.
    """

    n_hidden: int
    param_dtype: type = np.float64

    @nn.compact
    def __call__(self, v):
        # v : (..., n_visible)  values in {+1, -1}  (NetKet Spin(s=0.5) convention)
        n_vis = v.shape[-1]
        a = self.param("a", nn.initializers.normal(0.01), (n_vis,), self.param_dtype)
        b = self.param(
            "b", nn.initializers.normal(0.01), (self.n_hidden,), self.param_dtype
        )
        W = self.param(
            "W",
            nn.initializers.normal(0.01),
            (n_vis, self.n_hidden),
            self.param_dtype,
        )
        theta = v @ W + b  # (..., n_hidden)
        return -0.5 * (v @ a) + 0.5 * jnp.sum(jnp.logaddexp(theta, -theta), axis=-1)


# =============================================================================
# 2. Sampler state  (independent of backend; only an RNG key needed)
# =============================================================================


class OurSamplerState(SamplerState):
    """Minimal state: only an RNG key (not consumed by independent samplers; kept for NetKet compat)."""

    rng: jax.Array

    def __init__(self, rng: jax.Array):
        self.rng = rng
        super().__init__()

    def __repr__(self):
        return f"OurSamplerState(rng={self.rng})"


# =============================================================================
# 3. Custom sampler — NetKet Sampler subclass
# =============================================================================


class OurNetKetSampler(Sampler):
    """
    NetKet-compatible sampler that drives any of our backends under the hood:
      ClassicalSampler  (custom)  — metropolis / SA / gibbs / lsb / sbm
      DimodSampler      (dimod)   — SA / tabu / pegasus / zephyr
      VeloxSampler      (velox)   — velox / sbm

    Bridge to NumPy/PyTorch
    ~~~~~~~~~~~~~~~~~~~~~~~
    _sample_chain is jax.jit-compiled, but our samplers are NumPy + optional
    PyTorch/dimod.  jax.pure_callback handles this: during tracing it records
    the call site symbolically; at execution time JAX materialises the traced
    arrays to numpy and calls our Python function.

    Static fields  (pytree_node=False)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NetKet's struct.Pytree embeds pytree_node=False fields into the JIT cache
    key.  Changing them triggers a recompile.  We tag all backend config as
    static so they can be used as concrete Python values inside _sample_chain.
    """

    # total independent samples == MCState.n_samples when chain_length = 1
    n_chains:        int = struct.field(pytree_node=False)
    # which backend and method to use
    sampler_backend: str = struct.field(pytree_node=False, default="custom")
    sampling_method: str = struct.field(pytree_node=False, default="lsb")
    # RBM type: "full" → FullyConnectedRBM; "pegasus"/"zephyr" → DWaveTopologyRBM
    rbm_type:        str = struct.field(pytree_node=False, default="full")
    # scaling of Ising couplings (meaningful for D-Wave / velox / SA; ignored by MH)
    beta_x:          float = struct.field(pytree_node=False, default=1.0)
    # steps for lsb / sbm methods (ignored for all others)
    lsb_steps:       int = struct.field(pytree_node=False, default=500)

    def __init__(
        self,
        hilbert,
        n_chains: int = 512,
        sampler_backend: str = "custom",
        sampling_method: str = "lsb",
        rbm_type: str = "full",
        beta_x: float = 1.0,
        lsb_steps: int = 500,
        dtype=None,
    ):
        super().__init__(hilbert, dtype=dtype)
        self.n_chains        = n_chains
        self.sampler_backend = sampler_backend
        self.sampling_method = sampling_method
        self.rbm_type        = rbm_type
        self.beta_x          = beta_x
        self.lsb_steps       = lsb_steps

    # ------------------------------------------------------------------
    # Abstract method implementations required by nk.sampler.Sampler
    # ------------------------------------------------------------------

    def _init_state(self, machine, params, seed) -> OurSamplerState:
        """Called once when MCState is constructed."""
        return OurSamplerState(rng=seed)

    def _reset(self, machine, parameters, state) -> OurSamplerState:
        """Called before each sampling round (after a parameter update).
        Independent samplers are stateless — nothing to reset."""
        return state

    @partial(
        jax.jit,
        static_argnames=("machine", "chain_length", "return_log_probabilities"),
    )
    def _sample_chain(
        self,
        machine,
        parameters,
        state: OurSamplerState,
        chain_length: int,
        return_log_probabilities: bool = False,
    ):
        """
        Generate (n_batches × chain_length) independent samples via our backend.

        Required output shape: (n_batches, chain_length, hilbert.size).
        For a single-device run n_batches == n_chains, so the total sample
        count is n_chains × chain_length == MCState.n_samples.
        """
        n_batches = self.n_batches   # compile-time int (pytree_node=False)
        n_vis     = self.hilbert.size  # compile-time int
        n_total   = n_batches * chain_length  # compile-time int
        _backend  = self.sampler_backend  # compile-time str
        _method   = self.sampling_method  # compile-time str
        _rbm_type = self.rbm_type         # compile-time str
        _beta_x   = self.beta_x           # compile-time float
        _steps    = self.lsb_steps        # compile-time int

        # Extract RBM parameters from the Flax variable pytree
        W = parameters["params"]["W"]  # JAX array (n_vis, n_hid)
        a = parameters["params"]["a"]  # JAX array (n_vis,)
        b = parameters["params"]["b"]  # JAX array (n_hid,)

        # --- Bridge: call our sampler backend via jax.pure_callback ----------
        def _sampler_callback(W_np, a_np, b_np):
            """
            Executed at runtime (not during JAX tracing) with concrete numpy
            arrays.  Builds a temporary RBM + sampler, runs sampling, and
            returns samples as float64 of shape (n_batches, chain_length, n_vis).
            """
            n_hid = int(W_np.shape[1])

            # ── instantiate the right RBM ────────────────────────────────────
            if _rbm_type in ("pegasus", "zephyr"):
                rbm = DWaveTopologyRBM(int(W_np.shape[0]), n_hid, solver=_rbm_type)
            else:
                rbm = FullyConnectedRBM(int(W_np.shape[0]), n_hid)
            # set_weights re-applies the topology mask for DWaveTopologyRBM
            rbm.set_weights(
                np.concatenate([
                    a_np.astype(np.float64),
                    b_np.astype(np.float64),
                    W_np.astype(np.float64).ravel(),
                ])
            )

            # ── instantiate the right sampler ────────────────────────────────
            if _backend == "custom":
                sampler_obj = ClassicalSampler(_method)
            elif _backend == "dimod":
                sampler_obj = DimodSampler(_method)
            elif _backend == "velox":
                sampler_obj = VeloxSampler(_method)
            else:
                raise ValueError(f"Unknown sampler backend: {_backend!r}")

            # ── build config ─────────────────────────────────────────────────
            cfg: dict = {"beta_x": _beta_x}
            if _method in ("lsb", "sbm"):
                cfg["lsb_steps"] = _steps

            result = sampler_obj.sample(rbm, n_total, config=cfg)
            # lsb / gibbs return (v, h); all other methods return v directly
            V = result[0] if isinstance(result, tuple) else result
            return V.reshape(n_batches, chain_length, n_vis).astype(np.float64)

        # Declare the output shape/dtype so JAX can trace through the call site
        result_shape = jax.ShapeDtypeStruct(
            (n_batches, chain_length, n_vis), jnp.float64
        )
        samples = jax.pure_callback(_sampler_callback, result_shape, W, a, b)

        # Advance the dummy RNG key so NetKet's serialisation stays consistent
        new_rng, _ = jax.random.split(state.rng)
        new_state = state.replace(rng=new_rng)

        if return_log_probabilities:
            flat     = samples.reshape(n_total, n_vis)
            log_psi  = machine.apply(parameters, flat)
            log_prob = self.machine_pow * jnp.real(log_psi).reshape(
                n_batches, chain_length
            )
            return (samples, log_prob), new_state

        return samples, new_state


# =============================================================================
# 4. Main — VMC training loop
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NetKet VMC with pluggable sampler backend for TFIM"
    )
    parser.add_argument("--model",    choices=["1d", "2d"], default="1d")
    parser.add_argument("--size",     type=int,   default=8)
    parser.add_argument("--h",        type=float, default=0.5)
    parser.add_argument("--n-hidden", type=int,   default=None,
                        help="Hidden units (default = n_visible)")
    parser.add_argument("--n-samples", type=int,  default=512,
                        help="Samples per VMC iteration (= n_chains)")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--lr",       type=float, default=0.01,
                        help="SGD learning rate (used inside SR/NGD step)")
    parser.add_argument("--diag-shift", type=float, default=1e-4,
                        help="SR diagonal regularisation of the QGT")
    # ── sampler selection ──────────────────────────────────────────────────────
    parser.add_argument(
        "--sampler",
        choices=["custom", "dimod", "velox"],
        default="custom",
        help="Sampling backend",
    )
    parser.add_argument(
        "--sampling-method",
        choices=[
            "metropolis", "simulated_annealing", "gibbs", "lsb", "sbm",  # custom
            "tabu", "pegasus", "zephyr",                                   # dimod
            "velox",                                                        # velox
        ],
        default="lsb",
        help="Algorithm within the selected backend",
    )
    parser.add_argument(
        "--rbm",
        choices=["full", "pegasus", "zephyr"],
        default="full",
        help=(
            "RBM connectivity: 'full' = dense; "
            "'pegasus'/'zephyr' = sparse D-Wave topology (use with --sampling-method pegasus/zephyr)"
        ),
    )
    parser.add_argument(
        "--beta-x", type=float, default=1.0,
        help="Ising coupling scale factor (meaningful for D-Wave / velox / SA; ignored by MH)",
    )
    parser.add_argument(
        "--lsb-steps", type=int, default=500,
        help="Dynamics steps for lsb / sbm methods (ignored for all others)",
    )
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--output",      type=Path, default=Path("../plots/netket_vmc"),
                        help="Directory for convergence PNG plots")
    parser.add_argument("--results-dir", type=Path, default=Path("../results_netket"),
                        help="Directory for JSON result files (same format as main.py results)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    n_vis = args.size if args.model == "1d" else args.size**2
    n_hid = args.n_hidden or n_vis

    # ── 1. Hamiltonian  H = -J Σ σz_i σz_{i+1} - h Σ σx_i ──────────────
    if args.model == "1d":
        graph = nk.graph.Chain(args.size, max_neighbor_order=1)
    else:
        graph = nk.graph.Square(args.size, max_neighbor_order=1)

    hilbert = nk.hilbert.Spin(s=0.5, N=n_vis)
    H = nk.operator.Ising(hilbert=hilbert, graph=graph, h=args.h, J=1.0)

    # Reference energies per spin for 2D TFIM (thermodynamic limit)
    reference_energies_per_spin = {
        0.5: -2.0555,
        1.0: -2.1276,
        2.0: -2.4549,
        3.044: -3.0440,  # critical point
    }

    if args.model == "2d" and args.h in reference_energies_per_spin:
        E_exact = args.size**2 * reference_energies_per_spin[args.h]
    else:
        try:
            E_exact = float(nk.exact.lanczos_ed(H, compute_eigenvectors=False)[0])
            print(f"Exact ground energy : {E_exact:.6f}  ({E_exact / n_vis:.6f} / site)")
        except Exception as e:
            E_exact = None
            print(f"[warning] Could not compute exact energy: {e}")

    # ── 2. Variational state ──────────────────────────────────────────────
    model   = GardasRBM(n_hidden=n_hid)
    sampler = OurNetKetSampler(
        hilbert,
        n_chains        = args.n_samples,
        sampler_backend = args.sampler,
        sampling_method = args.sampling_method,
        rbm_type        = args.rbm,
        beta_x          = args.beta_x,
        lsb_steps       = args.lsb_steps,
    )
    # n_samples must equal n_chains when chain_length=1 (the default for
    # independent samplers).  MCState enforces n_samples % n_chains == 0.
    vs = nk.vqs.MCState(sampler, model, n_samples=args.n_samples, seed=args.seed)

    n_params = sum(p.size for p in jax.tree.leaves(vs.parameters))
    print(f"\nModel   : GardasRBM  N_vis={n_vis}  N_hid={n_hid}  params={n_params}")
    print(
        f"Sampler : {args.sampler}/{args.sampling_method}"
        f"  rbm={args.rbm}"
        f"  n_chains={args.n_samples}"
        f"  beta_x={args.beta_x}"
        + (f"  lsb_steps={args.lsb_steps}" if args.sampling_method in ("lsb", "sbm") else "")
    )

    # ── 3. VMC_SR driver  (SGD + Stochastic Reconfiguration) ─────────────
    import optax

    driver = nk.driver.VMC_SR(
        H,
        optax.sgd(learning_rate=args.lr),
        variational_state=vs,
        diag_shift=args.diag_shift,
    )

    # ── 4. Training loop ──────────────────────────────────────────────────
    print(
        f"\nTraining {args.iterations} iterations"
        f"  model={args.model}  N={args.size}  h={args.h}\n"
    )

    energies   = []
    errors     = []
    rel_errors = []

    for step in driver.iter(args.iterations):
        E_stats = driver._loss_stats
        e_mean  = float(E_stats.mean.real)
        e_err   = float(E_stats.error_of_mean)
        energies.append(e_mean)
        errors.append(e_err)
        if E_exact is not None:
            rel = abs(e_mean - E_exact) / abs(E_exact)
            rel_errors.append(rel)

        if step % 10 == 0 or step == args.iterations - 1:
            line = f"  iter {step:4d}  E = {e_mean:10.5f} ± {e_err:.4f}"
            if rel_errors:
                line += f"   rel_err = {rel_errors[-1]:.4f}"
            print(line)

    # ── 5. Final summary ──────────────────────────────────────────────────
    print(f"\nFinal energy  : {energies[-1]:.6f}")
    if E_exact is not None:
        print(f"Exact energy  : {E_exact:.6f}")
        print(f"Relative error: {rel_errors[-1]:.4f}")

    # ── 6. Convergence plot ───────────────────────────────────────────────
    iters = np.arange(len(energies))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    label = f"VMC ({args.sampler}/{args.sampling_method})"
    ax.plot(iters, energies, lw=1.5, label=label)
    ax.fill_between(
        iters,
        np.array(energies) - np.array(errors),
        np.array(energies) + np.array(errors),
        alpha=0.25,
    )
    if E_exact is not None:
        ax.axhline(E_exact, ls="--", color="k", lw=1.2, label=f"Exact ({E_exact:.4f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy convergence — {args.model} TFIM  N={args.size}  h={args.h}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if rel_errors:
        ax2.semilogy(iters, rel_errors, lw=1.5, color="C1")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("|E − E_exact| / |E_exact|")
        ax2.set_title("Relative error vs exact")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Exact energy unavailable",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(
        f"NetKet VMC + {args.sampler}/{args.sampling_method}  "
        f"(N_vis={n_vis}, N_hid={n_hid}, n_samples={args.n_samples})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    tag  = f"{args.model}_N{args.size}_h{args.h}_{args.sampler}_{args.sampling_method}"
    path = out / f"convergence_{tag}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── 7. Save JSON result (same schema as helpers.save_results) ─────────
    import json as _json

    # path: results_netket/{n_hid}/netket/{sampler}_{method}/result_...json
    # sampler="netket" in config makes sampler_analysis.py keep these separate
    method_key = f"{args.sampler}_{args.sampling_method}"
    res_dir = (
        args.results_dir
        / str(n_hid)
        / "netket"
        / method_key
    )
    res_dir.mkdir(parents=True, exist_ok=True)
    res_file = res_dir / (
        f"result_{args.model}"
        f"_h{args.h}"
        f"_rbm{args.rbm}"
        f"_nh{n_hid}"
        f"_lr{args.lr}"
        f"_ns{args.n_samples}"
        f"_seed{args.seed}"
        f"_iter{args.iterations}"
        f".json"
    )

    result = {
        "config": {
            "model":           args.model,
            "size":            args.size,
            "h":               args.h,
            "sampler":         "netket",           # keeps sampler_analysis keys separate
            "sampling_method": method_key,         # e.g. "custom_lsb", "dimod_pegasus"
            "rbm":             args.rbm,
            "n_hidden":        n_hid,
            "learning_rate":   args.lr,
            "n_samples":       args.n_samples,
            "n_iterations":    args.iterations,
            "seed":            args.seed,
            "framework":       "netket",           # informational tag
            "backend_sampler": args.sampler,
            "backend_method":  args.sampling_method,
        },
        "final_energy": energies[-1] if energies else None,
        "exact_energy":  E_exact,
        "error":         (energies[-1] - E_exact) if (energies and E_exact is not None) else None,
        "history": {
            "energy":       energies,
            "energy_error": errors,
            "rel_error":    rel_errors,
            # fields expected by sampler_analysis.py (empty = handled gracefully)
            "beta_x":          [],
            "beta_eff_cem":    [],
            "grad_norm":       [],
            "weight_norm":     [],
            "cg_iterations":   [],
            "sampling_time_s": [],
            "ess":             [],
            "kl_exact":        [],
        },
    }
    res_file.write_text(_json.dumps(result, indent=2))
    print(f"Saved JSON : {res_file}")


if __name__ == "__main__":
    main()

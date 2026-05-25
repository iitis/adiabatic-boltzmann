"""
Sample-quality assessment: fast anneal vs. standard anneal vs. |Ψ(v)|².

In the Gardas et al. VMC framework, |Ψ(v)|² is a Gibbs distribution at
inverse temperature β=1 for the RBM Hamiltonian.  The QPU must sample this
distribution to produce unbiased VMC gradients.

This experiment sweeps the inverse temperature β at which the problem is
submitted to the QPU and measures how closely the output matches |Ψ|²:

  - fast anneal (7 ns): h biases dropped (hardware constraint); J couplings only
  - standard anneal at t ∈ {1, 5, 25} µs: full Hamiltonian (J + h) submitted

  β_in sweep ∈ [0.1 … 1.0]  (inverse temperature of submitted Hamiltonian,
                               normalised so β_in=1 uses max hardware coupling)

Three metrics per (condition, β_in):
  D_TV(ν, |Ψ|²)               — quality as VMC sampler
  D_TV(ν, p_target(β_eff))    — hardware noise: how well QPU samples its own target
  D_TV(p_target(β_eff), |Ψ|²) — structural approximation error (no QPU needed)

  β_eff = argmin_β D_TV(ν, p_target(·;β)) — effective inverse temperature delivered
                                             by the QPU.

  The reference distribution p_target is condition-specific:
    fast anneal  → p_J(v;β)    ∝ ∏_j 2·cosh(β·(W^T v)_j)   [h-biases absent]
    standard     → p_full(v;β) ∝ exp[β·a·v]·∏_j 2·cosh(β·(b_j+(W^T v)_j))

  Triangle inequality: D_TV(ν,|Ψ|²) ≤ D_TV(ν,p_target(β_eff)) + D_TV(p_target(β_eff),|Ψ|²)

Protocol follows Nelson et al. (2022):
  - auto_scale=False; couplings submitted as β_in · H_RBM / ‖H_RBM‖_∞.
  - Gauge transform applied every batch to mitigate hardware bias.
  - Finite-sampling lower bound estimated from bootstrap resampling of |Ψ|².

Usage
-----
    cd <repo-root>
    python scripts/exper/experiment_sample_quality.py
    python scripts/exper/experiment_sample_quality.py --dry-run
    python scripts/exper/experiment_sample_quality.py --beta-in 0.2 0.3 0.4
    python scripts/exper/experiment_sample_quality.py --replot results/sample_quality/sample_quality_n8_h0.5_nh8.json

Results saved to results/sample_quality/.
"""

import argparse
import itertools
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

from encoder import Trainer
from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM
from sampler import DimodSampler

# ── experiment parameters ──────────────────────────────────────────────────────

N_VISIBLE       = 8
N_HIDDEN        = 8
H_FIELD         = 0.5
SEED            = 42

# RBM training (done once; checkpoint is reused on subsequent runs)
TRAIN_ITERS     = 150
TRAIN_LR        = 0.01
TRAIN_N_SAMPLES = 500
REG             = 1e-5

# QPU sweep
DEFAULT_BETA_IN   = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0]
ANNEAL_TIMES_US   = [1, 5, 25]       # standard anneal times (µs)
FAST_ANNEAL_NS    = 7.0

READS_PER_BATCH   = 500              # QPU reads per gauge-transform batch
N_GAUGE_BATCHES   = 10               # → 5 000 total samples per condition

# β range for finding effective output inverse temperature
# Start near 0 so low-temperature (small β_in) conditions are not clamped.
BETA_EFF_RANGE    = np.linspace(0.01, 15.0, 500)

# finite-sampling lower bound estimation
N_BOOTSTRAP_LB    = 200

OUT_DIR           = Path("results/sample_quality")
CHECKPOINT_PATH   = OUT_DIR / "rbm_n8_h0.5_seed42.pkl"
SOLVER            = "Advantage_system6.4"  # Pegasus


# ── exact distributions (closed form) ─────────────────────────────────────────

def enumerate_visible(n_v: int) -> np.ndarray:
    """All 2^n_v configs in {-1,+1}^n_v, shape (2^n_v, n_v)."""
    return np.array(list(itertools.product([-1, 1], repeat=n_v)), dtype=np.float64)


def exact_psi_sq(rbm, all_v: np.ndarray) -> np.ndarray:
    """Normalised |Ψ(v)|² over all visible configs."""
    log_psi = np.array([float(rbm.log_psi(jnp.asarray(v))) for v in all_v])
    log_p = 2.0 * log_psi
    log_p -= logsumexp(log_p)
    return np.exp(log_p)


def exact_p_j(rbm, all_v: np.ndarray, beta: float) -> np.ndarray:
    """
    Normalised p_J(v; β) ∝ ∏_j 2·cosh(β·(W^T v)_j) — Gibbs distribution at
    inverse temperature β for the h-dropped Hamiltonian (J couplings only).
    """
    W = np.asarray(rbm.W)
    theta = all_v @ W                   # (2^n, n_h)
    log_p = np.sum(np.log(2.0 * np.cosh(beta * theta)), axis=1)
    log_p -= logsumexp(log_p)
    return np.exp(log_p)


def exact_p_full(rbm, all_v: np.ndarray, beta: float) -> np.ndarray:
    """
    Normalised p_full(v; β) — Gibbs distribution at inverse temperature β for
    the full RBM Hamiltonian (J + h), as mapped by rbm_to_ising convention.

    Marginalising over hidden units:
      p_full(v; β) ∝ exp[β·a·v] · ∏_j 2·cosh(β·(b_j + (W^T v)_j))

    At β=1, a=0: equals |Ψ|² exactly (the VMC target).
    The sign-of-a term (exp[β·a·v] vs exp[-a·v] in |Ψ|²) is a known artefact
    of the rbm_to_ising convention; it vanishes when a≈0.
    """
    W = np.asarray(rbm.W)
    a = np.asarray(rbm.a)
    b = np.asarray(rbm.b)
    theta = all_v @ W                                    # (2^n, n_h)
    log_p = (beta * (all_v @ a)
             + np.sum(np.log(2.0 * np.cosh(beta * (b + theta))), axis=1))
    log_p -= logsumexp(log_p)
    return np.exp(log_p)


def dtv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def finite_sampling_lb(p: np.ndarray, n_samples: int, rng: np.random.Generator) -> float:
    """
    Mean D_TV(p̂, p) over bootstrap draws of n_samples from p.
    This is the lower bound achievable purely from finite sampling.
    """
    states = np.arange(len(p))
    dtvs = []
    for _ in range(N_BOOTSTRAP_LB):
        idx = rng.choice(states, size=n_samples, p=p)
        counts = np.bincount(idx, minlength=len(p))
        dtvs.append(dtv(counts / n_samples, p))
    return float(np.mean(dtvs))


def find_beta_eff(nu_emp: np.ndarray, target_fn) -> tuple[float, float]:
    """
    β_eff = argmin_β D_TV(ν_emp, target_fn(β)) — the effective inverse temperature
    delivered by the QPU for the submitted model family.
    Returns (beta_eff, min_dtv).
    """
    dtvs = [dtv(nu_emp, target_fn(b)) for b in BETA_EFF_RANGE]
    idx = int(np.argmin(dtvs))
    return float(BETA_EFF_RANGE[idx]), float(dtvs[idx])


# ── empirical distribution ─────────────────────────────────────────────────────

def empirical_dist(samples_v: np.ndarray, n_visible: int) -> np.ndarray:
    """
    Build empirical distribution from visible samples ∈ {-1,+1}^n_v.
    Index k matches enumerate_visible (itertools.product order): MSB-first,
    i.e. index = v[0]·2^{n-1} + v[1]·2^{n-2} + … + v[n-1]·2^0.
    """
    n_states = 2 ** n_visible
    bits = ((samples_v + 1) // 2).astype(int)           # {-1,+1} → {0,1}
    # MSB-first: leftmost spin has weight 2^(n-1), rightmost has weight 1.
    indices = bits @ (2 ** np.arange(n_visible - 1, -1, -1))
    counts = np.bincount(indices.astype(int), minlength=n_states)
    return counts / counts.sum()


# ── BQM construction & gauge transforms ───────────────────────────────────────

def build_j_only_bqm(rbm, beta_in: float):
    """
    J-only BQM: no linear terms, J_ij = β_in · W_ij / ‖W‖_∞.
    Used for fast anneal (hardware requires h=0).
    Returns (bqm, w_max) where w_max is the normalization constant.
    """
    import dimod
    W = np.asarray(rbm.W)
    w_max = float(np.max(np.abs(W)))
    if w_max < 1e-12:
        raise ValueError("RBM weights are all zero; cannot build BQM.")
    n_v, n_h = rbm.n_visible, rbm.n_hidden
    quadratic = {
        (i, n_v + j): -float(W[i, j]) / w_max * beta_in
        for i in range(n_v) for j in range(n_h)
        if abs(W[i, j]) > 1e-10
    }
    linear = {k: 0.0 for k in range(n_v + n_h)}
    return dimod.BinaryQuadraticModel.from_ising(linear, quadratic), w_max


def build_full_bqm(rbm, beta_in: float, dsampler: DimodSampler):
    """
    Full BQM (J + h) via DimodSampler.rbm_to_ising at inverse temperature β_in.
    Couplings are submitted as β_in · H_RBM / ‖H_RBM‖_∞ to stay within [-1,1].
    Returns (bqm, w_max) where w_max = ‖H_RBM‖_∞ is the normalization constant.
    """
    import dimod
    W = np.asarray(rbm.W); a = np.asarray(rbm.a); b = np.asarray(rbm.b)
    w_max = float(max(np.max(np.abs(W)), np.max(np.abs(a)), np.max(np.abs(b))))
    if w_max < 1e-12:
        raise ValueError("RBM parameters all zero; cannot build BQM.")
    # rbm_to_ising divides by beta_x, so beta_x = w_max/beta_in gives
    # submitted values = beta_in * param / w_max ∈ [-beta_in, beta_in]
    beta_x = w_max / beta_in
    quadratic, linear = dsampler.rbm_to_ising(rbm, beta_x=beta_x)
    bqm = dimod.BinaryQuadraticModel.from_ising(linear, quadratic)
    return bqm, w_max


def gauge_transform(bqm, n_total: int, rng: np.random.Generator):
    """
    Random spin-reversal gauge: h_i → g_i h_i, J_ij → g_i g_j J_ij.
    Returns (transformed_bqm, gauge_vec g ∈ {-1,+1}^n_total).
    Undo on output: σ_actual = g * σ_sampled.
    """
    import dimod
    g = rng.choice(np.array([-1, 1]), size=n_total)
    new_lin  = {i: float(bqm.get_linear(i)) * float(g[i]) for i in range(n_total)}
    new_quad = {(i, j): v * float(g[i]) * float(g[j]) for (i, j), v in bqm.quadratic.items()}
    return dimod.BinaryQuadraticModel.from_ising(new_lin, new_quad), g


# ── QPU sampling ──────────────────────────────────────────────────────────────

def _prepare_sampler(dsampler: DimodSampler, n_v: int, n_h: int) -> None:
    """Set sampler attributes needed by fast_anneal / dwave without going through sample()."""
    dsampler.n_visible = n_v
    dsampler.n_hidden  = n_h
    dsampler._n_cache  = n_v + n_h


def collect_samples(
    dsampler: DimodSampler,
    bqm,
    n_v: int,
    n_h: int,
    reads_per_batch: int,
    n_batches: int,
    rng: np.random.Generator,
    is_fast: bool,
    anneal_time: float,     # ns if fast, µs if standard
    dry_run: bool = False,
) -> np.ndarray:
    """
    Collect visible samples with gauge transforms.
    Returns ndarray of shape (reads_per_batch * n_batches, n_v) in {-1,+1}.
    """
    _prepare_sampler(dsampler, n_v, n_h)
    n_total = n_v + n_h
    all_visible = []

    for b in range(n_batches):
        bqm_g, g = gauge_transform(bqm, n_total, rng)
        config = {"auto_scale": False, "solver": SOLVER, "num_reads": reads_per_batch}

        if dry_run:
            # Return random ±1 samples without hitting the QPU.
            v_raw = rng.choice(np.array([-1, 1]), size=(reads_per_batch, n_v))
        elif is_fast:
            config["fast_anneal_time_ns"] = anneal_time
            v_raw = dsampler.fast_anneal(bqm_g, reads_per_batch, config=config)
        else:
            config["annealing_time"] = int(anneal_time)
            v_raw = dsampler.dwave(bqm_g, reads_per_batch, config=config)

        v_corrected = v_raw * g[:n_v]
        all_visible.append(v_corrected)
        unique = len(set(map(tuple, v_corrected.tolist())))
        print(f"      batch {b+1}/{n_batches}: {unique} unique visible configs")

    return np.vstack(all_visible)


# ── training / checkpoint ──────────────────────────────────────────────────────

def train_or_load_rbm(dry_run: bool) -> FullyConnectedRBM:
    """Train a fresh RBM on TFIM-1D N=8 or load an existing checkpoint."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_PATH.exists():
        print(f"[checkpoint] Loading RBM from {CHECKPOINT_PATH}")
        with CHECKPOINT_PATH.open("rb") as f:
            rbm = pickle.load(f)
        return rbm

    if dry_run:
        print("[dry-run] Skipping training — returning uninitialised RBM.")
        key = jax.random.PRNGKey(SEED)
        return FullyConnectedRBM(N_VISIBLE, N_HIDDEN, key)

    print(f"[train] TFIM-1D N={N_VISIBLE}, h={H_FIELD}, n_hidden={N_HIDDEN}, iters={TRAIN_ITERS}")
    ising = TransverseFieldIsing1D(N_VISIBLE, H_FIELD)
    key   = jax.random.PRNGKey(SEED)
    rbm   = FullyConnectedRBM(N_VISIBLE, N_HIDDEN, key)

    from sampler import ClassicalSampler
    sampler = ClassicalSampler("metropolis")
    trainer = Trainer(
        rbm, ising, sampler,
        config={
            "n_samples":      TRAIN_N_SAMPLES,
            "learning_rate":  TRAIN_LR,
            "regularization": REG,
            "seed":           SEED,
            "n_iterations":   TRAIN_ITERS,
        },
    )
    trainer.train()

    print(f"[checkpoint] Saving to {CHECKPOINT_PATH}")
    with CHECKPOINT_PATH.open("wb") as f:
        pickle.dump(rbm, f, protocol=5)
    return rbm


# ── main experiment loop ───────────────────────────────────────────────────────

def run_experiment(beta_in_list: list[float], dry_run: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── 1. RBM ────────────────────────────────────────────────────────────────
    rbm      = train_or_load_rbm(dry_run)
    all_v    = enumerate_visible(N_VISIBLE)
    p_psi_sq = exact_psi_sq(rbm, all_v)
    n_total_samples = READS_PER_BATCH * N_GAUGE_BATCHES

    print(f"\n[exact] |Ψ|² over {len(all_v)} configs computed.")
    print(f"[finite-sampling LB] estimating with {N_BOOTSTRAP_LB} bootstrap reps …")
    fs_lb = finite_sampling_lb(p_psi_sq, n_total_samples, rng)
    print(f"[finite-sampling LB] D_TV lower bound ≈ {fs_lb*100:.2f}%")

    # ── 2. Conditions to run ──────────────────────────────────────────────────
    conditions = [("fast", True, FAST_ANNEAL_NS, "fast_7ns")]
    for t in ANNEAL_TIMES_US:
        conditions.append(("std", False, float(t), f"std_{t}us"))

    # ── 3. DimodSampler (single instance, embedding cached) ───────────────────
    # Use pegasus_fast method; we'll override inside collect_samples per condition.
    dsampler = DimodSampler("pegasus_fast")

    # ── 5. Main sweep ─────────────────────────────────────────────────────────
    results = {
        "meta": {
            "n_visible": N_VISIBLE,
            "n_hidden":  N_HIDDEN,
            "h_field":   H_FIELD,
            "beta_in":   beta_in_list,
            "anneal_times_us": ANNEAL_TIMES_US,
            "reads_per_batch": READS_PER_BATCH,
            "n_gauge_batches": N_GAUGE_BATCHES,
            "finite_sampling_lb": fs_lb,
            "timestamp": datetime.now().isoformat(),
        },
        "conditions": {},
    }

    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(exist_ok=True)

    for kind, is_fast, anneal_t, label in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {label}  ({'fast anneal' if is_fast else f'{int(anneal_t)} µs standard'})")
        results["conditions"][label] = {"beta_in": [], "entries": []}

        for beta_in in beta_in_list:
            print(f"\n  β_in = {beta_in:.3f}")

            # 5a. Build BQM — fast anneal drops h, standard anneal keeps h.
            # target_fn must match the distribution the QPU actually samples:
            #   fast anneal → p_J (J only); standard anneal → p_full (J + h).
            # Using the wrong family makes β_eff meaningless (a spurious fit).
            if is_fast:
                bqm, w_max = build_j_only_bqm(rbm, beta_in)
                target_fn = lambda b: exact_p_j(rbm, all_v, b)
            else:
                bqm, w_max = build_full_bqm(rbm, beta_in, dsampler)
                target_fn = lambda b: exact_p_full(rbm, all_v, b)

            # 5b. Collect QPU samples
            samples_v = collect_samples(
                dsampler, bqm, N_VISIBLE, N_HIDDEN,
                READS_PER_BATCH, N_GAUGE_BATCHES, rng,
                is_fast=is_fast, anneal_time=anneal_t, dry_run=dry_run,
            )

            # 5c. Save raw visible samples (overwrite on re-run)
            tag = f"n{N_VISIBLE}_h{H_FIELD}_nh{N_HIDDEN}"
            sample_path = samples_dir / f"samples_{tag}_{label}_bin{beta_in:.3f}.npy"
            np.save(sample_path, samples_v)

            # 5d. Empirical distribution over visible configs
            nu_emp = empirical_dist(samples_v, N_VISIBLE)

            # 5d. Effective inverse temperature recovered from QPU samples
            beta_eff, dtv_to_target = find_beta_eff(nu_emp, target_fn)

            # 5e. All metrics
            dtv_psi    = dtv(nu_emp, p_psi_sq)
            p_target   = target_fn(beta_eff)
            dtv_approx = dtv(p_target, p_psi_sq)  # approximation error at β_eff

            entry = {
                "beta_in":        beta_in,
                "beta_eff":       beta_eff,
                "w_max":          w_max,          # normalization: β_eff_physical = β_eff (in p_target units)
                "dtv_psi":        dtv_psi,        # D_TV(ν, |Ψ|²)
                "dtv_to_target":  dtv_to_target,  # D_TV(ν, p_target(β_eff)) — hardware noise
                "dtv_approx":     dtv_approx,     # D_TV(p_target(β_eff), |Ψ|²) — structural error
                "n_unique_vis":   int(np.sum(nu_emp > 0)),
                "qpu_time_ms":    getattr(dsampler, "last_sampling_time_s", 0.0) * 1000 * N_GAUGE_BATCHES,
            }
            results["conditions"][label]["beta_in"].append(beta_in)
            results["conditions"][label]["entries"].append(entry)

            print(
                f"    D_TV(ν,|Ψ|²)={dtv_psi*100:.1f}%  "
                f"D_TV(ν,target)={dtv_to_target*100:.1f}%  "
                f"approx_err={dtv_approx*100:.1f}%  "
                f"β_eff={beta_eff:.2f}  "
                f"unique={entry['n_unique_vis']}/256"
            )

    # ── 6. Save JSON ──────────────────────────────────────────────────────────
    tag = f"n{N_VISIBLE}_h{H_FIELD}_nh{N_HIDDEN}"
    json_path = OUT_DIR / f"sample_quality_{tag}.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {json_path}")

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    _plot_results(results, fs_lb, OUT_DIR / f"sample_quality_{tag}.pdf")
    _plot_beta_eff_heatmap(results, OUT_DIR / f"sample_quality_{tag}_betaeff.pdf")


def _plot_results(results: dict, fs_lb: float, out_path: Path) -> None:
    meta = results["meta"]
    conditions = results["conditions"]

    # Color map: fast=black, standard anneal by time
    palette = {
        "fast_7ns": ("k",  "fast 7 ns",      "^"),
        "std_1us":  ("#1f77b4", "std 1 µs",  "o"),
        "std_5us":  ("#2ca02c", "std 5 µs",  "s"),
        "std_25us": ("#d62728", "std 25 µs", "D"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_psi, ax_pj, ax_aout = axes

    for label, data in conditions.items():
        if not data["entries"]:
            continue
        color, name, marker = palette.get(label, ("gray", label, "o"))
        xs    = [e["beta_in"]       for e in data["entries"]]
        ypsi  = [e["dtv_psi"]       * 100 for e in data["entries"]]
        ypj   = [e["dtv_to_target"] * 100 for e in data["entries"]]
        yaout = [e["beta_eff"]            for e in data["entries"]]

        ax_psi.plot(xs, ypsi, marker=marker, color=color, label=name, ms=6)
        ax_pj.plot( xs, ypj,  marker=marker, color=color, label=name, ms=6)
        ax_aout.plot(xs, yaout, marker=marker, color=color, label=name, ms=6)

    # Structural approximation error per condition: D_TV(p_target(β_eff), |Ψ|²).
    # Fast anneal uses p_J → captures h-dropping + sign-of-a error.
    # Standard anneal uses p_full → captures sign-of-a error only.
    # Shown as dashed lines matching each condition's color.
    for label, data in conditions.items():
        if not data["entries"]:
            continue
        color, name, marker = palette.get(label, ("gray", label, "o"))
        xs_b = [e["beta_in"]    for e in data["entries"]]
        ys_b = [e["dtv_approx"] * 100 for e in data["entries"]]
        ax_psi.plot(xs_b, ys_b, color=color, ls="--", lw=1.2, zorder=0)

    # Finite-sampling lower bound
    for ax in (ax_psi, ax_pj):
        ax.axhline(fs_lb * 100, color="royalblue", lw=1.5, ls=":", label="finite-sampling LB")

    ax_psi.set_title(r"$D_\mathrm{TV}(\nu,\,|\Psi|^2)$  [%]")
    ax_pj.set_title(r"$D_\mathrm{TV}(\nu,\,p_\mathrm{target}(\beta_\mathrm{eff}))$  [%]")
    ax_aout.set_title(r"Effective inverse temperature $\beta_\mathrm{eff}$")

    # Add a proxy artist so the dashed approx-error style appears in the legend.
    from matplotlib.lines import Line2D
    approx_proxy = Line2D([0], [0], color="gray", ls="--", lw=1.2,
                          label=r"structural error $D_\mathrm{TV}(p_\mathrm{target},|\Psi|^2)$  (dashed)")

    for ax in axes:
        ax.set_xlabel(r"$\beta_\mathrm{in}$")
        ax.grid(alpha=0.3)

    for ax in (ax_psi, ax_pj, ax_aout):
        handles, labels = ax.get_legend_handles_labels()
        extra = [approx_proxy] if ax is ax_psi else []
        ax.legend(handles=handles + extra, fontsize=7)

    fig.suptitle(
        f"Sample quality: TFIM-1D N={meta['n_visible']}, h={meta['h_field']}  |  "
        f"{meta['reads_per_batch']*meta['n_gauge_batches']} samples/condition",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] {out_path}")
    plt.close(fig)


def _plot_beta_eff_heatmap(results: dict, out_path: Path) -> None:
    """
    Fig. 2 analog (Nelson et al. 2022): β_eff as a heatmap over
    β_in (columns) × condition (rows), with values annotated in each cell.

    Rows ordered by effective anneal time: fast_7ns at bottom (shortest),
    std_25us at top (longest).  β_eff for fast anneal is fitted against p_J;
    for standard anneal against p_full — so values are not directly comparable
    across the fast/standard boundary.
    """
    meta       = results["meta"]
    conditions = results["conditions"]

    # Row order: shortest anneal at bottom, longest at top (matches Nelson Fig. 2)
    row_order  = ["fast_7ns", "std_1us", "std_5us", "std_25us"]
    row_labels = ["fast 7 ns", "std 1 µs", "std 5 µs", "std 25 µs"]

    beta_in_vals = meta["beta_in"]
    n_cols = len(beta_in_vals)
    n_rows = len(row_order)

    # Build matrix of β_eff values (rows = conditions, cols = β_in)
    matrix = np.full((n_rows, n_cols), np.nan)
    for r, label in enumerate(row_order):
        if label not in conditions:
            continue
        entries = conditions[label]["entries"]
        for entry in entries:
            bi = entry["beta_in"]
            if bi in beta_in_vals:
                c = beta_in_vals.index(bi)
                matrix[r, c] = entry["beta_eff"]

    # Clamp display range: exclude pathological freeze-out values for color scale
    vmin = np.nanmin(matrix)
    # Use 95th percentile as vmax to avoid one outlier (e.g. std_25us β_in=0.75)
    vmax = np.nanpercentile(matrix, 95)

    fig, ax = plt.subplots(figsize=(len(beta_in_vals) * 0.9 + 1.5, n_rows * 0.85 + 1.2))

    im = ax.imshow(matrix, aspect="auto", origin="lower",
                   cmap="YlGn", vmin=vmin, vmax=vmax)

    # Annotate each cell with the β_eff value
    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix[r, c]
            if np.isnan(val):
                continue
            # White text on dark cells, black on light cells
            normed = (val - vmin) / max(vmax - vmin, 1e-9)
            txt_color = "white" if normed > 0.6 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{b:.2f}" for b in beta_in_vals], fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(r"$\beta_\mathrm{in}$", fontsize=11)
    ax.set_ylabel("Condition", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\beta_\mathrm{eff}$", fontsize=11)

    # Mark the Gardas target β=1 region: cells where β_eff ≈ 1
    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix[r, c]
            if not np.isnan(val) and 0.85 <= val <= 1.15:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=False, edgecolor="red", lw=2.0, zorder=5,
                ))

    ax.set_title(
        rf"Effective inverse temperature $\beta_\mathrm{{eff}}$"
        f"  |  TFIM-1D N={meta['n_visible']}, h={meta['h_field']}\n"
        r"Red border: $\beta_\mathrm{eff} \in [0.85,\,1.15]$ (Gardas VMC target)",
        fontsize=9,
    )

    # Separator line between fast anneal and standard anneal rows
    ax.axhline(0.5, color="white", lw=2.0, ls="--")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plot] {out_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run",  action="store_true", help="Skip QPU calls, use random samples.")
    p.add_argument("--beta-in", type=float, nargs="+", default=DEFAULT_BETA_IN,
                   help="β_in values to sweep (default: 0.1 0.2 0.25 0.3 0.35 0.4 0.5 0.75 1.0).")
    p.add_argument("--replot", type=str, default=None, metavar="JSON",
                   help="Regenerate PDF from an existing JSON result file; no QPU calls.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.replot:
        with open(args.replot) as f:
            results = json.load(f)
        fs_lb    = results["meta"]["finite_sampling_lb"]
        base     = Path(args.replot).with_suffix("")
        _plot_results(results, fs_lb, Path(str(base) + ".pdf"))
        _plot_beta_eff_heatmap(results, Path(str(base) + "_betaeff.pdf"))
    else:
        run_experiment(
            beta_in_list=sorted(args.beta_in),
            dry_run=args.dry_run,
        )

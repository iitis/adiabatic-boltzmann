"""
plot_hparam.py — visualise Optuna hyperparameter search results.

Three figures per study-combo (one per index.jsonl found):

  fig_optim_history.*    — objective vs trial number; running best highlighted
  fig_param_scatter.*    — each hyperparameter vs objective (scatter)
  fig_best_convergence.* — energy/spin training curves for top / mid / bottom trials

Usage (from project root):
    python scripts/plot_hparam.py
    python scripts/plot_hparam.py --search-dir results/hparam_search
    python scripts/plot_hparam.py --search-dir results/hparam_search/j1j2_1d/j1j2_gibbs_lsb_v1/N8_J20.0
    python scripts/plot_hparam.py --dpi 200 --top-k 8
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_METHOD_COLOR = {
    "gibbs": "#d62728",
    "lsb":   "#9467bd",
    "metropolis": "#1f77b4",
    "simulated_annealing": "#ff7f0e",
    "exchange": "#2ca02c",
    "veloxq_sa": "#17becf",
}
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Hyperparameters always present; we'll render extras (lsb_*) when available
_CORE_PARAMS = [
    ("learning_rate",  True,  "learning rate"),
    ("regularization", True,  "regularisation"),
    ("n_samples",      False, "n_samples"),
    ("n_hidden_alpha", False, "n_hidden / N"),
    ("n_warmup",       False, "n_warmup"),
    ("cg_tol",         True,  "CG tolerance"),
]
_LSB_PARAMS = [
    ("lsb_sigma", True,  "lsb σ"),
    ("lsb_steps", False, "lsb steps"),
    ("lsb_delta", False, "lsb δ"),
    ("cem_ema_alpha", False, "CEM α"),
]
_SA_PARAMS = [
    ("num_sweeps_per_step", True,  "sweeps / step"),
    ("start_temp",          False, "start temp (β⁻¹)"),
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_index(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _load_histories(combo_dir: Path) -> list[dict]:
    """Load all result_*.json training histories from a combo directory."""
    results = []
    for p in combo_dir.rglob("result_*.json"):
        try:
            d = json.load(open(p))
            if "history" in d and "config" in d:
                results.append(d)
        except Exception:
            pass
    return results


def _smooth(arr, w=15):
    if len(arr) < 2 * w:
        w = max(1, len(arr) // 4)
    # Edge-replicate before convolving to avoid zero-padding artifacts at boundaries.
    # mode="same" with zero-padding pulls negative energy toward 0 at the tail,
    # producing a spurious upward peak.
    padded = np.pad(arr, (w // 2, w // 2), mode="edge")
    smoothed = np.convolve(padded, np.ones(w) / w, mode="valid")
    return smoothed[: len(arr)]


def _converged_energy(arr: np.ndarray, window: int = 20,
                      rel_tol: float = 1e-3) -> float:
    """
    Mean energy over the stable tail.
    Convergence when |Δmean| / |mean| < rel_tol over W consecutive iterations.
    Falls back to last-20 % mean.
    """
    n = len(arr)
    if n < 2 * window:
        return float(arr.mean())
    for t in range(2 * window, n):
        prev = float(arr[t - 2 * window : t - window].mean())
        curr = float(arr[t - window : t].mean())
        if prev != 0 and abs(curr - prev) / abs(prev) < rel_tol:
            return float(arr[t - window :].mean())
    return float(arr[max(0, int(0.8 * n)) :].mean())


def _rel_err(energy, exact):
    if exact is None or exact == 0:
        return None
    return abs(energy - exact) / abs(exact)


def _running_best(values: list[float]) -> list[float]:
    best = float("inf")
    out = []
    for v in values:
        if v is not None and math.isfinite(v) and v < best:
            best = v
        out.append(best if math.isfinite(best) else None)
    return out


def _safe_log(v):
    return math.log10(v) if v is not None and v > 0 else None


# ── Figure 1: optimisation history ────────────────────────────────────────────

def fig_optim_history(records: list[dict], combo_label: str,
                      out: Path, dpi: int):
    """Objective vs trial number, per sampling method."""
    by_method: dict[str, list] = defaultdict(list)
    for r in records:
        method = r["params"].get("sampling_method", "?")
        by_method[method].append(r)

    methods = sorted(by_method)
    ncols = len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4.0, 3.5),
                              squeeze=False)
    fig.suptitle(f"Optuna optimisation history — {combo_label}",
                 fontsize=10, y=1.01)

    for ci, method in enumerate(methods):
        ax = axes[0, ci]
        recs = sorted(by_method[method], key=lambda r: r["trial"])
        trials  = [r["trial"] for r in recs]
        objs    = [r["objective"] for r in recs]
        running = _running_best(objs)

        color = _METHOD_COLOR.get(method, "#333")

        # All trials as scatter
        valid = [(t, o) for t, o in zip(trials, objs)
                 if o is not None and math.isfinite(o) and o > 0]
        if valid:
            tx, oy = zip(*valid)
            ax.scatter(tx, oy, color=color, alpha=0.4, s=18, zorder=2,
                       label="trial")

        # Running best
        rb_pairs = [(t, r) for t, r in zip(trials, running)
                    if r is not None and math.isfinite(r) and r > 0]
        if rb_pairs:
            rtx, rby = zip(*rb_pairs)
            ax.step(rtx, rby, where="post", color=color, linewidth=2.0,
                    label="running best", zorder=3)
            best_val = min(rby)
            ax.annotate(f"best={best_val:.2e}",
                        xy=(rtx[rby.index(best_val)], best_val),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

        ax.set_yscale("log")
        ax.set_xlabel("trial number")
        ax.set_ylabel("relative error  ε")
        ax.set_title(f"{method}  ({len(recs)} trials)")
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    fig.tight_layout()
    stem = "fig_optim_history"
    _save(fig, out, stem, dpi)


# ── Figure 2: hyperparameter scatter ──────────────────────────────────────────

def fig_param_scatter(records: list[dict], combo_label: str,
                      out: Path, dpi: int):
    """One scatter panel per hyperparameter; y = objective (log scale)."""
    methods = sorted({r["params"].get("sampling_method", "?") for r in records})
    method_color = {m: _METHOD_COLOR.get(m, _PALETTE[i % len(_PALETTE)])
                    for i, m in enumerate(methods)}

    # Which params are present?
    all_keys = set()
    for r in records:
        all_keys.update(r["params"].keys())

    params_to_plot = [p for p in _CORE_PARAMS if p[0] in all_keys]
    lsb_present    = [p for p in _LSB_PARAMS  if p[0] in all_keys]
    sa_present     = [p for p in _SA_PARAMS   if p[0] in all_keys]
    params_to_plot += lsb_present + sa_present

    ncols = 3
    nrows = math.ceil(len(params_to_plot) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.8),
                              squeeze=False)
    fig.suptitle(f"Hyperparameter sensitivity — {combo_label}", fontsize=10, y=1.01)

    for idx, (key, log_x, label) in enumerate(params_to_plot):
        ax = axes[idx // ncols, idx % ncols]
        for method in methods:
            recs = [r for r in records if r["params"].get("sampling_method") == method
                    and key in r["params"] and r["objective"] is not None
                    and math.isfinite(r["objective"]) and r["objective"] > 0]
            if not recs:
                continue
            xs = [float(r["params"][key]) for r in recs]
            ys = [r["objective"] for r in recs]
            color = method_color[method]
            ax.scatter(xs, ys, color=color, alpha=0.55, s=20,
                       label=method, zorder=2)

        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("ε" if idx % ncols == 0 else "")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # Hide empty axes
    for idx in range(len(params_to_plot), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # One shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=method_color[m], markersize=6, label=m)
               for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    stem = "fig_param_scatter"
    _save(fig, out, stem, dpi)


# ── Figure 3: best convergence curves ─────────────────────────────────────────

def fig_best_convergence(records: list[dict], histories: list[dict],
                         combo_label: str, out: Path, dpi: int,
                         top_k: int = 8,
                         energy_clip: float | None = None):
    """
    Top-k trials by objective, one panel per sampler.
    Shows E/N convergence curves with exact E₀/N dashed line.
    Lines are coloured by learning rate.

    energy_clip: if set, skip any run where |E/N| exceeds this value at any
                 iteration (removes diverged/exploded runs before selecting top-k).
    """
    methods = sorted({r["params"].get("sampling_method", "?") for r in records})

    _hist_lookup: dict[tuple, dict] = {}
    for h in histories:
        c = h["config"]
        key = (
            c.get("sampling_method"),
            round(float(c.get("learning_rate", 0)), 8),
            int(c.get("n_samples", 0)),
            int(c.get("seed", 0)),
            int(c.get("n_hidden", 0)),
        )
        _hist_lookup[key] = h

    def _find_history(trial_rec: dict) -> dict | None:
        p = trial_rec["params"]
        key = (
            p.get("sampling_method"),
            round(float(p.get("learning_rate", 0)), 8),
            int(p.get("n_samples", 0)),
            int(p.get("seed", 0)),
            int(trial_rec.get("n_hidden", 0)),
        )
        return _hist_lookup.get(key)

    def _load_eng(rec: dict) -> tuple[np.ndarray, int] | None:
        """Return (energy_array, N) for a trial, or None if missing/bad."""
        h = _find_history(rec)
        if h is None:
            return None
        N = h["config"].get("size", rec.get("N", 8))
        eng = np.array(
            [e for e in h["history"]["energy"] if e is not None],
            dtype=float)
        if eng.size == 0 or not np.all(np.isfinite(eng)):
            return None
        if energy_clip is not None and np.any(np.abs(eng / N) > energy_clip):
            return None
        return eng, N

    clip_suffix = f"  (|E/N| ≤ {energy_clip})" if energy_clip is not None else ""
    nrows = len(methods)
    fig, axes = plt.subplots(nrows, 1,
                              figsize=(5.0, nrows * 2.8),
                              squeeze=False)
    fig.suptitle(
        f"Best {top_k} convergence curves — {combo_label}{clip_suffix}\n"
        "Colour = learning rate",
        fontsize=10, y=1.01)

    all_lrs = [float(r["params"]["learning_rate"]) for r in records
               if "learning_rate" in r.get("params", {})]
    lr_min = min(all_lrs) if all_lrs else 1e-4
    lr_max = max(all_lrs) if all_lrs else 1e-1
    cmap = plt.cm.plasma

    def lr_color(lr):
        if lr_max == lr_min:
            return cmap(0.5)
        return cmap((math.log10(lr) - math.log10(lr_min)) /
                    (math.log10(lr_max) - math.log10(lr_min)))

    for ri, method in enumerate(methods):
        ax = axes[ri, 0]
        recs_m = [r for r in records
                  if r["params"].get("sampling_method") == method
                  and r["objective"] is not None
                  and math.isfinite(r["objective"])]

        # When clipping, further restrict to trials whose history passes the filter
        if energy_clip is not None:
            recs_m = [r for r in recs_m if _load_eng(r) is not None]

        if not recs_m:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#bbb")
            ax.set_title(method, fontsize=8)
            continue

        top_recs = sorted(recs_m, key=lambda r: r["objective"])[:top_k]
        n_clipped = (len([r for r in records
                          if r["params"].get("sampling_method") == method
                          and r["objective"] is not None
                          and math.isfinite(r["objective"])]) - len(recs_m)
                     if energy_clip is not None else 0)
        title = f"{method} — top {top_k}"
        if n_clipped:
            title += f"  ({n_clipped} runs clipped)"
        ax.set_title(title, fontsize=8)

        exact = None
        N = 8
        for rec in top_recs:
            result = _load_eng(rec)
            if result is None:
                continue
            eng, N = result
            if exact is None:
                h = _find_history(rec)
                if h is not None:
                    exact = h.get("exact_energy") or rec.get("exact_energy")

            sm = _smooth(eng / N)
            col = lr_color(float(rec["params"]["learning_rate"]))
            ax.plot(sm, color=col, linewidth=1.0, alpha=0.75)

        if exact is not None:
            ax.axhline(exact / N, color="#333", linestyle="--",
                       linewidth=0.85, zorder=0, label="exact E₀/N")

        if top_recs:
            best_obj = top_recs[0]["objective"]
            col_badge = ("#2ca02c" if best_obj < 0.01
                         else "#ff7f0e" if best_obj < 0.05 else "#d62728")
            ax.annotate(f"best ε={best_obj:.2e}",
                        xy=(0.97, 0.06), xycoords="axes fraction",
                        ha="right", fontsize=6.5, color=col_badge,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.8, ec="none"))

        ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        if method == "simulated_annealing":
            ax.set_ylim(-2, 2)
        ax.set_ylabel(f"{method}\n⟨E⟩/N", fontsize=8)
        if ri == nrows - 1:
            ax.set_xlabel("iteration")
        ax.legend(frameon=False, fontsize=7)

    sm_cb = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=matplotlib.colors.LogNorm(vmin=lr_min, vmax=lr_max))
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label("learning rate", fontsize=8)

    fig.tight_layout()
    stem = ("fig_best_convergence_clipped"
            if energy_clip is not None
            else "fig_best_convergence")
    _save(fig, out, stem, dpi)


# ── Figure 4: per-J₂ convergence (one row per J₂, one line per trial) ────────

def fig_j2_convergence(j2_data: dict, study_label: str, out: Path, dpi: int):
    """
    One row per J₂ value; each line = one trial's smoothed E/N convergence.

    j2_data: {j2_float -> {"records": list[dict], "histories": list[dict], "N": int}}
    Lines are coloured by learning rate (plasma, log scale).
    """
    j2_vals = sorted(j2_data.keys())
    if not j2_vals:
        return

    # Global lr range for a shared colorbar
    all_lrs = [
        float(r["params"]["learning_rate"])
        for group in j2_data.values()
        for r in group["records"]
        if "learning_rate" in r.get("params", {})
    ]
    if not all_lrs:
        return
    lr_min, lr_max = min(all_lrs), max(all_lrs)
    cmap = plt.cm.plasma

    def _lr_color(lr):
        if lr_max == lr_min:
            return cmap(0.5)
        t = (math.log10(lr) - math.log10(lr_min)) / (math.log10(lr_max) - math.log10(lr_min))
        return cmap(t)

    nrows = len(j2_vals)
    fig, axes = plt.subplots(nrows, 1,
                              figsize=(6.0, nrows * 2.2),
                              squeeze=False)
    fig.suptitle(
        f"Convergence per J₂ — {study_label}\n"
        "Each line = one trial · colour = learning rate",
        fontsize=10, y=1.01,
    )

    for ri, j2 in enumerate(j2_vals):
        ax = axes[ri, 0]
        group  = j2_data[j2]
        recs   = group["records"]
        hists  = group["histories"]
        N      = group["N"]

        # Build lookup: same key as in fig_best_convergence
        hist_lookup: dict[tuple, dict] = {}
        for h in hists:
            c = h["config"]
            key = (
                c.get("sampling_method"),
                round(float(c.get("learning_rate", 0)), 8),
                int(c.get("n_samples", 0)),
                int(c.get("seed", 0)),
                int(c.get("n_hidden", 0)),
            )
            hist_lookup[key] = h

        exact = None
        n_plotted = 0
        for rec in recs:
            p = rec["params"]
            key = (
                p.get("sampling_method"),
                round(float(p.get("learning_rate", 0)), 8),
                int(p.get("n_samples", 0)),
                int(p.get("seed", 0)),
                int(rec.get("n_hidden", 0)),
            )
            h = hist_lookup.get(key)
            if h is None:
                continue
            if exact is None:
                exact = h.get("exact_energy") or rec.get("exact_energy")

            eng = np.array(
                [e for e in h["history"]["energy"] if e is not None],
                dtype=float,
            )
            if eng.size == 0 or not np.all(np.isfinite(eng)):
                continue

            sm  = _smooth(eng / N)
            lr  = float(p["learning_rate"])
            obj = rec.get("objective")
            # dim failed/diverged trials
            alpha = 0.25 if (obj is None or not math.isfinite(obj)) else 0.65
            ax.plot(sm, color=_lr_color(lr), linewidth=0.85, alpha=alpha)
            n_plotted += 1

        if exact is not None:
            ax.axhline(exact / N, color="#333", linestyle="--",
                       linewidth=0.9, zorder=0, label="E₀/N")
            ax.legend(frameon=False, fontsize=7, loc="upper right")

        ax.set_ylim(-2, 2)
        ax.set_ylabel(f"J₂={j2:.3g}\n⟨E⟩/N", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        if n_plotted:
            ax.annotate(f"n={n_plotted}", xy=(0.97, 0.97),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.5, color="#555")
        if ri == nrows - 1:
            ax.set_xlabel("iteration")

    # Shared colorbar
    sm_cb = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=matplotlib.colors.LogNorm(vmin=lr_min, vmax=lr_max),
    )
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.02, shrink=0.6)
    cbar.set_label("learning rate", fontsize=8)

    fig.tight_layout()
    stem = f"fig_j2_convergence_N{_n_from_label(study_label)}"
    _save(fig, out, stem, dpi)


# ── Figure 5: VeloxQ SA signed-gap vs sweeps ─────────────────────────────────

def fig_sa_signed_gap(records: list[dict], combo_label: str,
                      out: Path, dpi: int):
    """Signed gap (⟨E⟩ − E_exact) vs num_sweeps_per_step for veloxq_sa trials.

    Each point is one trial.  Points below y=0 violate the variational bound.
    Colour encodes learning rate (plasma log scale).
    """
    sa_recs = [r for r in records
               if r["params"].get("sampling_method") == "veloxq_sa"
               and r.get("signed_gap") is not None
               and r["params"].get("num_sweeps_per_step") is not None]
    if not sa_recs:
        return

    all_lrs = [float(r["params"]["learning_rate"]) for r in sa_recs
               if "learning_rate" in r.get("params", {})]
    lr_min = min(all_lrs) if all_lrs else 1e-3
    lr_max = max(all_lrs) if all_lrs else 0.3
    cmap = plt.cm.plasma

    def _lr_color(lr):
        if lr_max == lr_min:
            return cmap(0.5)
        t = (math.log10(lr) - math.log10(lr_min)) / (math.log10(lr_max) - math.log10(lr_min))
        return cmap(t)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.suptitle(f"VeloxQ SA — signed gap vs sweeps  [{combo_label}]",
                 fontsize=10, y=1.01)

    n_subvar = 0
    for r in sa_recs:
        x = r["params"]["num_sweeps_per_step"]
        y = r["signed_gap"]
        lr = float(r["params"].get("learning_rate", lr_min))
        color = _lr_color(lr)
        marker = "v" if y < 0 else "o"
        ax.scatter(x, y, color=color, marker=marker, s=28, alpha=0.7,
                   zorder=3 if y < 0 else 2)
        if y < 0:
            n_subvar += 1

    ax.axhline(0, color="#333", linestyle="--", linewidth=1.0, zorder=1,
               label="variational bound (⟨E⟩ = E_exact)")

    ax.set_xscale("log")
    ax.set_xlabel("num_sweeps_per_step")
    ax.set_ylabel("⟨E⟩ − E_exact  (signed gap)")
    note = (f"{n_subvar}/{len(sa_recs)} sub-variational (v)"
            if n_subvar else f"0/{len(sa_recs)} sub-variational (ok)")
    ax.annotate(note, xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=7.5,
                color="#d62728" if n_subvar else "#2ca02c",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))
    ax.legend(frameon=False, fontsize=7.5)

    sm_cb = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=matplotlib.colors.LogNorm(vmin=lr_min, vmax=lr_max))
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label("learning rate", fontsize=8)

    fig.tight_layout()
    _save(fig, out, "fig_sa_signed_gap", dpi)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace(".", "p")


def _n_from_label(label: str) -> str:
    """Extract the trailing /N{n} from a study label like 'study/N8'."""
    part = label.rsplit("/", 1)[-1]  # e.g. "N8"
    return part[1:] if part.startswith("N") else part


def _save(fig, out: Path, stem: str, dpi: int):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  → {out / stem}.pdf")


# ── Discovery: find all combo dirs that have an index.jsonl ──────────────────

def _find_combo_dirs(search_root: Path) -> list[Path]:
    """Return all directories containing an index.jsonl (one Optuna study each)."""
    return sorted(p.parent for p in search_root.rglob("index.jsonl"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    _repo = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Plot Optuna hyperparameter search results")
    parser.add_argument(
        "--search-dir",
        default=str(_repo / "results" / "hparam_search"),
        help="Root to scan for index.jsonl files (default: results/hparam_search/)")
    parser.add_argument(
        "--output-dir",
        default=str(_repo / "plots" / "hparam_search"),
        help="Where to save figures (default: plots/hparam_search/)")
    parser.add_argument("--dpi",   type=int, default=150)
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of top trials to show (default: 8)")
    parser.add_argument("--energy-clip", type=float, default=10.0,
                        help="Max |E/N| allowed in the clipped convergence plot (default: 10)")
    args = parser.parse_args()

    search_root = Path(args.search_dir)
    out_root    = Path(args.output_dir)

    combo_dirs = _find_combo_dirs(search_root)
    if not combo_dirs:
        print(f"No index.jsonl found under {search_root}")
        return

    print(f"Found {len(combo_dirs)} combo(s) under {search_root}\n")

    for combo_dir in combo_dirs:
        records  = _load_index(combo_dir / "index.jsonl")
        if not records:
            continue

        histories = _load_histories(combo_dir)

        # Build a human-readable label from the path relative to search root
        rel = combo_dir.relative_to(search_root)
        combo_label = str(rel)
        out_dir = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        n_methods = len({r["params"].get("sampling_method") for r in records})
        print(f"[{combo_label}]  {len(records)} trials, "
              f"{n_methods} sampler(s), {len(histories)} histories")

        fig_optim_history(records, combo_label, out_dir, args.dpi)
        fig_param_scatter(records, combo_label, out_dir, args.dpi)
        fig_best_convergence(records, histories, combo_label, out_dir,
                             args.dpi, top_k=args.top_k)
        fig_best_convergence(records, histories, combo_label, out_dir,
                             args.dpi, top_k=args.top_k,
                             energy_clip=args.energy_clip)
        fig_sa_signed_gap(records, combo_label, out_dir, args.dpi)
        print()

    # ── Figure 4: per-J₂ convergence ─────────────────────────────────────────
    # Group combo_dirs that share the same study+N but differ in J₂.
    # Directory names follow the pattern  N{size}_J2{val}  (e.g. N8_J20.3).
    study_n_j2: dict[tuple, dict[float, Path]] = defaultdict(dict)
    for combo_dir in combo_dirs:
        name = combo_dir.name
        if "_J2" not in name:
            continue
        n_part, j2_str = name.split("_J2", 1)
        if not n_part.startswith("N"):
            continue
        try:
            N_val  = int(n_part[1:])
            j2_val = float(j2_str)
        except ValueError:
            continue
        study_n_j2[(combo_dir.parent, N_val)][j2_val] = combo_dir

    for (study_dir, N_val), j2_dirs in sorted(study_n_j2.items()):
        if len(j2_dirs) < 2:
            continue  # nothing interesting with a single J₂

        j2_data: dict[float, dict] = {}
        for j2_val, cdir in sorted(j2_dirs.items()):
            recs  = _load_index(cdir / "index.jsonl")
            if not recs:
                continue
            hists = _load_histories(cdir)
            N     = recs[0].get("N", N_val)
            j2_data[j2_val] = {"records": recs, "histories": hists, "N": N}

        if len(j2_data) < 2:
            continue

        rel         = study_dir.relative_to(search_root)
        study_label = f"{rel}/N{N_val}"
        out_dir     = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        j2_list = sorted(j2_data.keys())
        print(f"[{study_label}]  J₂ sweep: {j2_list}")
        fig_j2_convergence(j2_data, study_label, out_dir, args.dpi)
        print()

    print(f"All plots saved under  {out_root}/")


if __name__ == "__main__":
    main()

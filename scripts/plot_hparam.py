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
    return np.convolve(arr, np.ones(w) / w, mode="same")


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
    stem = f"fig_optim_history_{_slug(combo_label)}"
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
    params_to_plot += lsb_present

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
    stem = f"fig_param_scatter_{_slug(combo_label)}"
    _save(fig, out, stem, dpi)


# ── Figure 3: best / mid / worst convergence curves ───────────────────────────

def fig_best_convergence(records: list[dict], histories: list[dict],
                         combo_label: str, out: Path, dpi: int,
                         top_k: int = 8):
    """
    Top-k, middle-k and bottom-k trials by objective, per sampler.
    Shows E/N convergence curves with exact E₀/N dashed line.
    Lines are coloured by learning rate.
    """
    methods = sorted({r["params"].get("sampling_method", "?") for r in records})

    # Build a lookup: (method, lr_round, n_samples, seed) → history dict
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

    tiers = [(0, "top"), (1, "mid"), (2, "bottom")]
    nrows = len(methods)
    ncols = len(tiers)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.6, nrows * 2.6),
                              squeeze=False)
    fig.suptitle(
        f"Convergence: top / mid / bottom {top_k} trials — {combo_label}\n"
        "Colour = learning rate",
        fontsize=10, y=1.01)

    for ri, method in enumerate(methods):
        recs_m = [r for r in records
                  if r["params"].get("sampling_method") == method
                  and r["objective"] is not None
                  and math.isfinite(r["objective"])]
        if not recs_m:
            for ci in range(ncols):
                axes[ri, ci].text(0.5, 0.5, "no data", ha="center", va="center",
                                  transform=axes[ri, ci].transAxes, color="#bbb")
            continue

        recs_sorted = sorted(recs_m, key=lambda r: r["objective"])
        n = len(recs_sorted)
        mid_start = max(top_k, n // 2 - top_k // 2)

        tier_recs = {
            0: recs_sorted[:top_k],
            1: recs_sorted[mid_start : mid_start + top_k],
            2: recs_sorted[max(0, n - top_k):],
        }
        tier_labels = {0: f"top {top_k}", 1: f"mid {top_k}", 2: f"worst {top_k}"}

        # Colour by lr (log-normalised across all trials)
        all_lrs = [float(r["params"]["learning_rate"]) for r in recs_m]
        lr_min, lr_max = min(all_lrs), max(all_lrs)
        cmap = plt.cm.plasma

        def lr_color(lr):
            if lr_max == lr_min:
                return cmap(0.5)
            return cmap((math.log10(lr) - math.log10(lr_min)) /
                        (math.log10(lr_max) - math.log10(lr_min)))

        for ci, (tier_key, tier_lbl) in enumerate(tiers):
            ax = axes[ri, ci]
            tier_list = tier_recs[tier_key]
            ax.set_title(f"{method} — {tier_lbl}", fontsize=8)

            exact = None
            N_cell = recs_m[0].get("N") or recs_m[0]["phys_params"].get("J1") and 8

            for rec in tier_list:
                h = _find_history(rec)
                if h is None:
                    continue
                N = h["config"].get("size", rec.get("N", 8))
                if exact is None:
                    exact = h.get("exact_energy") or rec.get("exact_energy")

                eng = np.array(
                    [e for e in h["history"]["energy"] if e is not None],
                    dtype=float)
                if eng.size == 0 or not np.all(np.isfinite(eng)):
                    continue

                sm = _smooth(eng / N)
                lr = float(rec["params"]["learning_rate"])
                col = lr_color(lr)
                ax.plot(sm, color=col, linewidth=1.0, alpha=0.75)

                # Mark converged energy
                conv = _converged_energy(sm, window=20)
                #ax.plot(len(sm) - 1, conv, "o", color=col,
                        #markersize=3.5, zorder=4)

            if exact is not None and N:
                ax.axhline(exact / N, color="#333", linestyle="--",
                           linewidth=0.85, zorder=0, label="exact E₀/N")

            # Best objective badge
            if tier_list:
                best_obj = min(r["objective"] for r in tier_list)
                col_badge = ("#2ca02c" if best_obj < 0.01
                             else "#ff7f0e" if best_obj < 0.05 else "#d62728")
                ax.annotate(f"best ε={best_obj:.2e}",
                            xy=(0.97, 0.06), xycoords="axes fraction",
                            ha="right", fontsize=6.5, color=col_badge,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      alpha=0.8, ec="none"))

            ax.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            if ri == nrows - 1:
                ax.set_xlabel("iteration")
            if ci == 0:
                ax.set_ylabel(f"{method}\n⟨E⟩/N", fontsize=8)

    # Colorbar for lr
    sm_cb = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=matplotlib.colors.LogNorm(
            vmin=min(float(r["params"]["learning_rate"]) for r in records),
            vmax=max(float(r["params"]["learning_rate"]) for r in records)))
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.02, shrink=0.6)
    cbar.set_label("learning rate", fontsize=8)

    fig.tight_layout()
    stem = f"fig_best_convergence_{_slug(combo_label)}"
    _save(fig, out, stem, dpi)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace(".", "p")


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
                        help="Number of top/mid/bottom trials to show (default: 8)")
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
        print()

    print(f"All plots saved under  {out_root}/")


if __name__ == "__main__":
    main()

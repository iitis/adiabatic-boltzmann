#!/usr/bin/env python3
"""
plot_validation_convergence.py — per-seed convergence for a multi-seed
validation run of one fixed (winning) hyperparameter config, split by
whichever sibling "sweeps*" campaign directories are being compared (e.g.
sweeps100_v2 vs sweeps2000_v2 -- different VeloxQ SA num_sweeps budgets).

Companion to plot_hparam_search.py: that script searches over hyperparameters
and writes best_config.json for the winning trial; this script takes that
config and shows how it behaves across many seeds and across campaigns, so a
single lucky trial can be told apart from a config that's reliably good.
Nothing else in scripts/viz/ does this per-seed, campaign-split convergence
view: plot_hparam_search.py's own convergence panel overlays different
TRIALS within one search dir, not one fixed config across seeds/campaigns;
plot_ttc.py/plot_ite.py are campaign-aware but only for aggregate scaling
curves vs. N, not per-seed convergence at one fixed N.

Not hardcoded to N=128 or the "_v2" sweeps convention -- --size/--model/--h
and --campaigns are all free parameters, so the same script covers any
(model, N, h, sampler) validation run and any set of campaign directories.

Usage:
    # auto-load hyperparameters from a plot_hparam_search.py output dir
    python scripts/viz/plot_validation_convergence.py \\
        --model tfim_1d --size 128 --h 0.5 \\
        --hparam-dir plots/hparam_search/tfim_1d/veloxq_tfim/N128_h0.5 \\
        --campaigns sweeps100_v2 sweeps2000_v2

    # or specify hyperparameters explicitly (no hparam_search dir needed)
    python scripts/viz/plot_validation_convergence.py \\
        --model tfim_1d --size 128 --h 0.5 --sampler velox --sampling-method simulated_annealing \\
        --n-hidden 305 --lr 0.06528204230099208 --reg 4.815126214232072e-05 --n-samples 5400 \\
        --campaigns sweeps100_v2 sweeps2000_v2

Output: plots/validation/{model}_{size}/h{h}_{sampler}_{sampling_method}/
        (auto-derived per run; pass --out-dir to override)
"""

import argparse
import glob
import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy as np

MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"

CAMPAIGN_COLORS = ["#2a78d6", "#eb6834", "#008300", "#8a3fa0", "#bc5090"]


def style_axes(ax):
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def resolve_hparams(args):
    if args.hparam_dir:
        path = os.path.join(args.hparam_dir, "best_config.json")
        with open(path) as f:
            best = json.load(f)
        cfg = best["config"]
        print(f"loaded hyperparameters from {path}: err_per_spin={best['err_per_spin']:.3e}")
        return cfg["n_hidden"], cfg["learning_rate"], cfg["regularization"], cfg["n_samples"]
    missing = [name for name, v in [("--n-hidden", args.n_hidden), ("--lr", args.lr),
                                     ("--reg", args.reg), ("--n-samples", args.n_samples)] if v is None]
    if missing:
        raise SystemExit(f"either --hparam-dir or all of {missing} must be given")
    return args.n_hidden, args.lr, args.reg, args.n_samples


def load_campaign(results_dir, campaign, model, size, sampler, method, h, rbm, n_hidden, lr, reg, n_samples):
    pattern = os.path.join(results_dir, campaign, model, str(size), sampler, method, "result_*.json.gz")
    records = []
    for path in sorted(glob.glob(pattern)):
        try:
            with gzip.open(path, "rt") as f:
                r = json.load(f)
        except Exception as e:
            print(f"skipping unreadable {path}: {e}")
            continue
        c = r["config"]
        if (c.get("rbm") == rbm and c.get("n_hidden") == n_hidden
                and abs(c.get("h", h) - h) < 1e-9
                and abs(c["learning_rate"] - lr) < 1e-9
                and abs(c["regularization"] - reg) < 1e-9
                and c["n_samples"] == n_samples):
            records.append(r)
    return records


def plot(campaign_data, title_tag, out_dir, size):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, (campaign, recs) in enumerate(campaign_data.items()):
        color = CAMPAIGN_COLORS[i % len(CAMPAIGN_COLORS)]
        for r in recs:
            energies = np.array(r["history"]["energy"])
            exact = r["exact_energy"]
            err_per_spin = np.abs(energies - exact) / size
            ax1.plot(np.arange(1, len(err_per_spin) + 1), err_per_spin, color=color,
                     alpha=0.55, linewidth=1.0, zorder=2)
        ax1.plot([], [], color=color, linewidth=1.8, label=f"{campaign} (n={len(recs)})")

    ax1.set_yscale("log")
    ax1.set_xlabel("SR iteration")
    ax1.set_ylabel(r"Energy error per spin  $|E-E_\mathrm{exact}|/N$")
    ax1.set_title("Per-seed convergence")
    style_axes(ax1)
    ax1.legend(frameon=False, loc="upper right", fontsize=9)

    rng = np.random.RandomState(0)
    for i, (campaign, recs) in enumerate(campaign_data.items()):
        color = CAMPAIGN_COLORS[i % len(CAMPAIGN_COLORS)]
        errs = [abs(r["final_energy"] - r["exact_energy"]) / size for r in recs]
        xs = np.full(len(errs), i) + rng.uniform(-0.08, 0.08, len(errs))
        ax2.scatter(xs, errs, color=color, s=35, zorder=3, alpha=0.85)
        med = np.median(errs)
        ax2.hlines(med, i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax2.annotate(f"median={med:.2e}", (i, max(errs)), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, color=color, ha="center")

    ax2.set_yscale("log")
    ax2.set_xticks(range(len(campaign_data)))
    ax2.set_xticklabels(list(campaign_data.keys()))
    ax2.set_xlim(-0.6, len(campaign_data) - 0.4)
    ax2.set_ylabel(r"Final energy error per spin  $|E_\mathrm{final}-E_\mathrm{exact}|/N$")
    ax2.set_title("Final-error distribution per campaign")
    style_axes(ax2)

    fig.suptitle(f"Validation run convergence — {title_tag}", y=1.02)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"validation_convergence.{ext}")
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="tfim_1d", help="results/ subdirectory name, e.g. tfim_1d, tfim_2d, j1j2_1d")
    parser.add_argument("--size", "-N", type=int, required=True)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--rbm", default="full")
    parser.add_argument("--sampler", default="velox")
    parser.add_argument("--sampling-method", default="simulated_annealing")
    parser.add_argument("--campaigns", nargs="+", default=["sweeps100_v2", "sweeps2000_v2"],
                         help="sibling results/<campaign>/ directories to compare "
                              "(default: sweeps100_v2 sweeps2000_v2; pass any names/count)")
    parser.add_argument("--hparam-dir", default=None,
                         help="plot_hparam_search.py output dir to load best_config.json's hyperparameters from")
    parser.add_argument("--n-hidden", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--reg", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None,
                         help="default: plots/validation/{model}_{size}/h{h}_{sampler}_{sampling_method}/")
    args = parser.parse_args()

    n_hidden, lr, reg, n_samples = resolve_hparams(args)
    out_dir = args.out_dir or (
        f"plots/validation/{args.model}_{args.size}/h{args.h}_{args.sampler}_{args.sampling_method}"
    )

    campaign_data = {}
    for campaign in args.campaigns:
        recs = load_campaign(args.results_dir, campaign, args.model, args.size, args.sampler,
                              args.sampling_method, args.h, args.rbm, n_hidden, lr, reg, n_samples)
        if not recs:
            print(f"WARNING: no matching results found for campaign {campaign!r} -- skipping")
            continue
        campaign_data[campaign] = recs
        print(f"{campaign}: {len(recs)} seeds")

    if not campaign_data:
        raise SystemExit(
            "no matching results found in any campaign -- check --model/--size/--h/--rbm/"
            "--sampler/--sampling-method and the hyperparameters (--hparam-dir or --n-hidden/--lr/--reg/--n-samples)"
        )

    title_tag = f"{args.model} N={args.size} h={args.h}, nh={n_hidden} lr={lr:.3g} reg={reg:.2e} ns={n_samples}"
    plot(campaign_data, title_tag, out_dir, args.size)


if __name__ == "__main__":
    main()

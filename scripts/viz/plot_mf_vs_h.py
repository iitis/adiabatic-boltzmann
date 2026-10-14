#!/usr/bin/env python3
"""Trained RBM vs the best product state across h (data: scripts/exper/mf_vs_h.py).

Top row: final error per spin (E - E0)/N against h, one panel per N, median over seeds (markers) and
individual seeds (faint dots), with the mean-field error as a dashed reference curve.
Bottom row: the same as a ratio to the mean-field error (below 1 = better than a product state).
Every evaluated error is >= 0 (variational energies; checked), so no sign is hidden by the log scale.

  python scripts/viz/plot_mf_vs_h.py [--step 300] [--ceiling-step 3000] [--variants gibbs,gibbs_mfinit]
Output: plots/mf_vs_h/mf_vs_h_step<step>[_<variants>].{pdf,png} and a summary table on stdout
(median error per spin, relative error (E-E0)/|E0|, ratio to the mean-field error, number of seeds).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import setup_style

from ising import TransverseFieldIsing1D
from mean_field import mean_field_energy

DATA = REPO / "results" / "mf_vs_h"
OUT = REPO / "plots" / "mf_vs_h"
# variant -> (label, color, marker); the first three colors are the paper's (scripts/viz/paper_figures.py)
ARMS = {
    "exact_sr": (r"exact SR (ansatz ceiling)", "#008300", "s"),
    "exact": (r"SR, exact samples", "#2a78d6", "o"),
    "gibbs": (r"SR, persistent Gibbs", "#e87ba4", "^"),
    "gibbs_mfinit": (r"Gibbs, product-state start", "#7b3fbf", "v"),
    "gibbs_a2": (r"Gibbs, $M=2N$", "#d97a00", "D"),
}
DEFAULT_VARIANTS = ("exact_sr", "exact", "gibbs")
INK = "#0b0b0b"


def load(step, ceiling_step):
    """{(variant, N): {h: [(err per spin, |E0|/N), ...]}} at the requested checkpoint."""
    d = defaultdict(lambda: defaultdict(list))
    for f in DATA.glob("*.jsonl"):
        if f.name.endswith(".eval.jsonl"):
            continue
        ef = f.with_name(f.stem + ".eval.jsonl")  # deferred evaluations (mf_vs_h.py --evaluate-pending)
        deferred = {(e["seed"], e["step"]): e["err"] for e in map(json.loads, ef.read_text().splitlines())} \
            if ef.exists() else {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            want = ceiling_step if r["arm"] == "exact_sr" else step
            err = r.get("err", deferred.get((r["seed"], r["step"])))
            if r["step"] == want and err is not None:
                d[(r.get("variant", r["arm"]), r["N"])][r["h"]].append((err, abs(r["E0"]) / r["N"]))
    return d


def mf_curve(N, hs):
    out = []
    for h in hs:
        m = TransverseFieldIsing1D(N, h)
        out.append((mean_field_energy(m) - m.exact_ground_energy()) / N)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--ceiling-step", type=int, default=3000)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    args = ap.parse_args()
    variants = args.variants.split(",")
    d = load(args.step, args.ceiling_step)
    NS = sorted({N for (v, N) in d if v in variants})

    setup_style(fontsize=9, scale=2.0)
    fig, axes = plt.subplots(2, len(NS), figsize=(1.75 * len(NS), 3.6), sharex=True, sharey="row", squeeze=False)
    h_fine = np.linspace(0.45, 2.05, 81)
    print(f"median over seeds (SR step {args.step}; exact_sr step {args.ceiling_step})")
    for col, N in enumerate(NS):
        top, bot = axes[0, col], axes[1, col]
        mf_f = mf_curve(N, h_fine)
        top.plot(h_fine, mf_f, "--", color=INK, lw=1.0, label="best product state", zorder=1)
        bot.axhline(1.0, ls="--", color=INK, lw=1.0, zorder=1)
        for arm in variants:
            label, color, marker = ARMS[arm]
            cell = d.get((arm, N))
            if not cell:
                continue
            hs = np.array(sorted(cell))
            mf = mf_curve(N, hs)
            errs = [np.array([e for e, _ in cell[h]]) for h in hs]
            e0n = np.array([cell[h][0][1] for h in hs])
            med = np.array([np.median(e) for e in errs])
            for h, e, m in zip(hs, errs, mf):
                top.scatter(np.full(len(e), h), np.abs(e), s=6, color=color, alpha=0.35, lw=0, zorder=2)
                bot.scatter(np.full(len(e), h), np.abs(e) / m, s=6, color=color, alpha=0.35, lw=0, zorder=2)
            kw = dict(color=color, marker=marker, ms=4, lw=1.0, mec="white", mew=0.5, zorder=3)
            top.plot(hs, np.abs(med), label=label, **kw)
            bot.plot(hs, np.abs(med) / mf, **kw)
            for h, m, e, z in zip(hs, mf, med, e0n):
                print(f"  N={N:<3} {arm:<13} h={h:<5g} err/N={e:9.2e}  rel={100 * e / z:9.2e}%  "
                      f"MF={m:8.2e} (rel {100 * m / z:.3g}%)  ratio={e / m:9.3g}  n={len(cell[h])}")
        top.set_title(rf"$N={N}$")
        top.set_yscale("log")
        bot.set_yscale("log")
        bot.set_xlabel(r"$h$")
        for ax in (top, bot):
            ax.axvline(1.0, color="0.6", lw=0.6, zorder=0)
    axes[0, 0].set_ylabel(r"$(E-E_0)/N$")
    axes[1, 0].set_ylabel(r"error / mean-field error")
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes[0]), key=lambda hl: len(hl[1]))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, 0.925, rf"SR step {args.step}" + (rf"; exact SR step {args.ceiling_step}" if "exact_sr" in variants
                                                   else "") + r"; median over seeds, faint dots: single seeds",
             ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if variants == list(DEFAULT_VARIANTS) else "_" + "_".join(variants)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"mf_vs_h_step{args.step}{tag}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}/mf_vs_h_step{args.step}{tag}.pdf")


if __name__ == "__main__":
    main()

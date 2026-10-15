#!/usr/bin/env python3
"""Build notebooks/mf_vs_h.ipynb: ground-state methods for the 1D TFIM (product state, DMRG, RBM + VMC with
exact SR / exact samples / Gibbs) — what each does, the impact of each ingredient, error and cost comparisons.
Appendix: product state vs trained RBMs in detail.

    python scripts/viz/make_mf_vs_h_notebook.py            # write the notebook
    python scripts/viz/make_mf_vs_h_notebook.py --execute  # write and run it (outputs embedded)
"""
import argparse
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "mf_vs_h.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS = [
    md(r'''# Ground-state methods on two models: when is variational Monte Carlo competitive?

**Question.** For which models can a positive-amplitude RBM trained by variational Monte Carlo (VMC: SR updates,
samples from MCMC — or, later, a quantum annealer) beat the established alternatives — the best product state
(the floor), DMRG, and sign-free QMC (not run here)? Every method below gives a variational energy **and**
ground-state samples; exact energies are used only as references.

| model | Hamiltonian | why it is here | sizes | reference |
|---|---|---|---|---|
| **A. 1D TFIM chain** | $-\sum_i\sigma^z_i\sigma^z_{i+1} - h\sum_i\sigma^x_i$, periodic | unfrustrated, sign-free: DMRG's home ground; critical at $h=1$ | N = 8–128 | free fermions (exact) |
| **B. 2D triangular AF TFIM** | $+\sum_{\langle ij\rangle}\sigma^z_i\sigma^z_j - h\sum_i\sigma^x_i$, torus | frustrated but sign-free: DMRG limited by width, QMC cluster updates weakened; clock order for $h\lesssim1.6$ | 3×6, 6×6 | Lanczos (3×6); none at 6×6 |

| method | what it optimises | accuracy knob | samples |
|---|---|---|---|
| **product state** (mean field) | spin angles (uniform in A, one per site in B) | none — no entanglement | exact, i.i.d. |
| **DMRG / MPS** | matrix product state | bond dimension χ (χ = 1 = product state) | exact, i.i.d. |
| **RBM + exact SR** | RBM, exact gradients over all $2^N$ states | ansatz only — the ceiling ($N\le18$) | — |
| **RBM + SR, exact samples** | same, i.i.d. samples from $\lvert\psi\rvert^2$ by enumeration | + Monte Carlo noise | exact, i.i.d. |
| **RBM + SR, persistent Gibbs / Metropolis** | same, Markov-chain samples (200–20000 per step) | + MCMC autocorrelation | Markov chain |

**How to read it.** Sections 2 (model A) and 3 (model B) follow the same order — error against h, accuracy against
wall time, what limits the RBM — and section 4 puts both models side by side. Method diagnostics are in
Appendix A; the product-state analyses written for the reviewer's $h=0.5$ question are in Appendix B.

Data: model A RBM runs `results/mf_vs_h/` (`scripts/exper/mf_vs_h.py`) and `results/vmc_gpu/` (`vmc_gpu.py`,
more samples per step); model B `results/vmc_gpu/` (`scripts/exper/vmc_gpu.py`, `src/graph_tfim.py`); DMRG
`results/dmrg_baseline/{tfim1d,graph}.jsonl` (`scripts/validation/dmrg_{baseline,graph}.py`); product state
`src/mean_field.py`, `src/graph_tfim.py`. Re-run all cells after new runs land.'''),
    code(r'''import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
             if (p / "src" / "mean_field.py").exists()), None)
assert REPO is not None, "run this notebook from inside the adiabatic-boltzmann repository"
sys.path.insert(0, str(REPO / "src"))
import jax; jax.config.update("jax_enable_x64", True)
from ising import TransverseFieldIsing1D
from mean_field import mean_field_energy, mean_field_angle

DATA = REPO / "results" / "mf_vs_h"
pd.set_option("display.precision", 3)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"exact_sr": "#008300", "exact": "#2a78d6", "gibbs": "#e87ba4", "gibbs_mfinit": "#7b3fbf", "gibbs_a2": "#d97a00"}
MARKERS = {"exact_sr": "s", "exact": "o", "gibbs": "^", "gibbs_mfinit": "v", "gibbs_a2": "D"}
LABELS = {"exact_sr": "exact SR (ansatz ceiling)", "exact": "SR, exact samples", "gibbs": "SR, persistent Gibbs",
          "gibbs_mfinit": "Gibbs, product-state start", "gibbs_a2": "Gibbs, M = 2N"}'''),
    md(r'''## 1. Methods and data
### 1.1 Product state (mean field, χ = 1)
### 1.1 Product state (mean field, χ = 1)
Model A: best uniform product state, $\sin\alpha = h/2$ for $h\le 2$, $E_{MF}/N = -(1+h^2/4)$; x-polarised ($-h$) above.
No correlations at all, so its error measures how entangled the ground state is: tiny deep in either phase,
largest around (and, because mean field puts the transition at $h=2$, above) the critical point.
Model B: the best product state has one angle per site (three-sublattice pattern), found numerically.'''),
    code(r'''def e0_inf(h):
    """Thermodynamic-limit ground energy per spin (free fermions)."""
    k = np.linspace(0, np.pi, 200001)
    return -np.trapezoid(np.sqrt(1 + h * h - 2 * h * np.cos(k)), k) / np.pi

def mf_err(N, h):
    """Product-state error per spin, (E_MF - E0)/N."""
    m = TransverseFieldIsing1D(N, h)
    return (mean_field_energy(m) - m.exact_ground_energy()) / N

def e0_per_spin(N, h):
    return abs(TransverseFieldIsing1D(N, h).exact_ground_energy()) / N

def mf_rel(N, h):
    """Product-state relative error, (E_MF - E0)/|E0| (fraction, not %)."""
    return mf_err(N, h) / e0_per_spin(N, h)

rows = []
for h in (0.25, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
    emf = -(1 + h * h / 4) if h <= 2 else -h
    rows.append(dict(h=h, E0_inf=e0_inf(h), E_MF=emf, sz_MF=np.cos(mean_field_angle(TransverseFieldIsing1D(8, h))),
                     err_N8=mf_err(8, h), err_N16=mf_err(16, h), err_N64=mf_err(64, h), err_inf=emf - e0_inf(h),
                     rel_pct_N8=100 * mf_rel(8, h), rel_pct_N16=100 * mf_rel(16, h), rel_pct_N64=100 * mf_rel(64, h),
                     rel_pct_inf=100 * (emf - e0_inf(h)) / abs(e0_inf(h))))
mf_table = pd.DataFrame(rows).set_index("h")
mf_table  # err_*: error per spin (E_MF - E0)/N;  rel_pct_*: relative error (E_MF - E0)/|E0| in %'''),
    md(r'''### 1.2 DMRG (matrix product state, bond dimension χ)
Two-site DMRG (TeNPy) on the periodic chain, best of three starts (all up, all +x, previous χ's state — small χ
otherwise lands in start-dependent local minima); energy = ⟨H⟩ of the returned MPS. Each state yields exact i.i.d.
σᶻ samples by sequential conditional sampling (with a random global spin flip, since DMRG breaks the Z2 symmetry in
the ordered phase). In 1D the ground-state entanglement is bounded (it grows only as log N at $h=1$), so a modest χ
suffices; the error falls fastest with χ away from $h=1$.

Errors near 1e-14 per spin are the double-precision floor (relative error ~1e-14 of E0, i.e. round-off), shaded grey:
read them as "exact to all stored digits", not as a measured accuracy. Wall time = all three starts, single thread,
pure-Python TeNPy on a shared loaded machine (±2×).

Model B: the same DMRG on the 2D coupling graph (`dmrg_graph.py`): the torus laid out as a 1D MPS with long-range couplings, χ ladder up to 1024 — energies are upper bounds that improve with χ.'''),
    code(r'''dm = pd.DataFrame(map(json.loads, (REPO / "results" / "dmrg_baseline" / "tfim1d.jsonl").read_text().splitlines()))
dm = dm.drop_duplicates(["N", "h", "chi"], keep="last").sort_values(["N", "h", "chi"])  # last written wins
FLOOR = 1e-13  # per spin: round-off level for |E0|/N ~ 1-2 (relative ~1e-14)
dm["err_plot"] = dm.err.clip(lower=1e-15)  # round-off can make err ~ -1e-14; keep it on the log axis
CHI_CMAP = plt.get_cmap("Blues")
CHIS = sorted(dm.chi.unique())
chi_color = {c: CHI_CMAP(0.35 + 0.65 * i / max(len(CHIS) - 1, 1)) for i, c in enumerate(CHIS)}

def shade_floor(a):
    a.axhspan(1e-16, FLOOR, color="0.9", lw=0, zorder=0)

def vmc_at(arm, step):
    """Median error over the seeds that reached `step`, with their count, per (N, h)."""
    d = df[(df.arm == arm) & (df.step == step)].dropna(subset=["err"])
    return d.groupby(["N", "h"]).agg(err=("err", "median"), n=("err", "size"), wall_s=("wall_s", "median"))

print(len(dm), "DMRG points;", dm.groupby("N").size().to_dict(), "; winning start:", dm.start.value_counts().to_dict())
dm.pivot_table(index=["N", "h"], columns="chi", values="err")  # error per spin (E - E0)/N'''),
    md(r'''### 1.3 RBM + variational Monte Carlo (stochastic reconfiguration)
Full RBM with $M=N$ hidden units, random start, SR updates. The three arms (`exact_sr`, `exact`, `gibbs`) share the
ansatz, initialisation and SR settings and differ only in how expectations are estimated — see the table at the top.
Checkpoints at 100, 300, 1000, 3000 SR steps (phase 1: 100/300 for all; phase 2 extends some runs).
VMC wall time = cumulative training time to the checkpoint (includes JIT compilation, excludes evaluation;
hardware differs between runs) — orders of magnitude only.

Model B (and the extra model-A runs with more samples): `vmc_gpu.py`, persistent Gibbs or persistent Metropolis with global flips, 200–20000 samples per step, M = N or 2N, on the GPU.'''),
    code(r'''recs = []
for f in sorted(DATA.glob("*.jsonl")):
    if f.name.endswith(".eval.jsonl"):
        continue
    ef = f.with_name(f.stem + ".eval.jsonl")
    ev = {(e["seed"], e["step"]): e for e in map(json.loads, ef.read_text().splitlines())} if ef.exists() else {}
    for r in map(json.loads, f.read_text().splitlines()):
        e = r if r.get("E") is not None else ev.get((r["seed"], r["step"]), {})
        recs.append(dict(arm=r.get("variant", r["arm"]), N=r["N"], h=r["h"], seed=r["seed"], step=r["step"], E0=r["E0"],
                         err=e.get("err"), mf_err=r["mf_err"], absm=e.get("absm"), dw=e.get("dw"),
                         var=e.get("var"), train_err=(r["train_E_last10"] - r["E0"]) / r["N"], wall_s=r["wall_s"]))
df = pd.DataFrame(recs)
df["gain_over_mf"] = df.mf_err / df.err   # >1: better than the product state
df["rel_err"] = df.err * df.N / df.E0.abs()   # (E - E0)/|E0|
df["mf_rel_err"] = df.mf_err * df.N / df.E0.abs()
assert (df.err.dropna() >= 0).all(), "a variational energy below E0: evaluator or bookkeeping bug"
print(len(df), "checkpoints;", df.err.isna().sum(), "without evaluation")
df.groupby(["arm", "N", "step"]).size().unstack()  # checkpoints (seeds x h) per arm (variant), N and SR step'''),
    md(r'''### 1.4 Data for model B and for the extra model-A runs (`results/vmc_gpu/`, `results/dmrg_baseline/graph.jsonl`)'''),
    code(r'''tv = pd.DataFrame([json.loads(l) for f in sorted((REPO / "results" / "vmc_gpu").glob("*.jsonl"))
                    for l in f.read_text().splitlines()])
tv["err"] = (tv.E - tv.E0) / tv.N        # per spin; NaN where no exact energy (6x6)
tv["E_spin"] = tv.E / tv.N
tv["rel_err"] = (tv.E - tv.E0) / tv.E0.abs()
tv["vscore"] = tv.N * tv["var"] / tv.E ** 2  # reference-free quality (Wu et al.): 0 for an eigenstate
tv["label"] = tv.apply(lambda r: (f"exact SR lr{r.lr:g}" if r.arm == "exact_sr" else f"{r.arm} ns{r.ns}")
                       + (f" M={r.alpha}N" if r.alpha != 1 else "") + (" mf-start" if r.init == "mf" else ""), axis=1)
dg = pd.DataFrame(map(json.loads, (REPO / "results" / "dmrg_baseline" / "graph.jsonl").read_text().splitlines()))
for t_ in (tv, dg):  # cylinders get their own size labels, so the torus sections never mix the two geometries
    t_["size"] = np.where(t_.model == "tricyl", "cyl " + t_["size"].astype(str), t_["size"])
dg = dg.sort_values("E").drop_duplicates(["size", "h", "chi"]).sort_values(["size", "h", "chi"])  # best per chi
print(len(tv), "VMC checkpoints;", len(dg), "DMRG points; largest chi per h:",
      dg[dg["size"] == "6x6"].groupby("h").chi.max().to_dict())
tv.groupby(["model", "size", "label"]).size().unstack(0).fillna(0).astype(int)  # checkpoints per configuration'''),
    md(r'''## 2. Model A — 1D ferromagnetic TFIM chain
$H = -\sum_i\sigma^z_i\sigma^z_{i+1} - h\sum_i\sigma^x_i$, periodic, critical point $h=1$; exact $E_0$ from free fermions.
Unfrustrated and one-dimensional: the ground-state entanglement is bounded, so this is the model where DMRG is
expected to be hardest to beat.
### 2.1 The product state against h'''),
    code(r'''hh = np.linspace(0.02, 4, 400)
err = np.array([(-(1 + h * h / 4) if h <= 2 else -h) - e0_inf(h) for h in hh])
rel = err / np.abs([e0_inf(h) for h in hh])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(hh, err, "k"); ax[0].set_ylabel("product-state error per spin"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 0.3)
ax[1].plot(hh, 100 * rel, "k"); ax[1].set_ylabel("relative error [%]")
for a in ax:
    a.axvline(1, color="0.5", lw=0.8, ls=":"); a.axvline(2, color="0.5", lw=0.8, ls="--"); a.set_xlabel("h")
ax[1].text(1.02, 0.3, "exact $h_c=1$", color="0.4"); ax[1].text(2.05, 0.3, "MF $h_c=2$", color="0.4")
fig.suptitle(f"peak error {err.max():.3f}/spin at h={hh[err.argmax()]:.2f}; peak relative {100*rel.max():.1f}% at h={hh[rel.argmax()]:.2f}")
fig.tight_layout()'''),
    md(r'''### 2.2 Error of every method against h
VMC: median over the seeds that reached the given SR step (exact SR at step 3000); hollow markers = fewer than 3 seeds.
Phase-1 runs give all N at step 300; step 3000 only where phase 2 extended them.'''),
    code(r'''def dmrg_vs_h(step=300):
    Ns = sorted(dm.N.unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.4 * len(Ns), 3.8), sharey=True, squeeze=False)
    hf = np.linspace(0.45, 2.05, 81)
    vm = {arm: vmc_at(arm, st) for arm, st in (("exact_sr", 3000), ("gibbs", step))}
    for a, N in zip(axes[0], Ns):
        shade_floor(a)
        a.plot(hf, [mf_err(N, h) for h in hf], "k--", lw=1, label="product state (χ = 1)")
        for chi in CHIS:
            d = dm[(dm.N == N) & (dm.chi == chi)]
            a.plot(d.h, d.err_plot, "-o", color=chi_color[chi], ms=3, lw=1.2, label=f"DMRG χ = {chi}")
        for arm, v in vm.items():
            if N not in v.index.get_level_values("N"):
                continue
            g = v.loc[N]
            lab = f"{LABELS[arm]}, step {3000 if arm == 'exact_sr' else step}"
            a.plot(g.index, g.err, color=COLORS[arm], lw=1.6, label=lab)
            full = g.n >= 3
            a.scatter(g.index[full], g.err[full], color=COLORS[arm], marker=MARKERS[arm], s=30, ec="white", zorder=3)
            a.scatter(g.index[~full], g.err[~full], fc="white", ec=COLORS[arm], marker=MARKERS[arm], s=30, zorder=3)
        a.set_yscale("log"); a.set_title(f"N = {N}"); a.set_xlabel("h"); a.axvline(1, color="0.6", lw=0.6)
    axes[0, 0].set_ylabel("error per spin (E - E0)/N"); axes[0, 0].set_ylim(1e-15, 0.3)
    handles, labels = {}, []
    for a in axes[0]:
        for hd, lb in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lb, hd)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Gibbs SR at step {step}; grey band: double-precision floor")
    fig.tight_layout(rect=(0, 0.1, 1, 1))

dmrg_vs_h(300)'''),
    code(r'''dmrg_vs_h(3000)'''),
    md(r'''### 2.3 Accuracy vs wall time
The practical comparison: how much accuracy each method buys per second. Each VMC line is one seed through its
evaluated checkpoints (marker size grows with SR step; one seed per panel labelled 100/300/1k/3k); each DMRG point is
one χ. A line has 1–4 dots depending on how far that run has been extended and evaluated. Lower-left wins.'''),
    code(r'''STEP_SIZE = lambda step: 6 + 8 * np.log10(np.asarray(step) / 100 * 10)  # 100 -> 14, 300 -> 18, 1k -> 22, 3k -> 26

def cost_plot(hs=(0.5, 1.0, 1.5), arm="gibbs"):
    Ns = sorted(dm.N.unique())
    fig, axes = plt.subplots(len(hs), len(Ns), figsize=(3.3 * len(Ns), 3.0 * len(hs)), sharex=True, sharey=True,
                             squeeze=False)
    for i, h in enumerate(hs):
        for j, N in enumerate(Ns):
            a = axes[i, j]
            shade_floor(a)
            a.axhline(mf_err(N, h), color="k", ls="--", lw=1)
            d = dm[(dm.N == N) & (dm.h == h)]
            a.plot(d.t_dmrg, d.err_plot, "-o", color=chi_color[CHIS[-1]], ms=4, lw=1, label="DMRG")
            for _, r in d.iterrows():
                a.annotate(f"χ{int(r.chi)}", (r.t_dmrg, r.err_plot), fontsize=7, color="0.3",
                           xytext=(3, 2), textcoords="offset points")
            v = df[(df.arm == arm) & (df.N == N) & (df.h == h)].dropna(subset=["err"])
            longest = v.groupby("seed").size().idxmax() if len(v) else None
            for k, (s, g) in enumerate(v.groupby("seed")):
                g = g.sort_values("step")
                a.plot(g.wall_s, g.err, "-", color=COLORS[arm], lw=0.8, alpha=0.6)
                a.scatter(g.wall_s, g.err, s=STEP_SIZE(g.step), color=COLORS[arm], zorder=3, lw=0.5, ec="white",
                          label=f"{LABELS[arm]} (one line per seed)" if k == 0 else None)
                if s == longest:  # name the checkpoints on one run per panel
                    for _, r in g.iterrows():
                        a.annotate(f"{r.step:g}" if r.step < 1000 else f"{r.step / 1000:g}k", (r.wall_s, r.err),
                                   fontsize=7, color=COLORS[arm], xytext=(0, -9), textcoords="offset points",
                                   ha="center")
            a.set_xscale("log"); a.set_yscale("log")
            if i == 0: a.set_title(f"N = {N}")
            if j == 0: a.set_ylabel(f"h = {h:g}\nerror per spin")
            if i == len(hs) - 1: a.set_xlabel("wall time [s]")
    axes[0, 0].set_ylim(1e-15, 0.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("accuracy vs wall time (dashed: product state; grey: double-precision floor)")
    fig.tight_layout()

cost_plot()'''),
    md(r'''### 2.4 Summary table at the critical point h = 1
Per N: product-state error, DMRG error and time at each χ, and Gibbs-SR error (median over seeds, n in brackets)
and training time at steps 300 and 3000; exact SR (ceiling) where it exists.'''),
    code(r'''def summary(h=1.0):
    rows = {}
    for N in sorted(set(dm.N) | set(df.N)):
        r = {"product state": f"{mf_err(N, h):.1e}"}
        for _, d in dm[(dm.N == N) & (dm.h == h)].iterrows():
            r[f"DMRG χ={d.chi}"] = f"{d.err:.1e} ({d.t_dmrg:.0f} s)"
        for arm, st in (("gibbs", 300), ("gibbs", 3000), ("exact_sr", 3000)):
            v = vmc_at(arm, st)
            if (N, h) in v.index:
                x = v.loc[(N, h)]
                r[f"{arm} {st}"] = f"{x.err:.1e} ({x.wall_s:.0f} s, n={int(x.n)})"
        rows[N] = r
    t = pd.DataFrame(rows).T
    cols = ["product state"] + [f"DMRG χ={c}" for c in CHIS] + ["gibbs 300", "gibbs 3000", "exact_sr 3000"]
    return t.reindex(columns=[c for c in cols if c in t]).fillna("")

summary(1.0)'''),
    md(r'''### 2.5 Which DMRG χ matches the RBM's energy?
An *energy* equivalence, not a measure of the RBM's entanglement (equal energies do not imply equal states).
Per (N, h): median Gibbs error over the seeds that reached the step, entry `χ (n seeds)`. `MF/k` = worse than the
best product state (χ = 1) by a factor k; `>χmax` = better than every χ run at that (N, h). Periodic chains: an MPS needs a larger χ than with
open boundaries for the same accuracy.'''),
    code(r'''def chi_equivalent(step, arm="gibbs"):
    v = vmc_at(arm, step)
    out = {}
    for (N, h), r in v.iterrows():
        d = dm[(dm.N == N) & (dm.h == h)]
        if d.empty:
            continue
        ok = d[d.err <= r.err].chi
        mf = mf_err(N, h)
        chi = (f"MF/{r.err / mf:.2g}" if r.err > mf else str(ok.min()) if len(ok) else f">{d.chi.max()}")
        out[(N, h)] = f"{chi} ({int(r.n)})"
    return pd.Series(out, dtype=object).unstack() if out else pd.DataFrame()

for st in (300, 3000):
    print(f"Gibbs SR step {st}"); display(chi_equivalent(st))'''),
    md(r'''### 2.6 Scaling with system size N
Error per spin vs N at fixed h. A method whose error per spin stays flat scales; one whose error grows with N does not.
Product state and fixed-χ DMRG are (nearly) size-independent per spin; Gibbs-trained RBMs at a fixed step budget are not.'''),
    code(r'''def vs_N(hs=(0.5, 1.0, 1.5), chis=(4, 8, 16)):
    fig, axes = plt.subplots(1, len(hs), figsize=(4 * len(hs), 3.8), sharey=True, squeeze=False)
    Ns = sorted(set(df.N) | set(dm.N))
    for a, h in zip(axes[0], hs):
        shade_floor(a)
        a.plot(Ns, [mf_err(N, h) for N in Ns], "k--", lw=1, label="product state")
        for chi in chis:
            d = dm[(dm.h == h) & (dm.chi == chi)]
            if len(d):
                a.plot(d.N, d.err_plot, "-o", ms=4, color=chi_color.get(chi, "C0"), label=f"DMRG χ = {chi}")
        for arm, st, ls in (("gibbs", 300, "-"), ("gibbs", 3000, ":"), ("exact_sr", 3000, "-")):
            v = vmc_at(arm, st)
            if h in v.index.get_level_values("h"):
                g = v.xs(h, level="h")
                a.plot(g.index, g.err, ls, marker=MARKERS[arm], ms=5, color=COLORS[arm], mec="white",
                       label=f"{LABELS[arm]}, step {st}")
        a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks(Ns, [str(n) for n in Ns])
        a.set_title(f"h = {h:g}"); a.set_xlabel("N")
    axes[0, 0].set_ylabel("error per spin"); axes[0, 0].set_ylim(1e-15, 0.3)
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()

vs_N()'''),
    md(r'''### 2.7 More samples per step ("fair VMC", N = 64)
Error per spin against SR steps (left) and against cumulative training wall time (right), median over seeds, for
1k / 5k / 20k samples per step (`results/vmc_gpu`, GPU) and the 200-sample runs of section 3 (`results/mf_vs_h`,
CPU — wall times are not comparable across devices); dotted: DMRG at χ = 8, 16, 32; dashed: product state. h = 0.5
uses the product-state start (random starts get trapped at N = 64, section 2.9). The first GPU attempt at these
h = 0.5 runs started from a two-domain-wall state (the per-site mean-field optimiser missed the uniform optimum at
N = 64; fixed in `src/graph_tfim.py`, regression test added); those runs are archived in
`results/vmc_gpu/invalid_mfinit_domainwall/` and the runs shown are the reruns.'''),
    code(r'''def fair_vmc(N=64, hs=(1.0, 0.5)):
    fig, axes = plt.subplots(len(hs), 2, figsize=(11, 3.6 * len(hs)), squeeze=False)
    cmap = plt.get_cmap("plasma")
    for i, h in enumerate(hs):
        d = tv[(tv.model == "chain") & (tv.N == N) & (tv.h == h)]
        old = df[(df.arm == ("gibbs" if h >= 1 else "gibbs_mfinit")) & (df.N == N) & (df.h == h)].dropna(subset=["err"])
        series = [(f"ns 200 ({'mf-start' if h < 1 else 'random'}, CPU)", old)] + [
            (f"ns {ns}{' mf-start' if h < 1 else ''} (GPU)", x) for ns, x in d.groupby("ns") if not (ns == 200 and h < 1)]
        for k, (lab, x) in enumerate(series):
            g = x.groupby("step").agg(err=("err", "median"), wall=("wall_s", "median"))
            c = cmap(k / max(len(series), 2))
            axes[i, 0].plot(g.index, g.err, "o-", color=c, ms=4, label=lab)
            axes[i, 1].plot(g.wall, g.err, "o-", color=c, ms=4, label=lab)
        for chi in (8, 16, 32):
            r = dm[(dm.N == N) & (dm.h == h) & (dm.chi == chi)]
            for a in axes[i]:
                if len(r):
                    a.axhline(r.err.iloc[0], color=chi_color.get(chi, "b"), ls=":", lw=1)
        for a in axes[i]:
            a.axhline(mf_err(N, h), color="k", ls="--", lw=1); a.set_xscale("log"); a.set_yscale("log")
        axes[i, 0].set_ylabel(f"h = {h:g}: error per spin"); axes[i, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("SR step"); axes[-1, 1].set_xlabel("cumulative training wall time [s]")
    fig.tight_layout()

fair_vmc()'''),
    md(r'''### 2.8 What limits the RBM: ansatz → Monte Carlo noise → Markov chain
All three arms at the same SR step ($N\le16$, where all three exist). Factors > 1 are what each ingredient costs.'''),
    code(r'''def error_budget(step=300):
    med = {arm: df[(df.arm == arm) & (df.step == step)].dropna(subset=["err"]).groupby(["N", "h"]).err.median()
           for arm in ("exact_sr", "exact", "gibbs")}
    t = pd.DataFrame({"ansatz (exact SR)": med["exact_sr"], "exact samples": med["exact"], "Gibbs": med["gibbs"]}).dropna()
    t["× noise (exact/exact SR)"] = t["exact samples"] / t["ansatz (exact SR)"]
    t["× MCMC (Gibbs/exact)"] = t["Gibbs"] / t["exact samples"]
    return t

budget = error_budget(300)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for a, col, title in ((ax[0], "× noise (exact/exact SR)", "Monte Carlo noise (200 i.i.d. samples)"),
                      (ax[1], "× MCMC (Gibbs/exact)", "Markov chain instead of i.i.d. samples")):
    for N, g in budget.groupby(level="N"):
        a.plot(g.index.get_level_values("h"), g[col], "-o", ms=4, label=f"N = {N}")
    a.axhline(1, color="k", lw=0.8); a.set_yscale("log"); a.set_xlabel("h"); a.set_title(title, fontsize=10)
ax[0].set_ylabel("error factor at SR step 300"); ax[0].legend(fontsize=8)
fig.tight_layout()
budget.style.format("{:.2e}", subset=budget.columns[:3]).format("{:.1f}", subset=budget.columns[3:])'''),
    md(r'''Same three arms against h, with the product state as reference (rows: error per spin, relative error, ratio to the product state):'''),
    code(r'''def panel_plot(step=300, ceiling_step=3000, variants=("exact_sr", "exact", "gibbs")):
    Ns = sorted(df[df.arm.isin(variants)].N.unique())
    fig, axes = plt.subplots(3, len(Ns), figsize=(3.2 * len(Ns), 8.6), sharex=True, sharey="row", squeeze=False)
    hf = np.linspace(0.45, 2.05, 81)
    for j, N in enumerate(Ns):
        top, mid, bot = axes[0, j], axes[1, j], axes[2, j]
        top.plot(hf, [mf_err(N, h) for h in hf], "k--", lw=1, label="best product state")
        mid.plot(hf, [100 * mf_rel(N, h) for h in hf], "k--", lw=1)
        bot.axhline(1, color="k", ls="--", lw=1)
        for arm in variants:
            st = ceiling_step if arm == "exact_sr" else step
            d = df[(df.arm == arm) & (df.N == N) & (df.step == st)].dropna(subset=["err"])
            if d.empty:
                continue
            top.scatter(d.h, d.err, s=8, color=COLORS[arm], alpha=0.35, lw=0)
            mid.scatter(d.h, 100 * d.rel_err, s=8, color=COLORS[arm], alpha=0.35, lw=0)
            bot.scatter(d.h, d.err / d.mf_err, s=8, color=COLORS[arm], alpha=0.35, lw=0)
            g = d.groupby("h").agg(err=("err", "median"), rel=("rel_err", "median"), mf=("mf_err", "first"))
            kw = dict(color=COLORS[arm], marker=MARKERS[arm], ms=5, lw=1.2, mec="white")
            top.plot(g.index, g.err, label=LABELS[arm], **kw)
            mid.plot(g.index, 100 * g.rel, **kw)
            bot.plot(g.index, g.err / g.mf, **kw)
        top.set_title(f"N = {N}"); bot.set_xlabel("h")
        for a in (top, mid, bot):
            a.set_yscale("log"); a.axvline(1, color="0.6", lw=0.6)
    axes[0, 0].set_ylabel("error per spin (E - E0)/N"); axes[1, 0].set_ylabel("relative error (E - E0)/|E0| [%]")
    axes[2, 0].set_ylabel("error / product-state error")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"SR step {step}" + (f" (exact SR: step {ceiling_step})" if "exact_sr" in variants else "")
                 + "; lines: median over seeds, dots: single seeds")
    fig.tight_layout()

panel_plot(step=300)'''),
    md(r'''### 2.9 Product-state start and more hidden units
Product-state initialisation (`gibbs_mfinit`) and twice the hidden units (`gibbs_a2`) against the plain run,
at the latest checkpoint all variants share.'''),
    code(r'''V = ("gibbs", "gibbs_mfinit", "gibbs_a2")
have = df[df.arm.isin(V)].dropna(subset=["err"])
if have.arm.nunique() > 1:
    common = sorted(set.intersection(*(set(have[have.arm == v].step) for v in have.arm.unique())))
    panel_plot(step=common[-1], variants=tuple(v for v in V if v in set(have.arm)))
    display(have[have.step == common[-1]].groupby(["N", "h", "arm"]).gain_over_mf.median().unstack("arm"))
else:
    print("no variant runs evaluated yet")'''),
    md(r'''## 3. Model B — 2D triangular antiferromagnetic TFIM (frustrated, sign-free)
**Model.** Antiferromagnetic TFIM on the triangular lattice (torus), $H = +\sum_{\langle ij\rangle}\sigma^z_i\sigma^z_j - h\sum_i\sigma^x_i$
(`src/graph_tfim.py`). It is frustrated but still sign-free (the off-diagonal term is $-h\sigma^x$), so the ground state
is positive and the positive RBM can represent it; QMC cluster updates and DMRG (width) are both weakened in 2D.
Three-sublattice (clock) order for $h \lesssim 1.6$ (Isakov & Moessner 2003), paramagnet above; mean field puts the
transition at $h = 3$. Sizes: 3×6 ($N = 18$, exact energy by Lanczos) and 6×6 ($N = 36$, no exact energy).

**Methods.** Product state: best product state with one angle per site (not uniform on this lattice). DMRG:
`scripts/validation/dmrg_graph.py`, the torus as a 1D MPS with long-range MPO couplings, χ ladder up to 1024, CPU —
**upper bounds that improve with χ; the largest χ may still be running**. RBM: exact SR (ceiling), persistent Gibbs and
persistent Metropolis with global flips, with ns = 200 … 20000 samples per step, M = N or 2N; SR lr 0.08 / shift 0.05
unless noted (exact SR also with lr 0.05 / shift 1e-4). Final energies: enumeration ($N \le 18$) or GPU Metropolis
with global flips, 1024 chains (standard errors 2–7e-5 per spin at 6×6).

Runs: overnight job lists `scripts/exper/gpu14_{gpu,cpu}_jobs.txt`.'''),
    md(r'''### 3.1 Error of every method against h, several N (width-6 cylinders)
The counterpart of section 2.2 for model B, on cylinders of width 6 (periodic around, open along the length
L = 3, 6, 9, 12, i.e. N = 18–72) — the standard 2D DMRG geometry, used here for DMRG and VMC alike.
Reference: the exact energy for N = 18 (Lanczos); for larger N, DMRG extrapolated to χ → ∞ (linear fit of E
against the truncation error over the three largest χ; no extrapolation until three χ are done, then the best
DMRG energy is used). DMRG: parity-conserving two-site DMRG (`dmrg_graph.py`, χ ladder). VMC: persistent Gibbs,
1000 samples per step, M = N, median over seeds at the last checkpoint, bars = best and worst seed (hollow = one
seed); filled diamonds: M = 2N with 10⁴ samples where run; squares: exact SR (N = 18). Dashed: best product state.
Points below the reference (possible when the reference is an extrapolation) are shown on the symmetric-log axis.'''),
    code(r'''def dmrg_reference(D):
    """E(chi -> inf) per spin from a linear fit of E against the truncation error over the 3 largest chi, with an
    uncertainty |E_inf - E_best|; the best energy itself when fewer than 3 chi are available."""
    D = D.sort_values("chi")
    best = D.E_per_spin.min()
    if len(D) < 3 or D.max_trunc_err.tail(3).isna().any():
        return best, np.nan
    t, e = D.max_trunc_err.tail(3).values, D.E_per_spin.tail(3).values
    e_inf = min(np.polyfit(t, e, 1)[1], best)
    return e_inf, abs(best - e_inf)

def tri_reference(size, h):
    """Per-spin reference energy and its kind for model B: exact, DMRG-extrapolated, or best DMRG / VMC median."""
    v = tv[(tv["size"] == size) & (tv.h == h)]
    if len(v) and pd.notna(v.E0.iloc[0]):
        return v.E0.iloc[0] / v.N.iloc[0], "exact"
    D = dg[(dg["size"] == size) & (dg.h == h)]
    if len(D) and "E_exact" in D and D.E_exact.notna().any():  # DMRG records carry the Lanczos energy for N <= 20
        return D.E_exact.dropna().iloc[0] / D.N.iloc[0], "exact"
    if len(D) >= 3:
        return dmrg_reference(D)[0], "DMRG χ→∞"
    cands = [D.E_per_spin.min()] if len(D) else []
    if len(v):
        last = v[v.step == v.groupby("label").step.transform("max")]
        cands.append(last.groupby("label").E_spin.median().min())
    return (min(cands), "lowest DMRG / VMC median") if cands else (np.nan, "none")

def cyl_vs_h(sizes=("cyl 3x6", "cyl 6x6", "cyl 9x6", "cyl 12x6")):
    sizes = [s for s in sizes if ((tv["size"] == s).any() or (dg["size"] == s).any())]
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes), 3.9), sharey=True, squeeze=False)
    chis = sorted(dg[dg["size"].isin(sizes)].chi.unique())
    ccol = {c: CHI_CMAP(0.3 + 0.7 * i / max(len(chis) - 1, 1)) for i, c in enumerate(chis)}
    for a, size in zip(axes[0], sizes):
        hs = sorted(set(tv[tv["size"] == size].h) | set(dg[dg["size"] == size].h))
        ref = {h: tri_reference(size, h) for h in hs}
        D = dg[dg["size"] == size]
        N = int(D.N.iloc[0]) if len(D) else int(tv[tv["size"] == size].N.iloc[0])
        mf = D.groupby("h").E_mf.first() / N if len(D) else tv[tv["size"] == size].groupby("h").E_mf.first() / N
        a.plot(mf.index, [mf[h] - ref[h][0] for h in mf.index], "k--", lw=1, label="product state")
        for chi in chis:
            x = D[D.chi == chi].sort_values("h")
            if len(x):
                a.plot(x.h, [e - ref[h][0] for e, h in zip(x.E_per_spin, x.h)], "-o", color=ccol[chi], ms=3, lw=1.1,
                       label=f"DMRG χ = {chi}")
        v = tv[tv["size"] == size]
        v = v[v.step == v.groupby("run").step.transform("max")]
        for lab, mk, col in (("gibbs ns1000", "^", COLORS["gibbs"]), ("gibbs ns10000 M=2N", "D", COLORS["gibbs_a2"]),
                             ("exact SR lr0.08", "s", COLORS["exact_sr"])):
            x = v[v.label == lab]
            if x.empty:
                continue
            g = x.groupby("h").E_spin.agg(["median", "min", "max", "size"])
            r = np.array([ref[h][0] for h in g.index])
            a.plot(g.index, g["median"] - r, "-", color=col, lw=1.4)
            full = (g["size"] >= 2).values
            a.errorbar(g.index, g["median"] - r, yerr=[g["median"] - g["min"], g["max"] - g["median"]], fmt="none",
                       ecolor=col, capsize=3, lw=1)
            a.scatter(g.index[full], (g["median"] - r)[full], marker=mk, color=col, s=36, ec="white", zorder=3,
                      label=f"VMC {lab}")
            a.scatter(g.index[~full], (g["median"] - r)[~full], marker=mk, fc="white", ec=col, s=36, zorder=3)
        kinds = sorted({k for _, k in ref.values()})
        a.set_title(f"N = {N} ({size.split()[-1]})\nref: {', '.join(kinds)}", fontsize=9)
        a.set_yscale("symlog", linthresh=1e-8); a.set_xlabel("h"); a.axvline(1.6, color="0.7", lw=0.6)
    axes[0, 0].set_ylabel("error per spin (E − E_ref)/N  [symlog]"); axes[0, 0].set_ylim(-1e-6, 0.5)
    leg = {}
    for a in axes[0]:
        for hd, lb in zip(*a.get_legend_handles_labels()):
            leg.setdefault(lb, hd)
    fig.legend(leg.values(), leg.keys(), loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("triangular AF TFIM on width-6 cylinders: error of every method against h")
    fig.tight_layout(rect=(0, 0.1, 1, 1))

cyl_vs_h()'''),
    md(r'''### 3.2 Torus 3×6: error against h (exact reference)
Error per spin and relative error at the last checkpoint (3000 steps; M = 2N exact SR: 1000 steps), median over seeds;
dashed: best product state. Note the ceiling (exact SR) is ~7e-4 per spin in the ordered phase, about 50× higher
than for the 1D chain at a similar N: the frustrated state is harder for the M = N RBM itself.'''),
    code(r'''def tri_small(size="3x6"):
    d = tv[(tv["size"] == size)]
    last = d.groupby("run").step.transform("max")
    d = d[d.step == last]
    g = d.groupby(["label", "h"]).agg(err=("err", "median"), rel=("rel_err", "median"), n=("err", "size")).reset_index()
    mf = d.groupby("h").apply(lambda x: ((x.E_mf - x.E0) / x.N).iloc[0], include_groups=False)
    mfr = d.groupby("h").apply(lambda x: ((x.E_mf - x.E0) / x.E0.abs()).iloc[0], include_groups=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, col, ref in ((ax[0], "err", mf), (ax[1], "rel", 100 * mfr)):
        a.plot(ref.index, ref.values, "k--", lw=1, label="best product state")
        for lab, x in g.groupby("label"):
            y = x[col] * (100 if col == "rel" else 1)
            a.plot(x.h, y, "o-", ms=4, lw=1, label=lab)
        a.set_yscale("log"); a.set_xlabel("h"); a.axvline(1.6, color="0.7", lw=0.6)
    ax[0].set_ylabel("error per spin (E - E0)/N"); ax[1].set_ylabel("relative error (E - E0)/|E0| [%]")
    ax[1].legend(fontsize=7, ncol=2); fig.suptitle(f"triangular {size}, last checkpoint, median over seeds")
    fig.tight_layout()
    return g.pivot(index="label", columns="h", values="err")

tri_small().style.format("{:.2e}", na_rep="")  # error per spin'''),
    md(r'''### 3.3 Torus 6×6: error against h, against DMRG and the product state
No exact energy at N = 36. Each method is shown relative to a reference E_ref = the lowest of the DMRG energies
and the VMC **medians** (a single lucky seed never sets the reference): ΔE/N = (E − E_ref)/N. VMC points are the
median over seeds at step 3000, with bars spanning the best and worst seed (n seeds in the table). DMRG energies are
upper bounds that decrease with χ; while the CPU jobs run, the largest χ may be missing, so a VMC point below DMRG
means "below DMRG at the χ reached so far". Evaluation error bars (2–7e-5 per spin) are below the marker size.'''),
    code(r'''def tri_big(size="6x6", step=3000):
    d = tv[(tv["size"] == size) & (tv.step == step) & (tv.model == "tri")]
    D = dg[(dg["size"] == size) & (dg.model == "tri")]
    vm = d.groupby(["label", "h"]).E_spin.agg(["median", "min", "max", "size"]).reset_index()
    ref = pd.concat([vm.groupby("h")["median"].min(), D.groupby("h").E_per_spin.min()], axis=1).min(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for chi, x in D.groupby("chi"):
        ax.plot(x.h, x.E_per_spin.values - ref.loc[x.h].values, "-", color=CHI_CMAP(0.3 + 0.7 * np.log2(chi) / 10),
                lw=1.2, marker=".", label=f"DMRG χ={chi}")
    for k, (lab, x) in enumerate(vm.groupby("label")):
        dx = 0.02 * (k - 1)  # small horizontal offset so the configurations do not overlap
        r = ref.loc[x.h].values
        ax.errorbar(x.h + dx, x["median"] - r, yerr=[x["median"] - x["min"], x["max"] - x["median"]], fmt="o", ms=7,
                    mec="k", capsize=3, lw=1, label=f"VMC {lab} (median, min–max over seeds)")
    mf = d.groupby("h").apply(lambda x: (x.E_mf / x.N).iloc[0], include_groups=False)
    ax.plot(mf.index, mf.values - ref.loc[mf.index].values, "k--", lw=1, label="best product state")
    ax.set_yscale("symlog", linthresh=1e-6); ax.set_xlabel("h"); ax.set_ylabel("(E − E_ref)/N  [symlog]")
    ax.axvline(1.6, color="0.7", lw=0.6); ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"triangular {size} torus: distance to E_ref (lowest DMRG energy or VMC median)"); fig.tight_layout()
    t = pd.DataFrame({"product state": mf, "DMRG best (χ)": D.groupby("h").apply(
        lambda x: f"{x.E_per_spin.min():.6f} ({x.chi[x.E_per_spin.idxmin()]})", include_groups=False)})
    for lab, x in vm.groupby("label"):
        t[f"VMC {lab}"] = pd.Series([f"{m:.6f} (n={int(n)})" for m, n in zip(x["median"], x["size"])], index=x.h)
    return t

tri_big()  # E/N per method; VMC = median over seeds at step 3000'''),
    md(r'''### 3.4 Accuracy vs wall time (torus 3×6, 6×6, and the cylinders)
The practical comparison, as in section 2.2: how much accuracy each method buys per second. Reference per panel:
the exact energy (3×6), else DMRG extrapolated to χ → ∞ (cylinders, once three χ are done), else the lowest of the
DMRG energies and VMC medians (torus 6×6 — "best found", not exact; a single VMC seed never sets it). First figure:
torus; second: width-6 cylinders. Each VMC line is one seed through its
checkpoints (marker size grows with SR step 300 → 1k → 3k); each DMRG point is one χ, at the **cumulative**
time of the χ ladder up to it (every χ is warm-started from the previous one, three starts each). Dashed: best
product state. Lower-left wins.

Hardware: VMC on one GPU (TITAN RTX, two jobs sharing it; training time only, evaluation excluded); DMRG on CPU
(TeNPy, 4 threads at 6×6, 2 at 3×6), both on a shared machine — treat times as ±2×.'''),
    code(r'''TRI_COLORS = {"gibbs ns200": "#f4a3c0", "gibbs ns1000": "#e87ba4", "gibbs ns5000": "#b8326b",
              "gibbs ns10000 M=2N": "#d97a00", "mh ns1000": "#9ecae1", "mh ns5000": "#4a90c2",
              "mh ns10000 M=2N": "#1f4e8c"}

def tri_cost(hs=(0.5, 1.0, 1.5, 2.0, 3.0), sizes=("3x6", "6x6")):
    fig, axes = plt.subplots(len(hs), len(sizes), figsize=(5.6 * len(sizes), 2.9 * len(hs)), squeeze=False,
                             sharex=True)
    for i, h in enumerate(hs):
        for j, size in enumerate(sizes):
            a = axes[i, j]
            v = tv[(tv["size"] == size) & (tv.h == h) & (tv.arm != "exact_sr")]
            D = dg[(dg["size"] == size) & (dg.h == h)].sort_values("chi")
            if v.empty and D.empty:
                a.set_visible(False); continue
            N = (v.N.iloc[0] if len(v) else D.N.iloc[0])
            ref, kind = tri_reference(size, h)
            emf = (v.E_mf.iloc[0] / N) if len(v) else D.E_mf.iloc[0] / N
            a.axhline(emf - ref, color="k", ls="--", lw=1)
            if len(D):
                t = D.t_dmrg.cumsum()
                a.plot(t, D.E_per_spin - ref, "-o", color=CHI_CMAP(0.85), ms=4, lw=1.2, label="DMRG (χ ladder)")
                for tt, e, chi in zip(t, D.E_per_spin - ref, D.chi):
                    a.annotate(f"χ{chi}", (tt, e), fontsize=7, color="0.3", xytext=(3, 2), textcoords="offset points")
            for lab, x in v.groupby("label"):
                c = TRI_COLORS.get(lab, "0.5")
                for k, (s, g) in enumerate(x.groupby("seed")):
                    g = g.sort_values("step")
                    a.plot(g.wall_s, g.E_spin - ref, "-", color=c, lw=0.8, alpha=0.7)
                    a.scatter(g.wall_s, g.E_spin - ref, s=STEP_SIZE(g.step), color=c, ec="white", lw=0.5, zorder=3,
                              label=f"VMC {lab}" if k == 0 else None)
            a.set_xscale("log"); a.set_yscale("symlog", linthresh=1e-6); a.set_ylim(-1e-7, 0.5)
            if i == 0:
                a.set_title(f"{size}: error vs {kind}", fontsize=9)
            if j == 0: a.set_ylabel(f"h = {h:g}\nΔE per spin [symlog]")
            if i == len(hs) - 1: a.set_xlabel("wall time [s]")
    leg = {}
    for a in axes.flat:  # one legend for every configuration that appears in any panel
        for hd, lb in zip(*a.get_legend_handles_labels()):
            leg.setdefault(lb, hd)
    fig.legend(leg.values(), leg.keys(), loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("triangular AF TFIM: accuracy vs wall time (dashed: best product state)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

tri_cost()'''),
    code(r'''tri_cost(hs=(0.5, 1.0, 1.5, 3.0), sizes=("cyl 3x6", "cyl 6x6", "cyl 9x6", "cyl 12x6"))'''),
    md(r'''### 3.5 What limits the RBM: samples per step and sampler (torus 3×6, h = 1.0 and 1.5)
Error per spin against samples per step at 300 / 1000 / 3000 SR steps; horizontal lines: exact SR (the ns → ∞ limit
with the same optimiser, lr 0.08 / shift 0.05) at the same step. Where the sampled runs reach the line, sampling noise
no longer limits the error — the ansatz and the optimiser do.'''),
    code(r'''def ns_scan(size="3x6", hs=(1.0, 1.5)):
    fig, axes = plt.subplots(1, len(hs), figsize=(5.2 * len(hs), 3.8), sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for a, h in zip(axes[0], hs):
        for i, st in enumerate((300, 1000, 3000)):
            c = cmap(i / 2)
            for arm, mk in (("gibbs", "o"), ("mh", "s")):
                d = tv[(tv["size"] == size) & (tv.h == h) & (tv.arm == arm) & (tv.alpha == 1) & (tv.step == st)]
                g = d.groupby("ns").err.median()
                if len(g):
                    a.plot(g.index, g.values, mk + "-", color=c, ms=5, lw=1, mfc=c if arm == "gibbs" else "white",
                           label=f"{arm}, step {st}")
            e = tv[(tv["size"] == size) & (tv.h == h) & (tv.arm == "exact_sr") & (tv.lr == 0.08) & (tv.step == st)].err
            if len(e):
                a.axhline(e.median(), color=c, ls="--", lw=1)
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"{size}, h = {h:g}"); a.set_xlabel("samples per SR step")
    axes[0, 0].set_ylabel("error per spin"); axes[0, 0].legend(fontsize=7)
    fig.tight_layout()

ns_scan()'''),
    md(r'''## 4. Both models side by side: can VMC match DMRG for the same time?
For every (model, size, h): the best VMC configuration by its **median over seeds** at its last checkpoint (per-spin error against
$E_0$, or at 6×6 against the best energy any method found) with its training wall time, and the DMRG χ ladder:
the smallest χ whose energy is at least as good and the cumulative DMRG time to reach it. "VMC" in the last
column means VMC reached its accuracy faster than DMRG — or DMRG's largest χ so far did not reach it at all.
Times: VMC training only (GPU for `vmc_gpu` runs, CPU for most 200-sample runs), DMRG on CPU; ±2× at best.'''),
    code(r'''def compare_models():
    rows = []

    def ladder(D):
        return D.sort_values("chi").assign(t_cum=lambda x: x.t_dmrg.cumsum())

    def match(D, key, target):
        hit = D[D[key] <= target]
        return (int(hit.chi.iloc[0]), float(hit.t_cum.iloc[0])) if len(hit) else (np.nan, np.nan)

    for N in (16, 32, 64, 128):  # model A
        for h in (0.5, 1.0, 1.5):
            D = ladder(dm[(dm.N == N) & (dm.h == h)])
            a = df[(df.N == N) & (df.h == h) & df.arm.str.startswith("gibbs")][["err", "wall_s", "arm", "step"]]
            b = tv[(tv.model == "chain") & (tv.N == N) & (tv.h == h) & (tv.arm != "exact_sr")]
            v = pd.concat([a.assign(cfg=a.arm + " ns200"),
                           b[["err", "wall_s", "label", "step"]].rename(columns={"label": "cfg"})]).dropna(subset=["err"])
            if D.empty or v.empty:
                continue
            v = v[v.step == v.groupby("cfg").step.transform("max")]
            med = v.groupby("cfg").agg(err=("err", "median"), wall_s=("wall_s", "median"), step=("step", "max"))
            best = med.loc[med.err.idxmin()].copy(); best["cfg"] = med.err.idxmin()
            chi, t = match(D, "err", best.err)
            rows.append(dict(model="A chain", size=f"N={N}", h=h, product_state=mf_err(N, h), vmc=best.err,
                             vmc_config=f"{best.cfg} @{int(best.step)}", vmc_s=best.wall_s, dmrg_chi_to_match=chi,
                             dmrg_s_to_match=t, dmrg_best=D.err.min(), dmrg_chi_max=int(D.chi.max())))
    for size in ("3x6", "6x6", "cyl 3x6", "cyl 6x6", "cyl 9x6", "cyl 12x6"):  # model B: torus, then cylinders
        for h in sorted(tv[tv.model.isin(["tri", "tricyl"]) & (tv["size"] == size)].h.unique()):
            v = tv[(tv["size"] == size) & (tv.h == h) & (tv.arm != "exact_sr")]
            D = ladder(dg[(dg["size"] == size) & (dg.h == h)])
            if v.empty or D.empty:
                continue
            N, e0 = v.N.iloc[0], v.E0.iloc[0]
            last = v[v.step == v.groupby("label").step.transform("max")]
            med = last.groupby("label").agg(E_spin=("E_spin", "median"), wall_s=("wall_s", "median"),
                                            step=("step", "max"), n=("seed", "size"))
            best = med.loc[med.E_spin.idxmin()].copy(); best["label"] = med.E_spin.idxmin()
            ref = tri_reference(size, h)[0]
            chi, t = match(D, "E_per_spin", best.E_spin)
            rows.append(dict(model="B triangular", size=size, h=h, product_state=v.E_mf.iloc[0] / N - ref,
                             vmc=best.E_spin - ref, vmc_config=f"{best.label} @{int(best.step)}", vmc_s=best.wall_s,
                             dmrg_chi_to_match=chi, dmrg_s_to_match=t, dmrg_best=D.E_per_spin.min() - ref,
                             dmrg_chi_max=int(D.chi.max())))
    t = pd.DataFrame(rows).set_index(["model", "size", "h"])
    t["faster_to_vmc_accuracy"] = np.where(t.dmrg_s_to_match.isna(), "VMC (beyond DMRG χmax)",
                                           np.where(t.dmrg_s_to_match < t.vmc_s, "DMRG", "VMC"))
    return t

compare_models().style.format({"product_state": "{:.1e}", "vmc": "{:.1e}", "dmrg_best": "{:.1e}", "vmc_s": "{:.0f}",
                               "dmrg_s_to_match": "{:.0f}", "dmrg_chi_to_match": "{:.0f}"}, na_rep="—")'''),
    md(r'''## Appendix A — method diagnostics
### A.1 DMRG: payoff of the bond dimension χ (model A)
Error per spin vs χ, one line per N. The slope is the payoff of entanglement: steepest away from $h=1$, flattest at
the critical point and for larger N (periodic chains also need a larger χ than open ones).'''),
    code(r'''hs_chi = (0.5, 1.0, 1.5)
fig, axes = plt.subplots(1, len(hs_chi), figsize=(4 * len(hs_chi), 3.4), sharey=True)
for a, h in zip(axes, hs_chi):
    shade_floor(a)
    for i, (N, d) in enumerate(dm[dm.h == h].groupby("N")):
        line, = a.plot(d.chi, d.err_plot, "-o", ms=4, label=f"N = {N}")
        a.plot([1], [mf_err(N, h)], "o", mfc="white", color=line.get_color())  # χ = 1: product state
    a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("χ (open circle: χ = 1, product state)")
    a.set_xticks([1] + CHIS, [str(c) for c in [1] + CHIS])
    a.set_title(f"h = {h:g}")
axes[0].set_ylabel("error per spin"); axes[0].set_ylim(1e-15, 0.3); axes[0].legend(fontsize=8)
fig.tight_layout()'''),
    md(r'''### A.2 RBM: training length (model A)
Median error vs SR step, one column per N, one line per h; dashed: product state. Then the raw training
trajectories (sampled energy during SR).

Median over seeds against SR step, one column per N, one line per h; top: error per spin, bottom: relative
error; dashed: product state.
Checkpoints at 100, 300, 1000, 3000 steps (phase 2 extends phase 1).'''),
    code(r'''def convergence(arm="gibbs", hs=(0.5, 0.9, 1.0, 1.1, 1.5)):
    Ns = sorted(df[df.arm == arm].N.unique())
    fig, axes = plt.subplots(2, len(Ns), figsize=(3.2 * len(Ns), 6), sharey="row", sharex=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for j, N in enumerate(Ns):
        for i, h in enumerate(hs):
            d = df[(df.arm == arm) & (df.N == N) & (df.h == h)].dropna(subset=["err"])
            if d.empty:
                continue
            g = d.groupby("step").agg(err=("err", "median"), rel=("rel_err", "median"))
            c = cmap(i / max(len(hs) - 1, 1))
            axes[0, j].plot(g.index, g.err, "o-", color=c, ms=3, label=f"h={h:g}")
            axes[0, j].axhline(mf_err(N, h), color=c, ls="--", lw=0.8)
            axes[1, j].plot(g.index, 100 * g.rel, "o-", color=c, ms=3)
            axes[1, j].axhline(100 * mf_rel(N, h), color=c, ls="--", lw=0.8)
        axes[0, j].set_title(f"{arm}  N={N}"); axes[1, j].set_xlabel("SR step")
        for a in axes[:, j]:
            a.set_xscale("log"); a.set_yscale("log")
    axes[0, 0].set_ylabel("median (E - E0)/N"); axes[1, 0].set_ylabel("median (E - E0)/|E0| [%]")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()

convergence()'''),
    md(r'''Signed, on a symmetric-log axis: the per-step 200-sample estimates scatter **below** E0 (their noise is larger
than the remaining error), so the thin lines are not convergence; the thick lines are 20-step running means.
Exact SR has no sampling noise.'''),
    code(r'''def trajectories(arm, N, hs=(0.5, 1.0, 1.5), seeds=(0, 1, 2)):
    fig, ax = plt.subplots(2, len(hs), figsize=(4 * len(hs), 6), sharey="row", sharex=True, squeeze=False)
    for j, h in enumerate(hs):
        m = TransverseFieldIsing1D(N, h); e0 = m.exact_ground_energy()
        for row, scale, ref in ((0, 1 / N, mf_err(N, h)), (1, 100 / abs(e0), 100 * mf_rel(N, h))):
            a = ax[row, j]
            for s in seeds:
                f = DATA / "state" / f"{arm}_N{N}_h{h:g}_s{s}.npz"
                if f.exists():
                    E = (np.load(f)["energy"] - e0) * scale
                    x = np.arange(1, len(E) + 1)
                    line, = a.plot(x, E, lw=0.4, alpha=0.4)
                    w = 20 if len(E) >= 40 else 1
                    a.plot(x[w - 1:], np.convolve(E, np.ones(w) / w, "valid"), lw=1.2, color=line.get_color(),
                           label=f"seed {s} ({w}-step mean)")
            a.axhline(ref, color="k", ls="--", lw=1, label="product state")
            a.axhline(0, color="0.4", lw=0.6)
            a.set_yscale("symlog", linthresh=1e-5 * scale * N); a.set_xscale("log")
        ax[0, j].set_title(f"{arm} N={N} h={h}"); ax[1, j].set_xlabel("SR step")
    ax[0, 0].set_ylabel("(E_train - E0)/N  [symlog]"); ax[1, 0].set_ylabel("(E_train - E0)/|E0| [%, symlog]")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()

trajectories("gibbs", 64)'''),
    code(r'''trajectories("gibbs", 16)'''),
    code(r'''trajectories("exact_sr", 16)'''),
    md(r'''### A.3 Model B: order parameter and energy variance
⟨m₃²⟩ = ⟨|Σᵢ vᵢ e^{2πi sᵢ/3}|²⟩/N² (three-sublattice structure factor) and the V-score N·Var(H)/E² (reference-free;
0 for an exact eigenstate), at the last checkpoint, median over seeds. Finite-size ⟨m₃²⟩ stays nonzero in the
paramagnet; compare sizes rather than reading absolute values.'''),
    code(r'''def order_and_vscore():
    d = tv[(tv.model == "tri") & (tv.arm != "exact_sr")]
    d = d[d.step == d.groupby("run").step.transform("max")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (size, lab), x in d.groupby(["size", "label"]):
        g = x.groupby("h")[["m3sq", "vscore"]].median()
        ls = "-" if size == "6x6" else ":"
        ax[0].plot(g.index, g.m3sq, "o" + ls, ms=4, lw=1, label=f"{size} {lab}")
        ax[1].plot(g.index, g.vscore, "o" + ls, ms=4, lw=1, label=f"{size} {lab}")
    ax[0].set_ylabel("⟨m₃²⟩"); ax[1].set_ylabel("V-score N·Var(H)/E²"); ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("h"); a.axvline(1.6, color="0.7", lw=0.6)
    ax[1].legend(fontsize=6, ncol=2); fig.tight_layout()

order_and_vscore()'''),
    md(r'''### A.4 Validation of the DMRG samples (model A)
sample mean (from the per-sample bond averages, which are i.i.d.). z should look standard normal (about 95 % within 2).
Exactness of the sampler was also checked against full |ψ|² histograms (χ² test, N = 6–8) in the code audit.
This validates the sampler against *its* MPS, not against |ψ₀|². One known difference: for h < 1 DMRG returns a
symmetry-broken state (all samples m > 0), while the finite-N ground state has both signs equally; `sample()` now flips
each sample globally with probability 1/2 (`z2_symmetrize`). The zz check is Z2-even and blind to this.'''),
    code(r'''chk = dm.assign(z=lambda x: (x.zz_samples - x.zz_mps) / x.zz_samples_se)
print(f"{len(chk)} checks, |z| < 2: {(chk.z.abs() < 2).mean():.0%}, max |z| = {chk.z.abs().max():.2f}, "
      f"mean z = {chk.z.mean():+.2f}")
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.hist(chk.z, bins=20, color=chi_color[CHIS[-1]], density=True)
zz = np.linspace(-4, 4, 200); ax.plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=1)
ax.set_xlabel("z (samples vs MPS)"); ax.set_ylabel("density"); fig.tight_layout()'''),
    md(r'''## Appendix B — product state vs trained RBMs in detail (model A, reviewer's question)
The analyses written for the reviewer's question (is $h=0.5$ indistinguishable from a product state?).'''),
    md(r'''### B.1 SR step 100'''),
    code(r'''panel_plot(step=100)'''),
    md(r'''### B.2 How many times better than the product state? (median over seeds; < 1 = worse)
'''),
    code(r'''def gain_table(step=300, ceiling_step=3000):
    d = df[((df.arm == "exact_sr") & (df.step == ceiling_step)) | ((df.arm != "exact_sr") & (df.step == step))]
    t = d.groupby(["arm", "N", "h"]).gain_over_mf.median().unstack("h")
    return t.reindex(["exact_sr", "exact", "gibbs"], level="arm")

def show_gain(step=300):
    t = gain_table(step)
    L = np.nanmax(np.abs(np.log10(t.values)))  # diverging scale centred on 1x, spanning all values
    return t.style.format("{:.3g}", na_rep="").background_gradient(cmap="RdYlGn", axis=None, vmin=-L, vmax=L,
                                                                   gmap=np.log10(t))

show_gain(300)'''),
    code(r'''show_gain(100)'''),
    md(r'''### B.3 Relative error (E - E0)/|E0| in % (median over seeds), with the product state in the first row
'''),
    code(r'''def rel_table(step=300, ceiling_step=3000):
    d = df[((df.arm == "exact_sr") & (df.step == ceiling_step)) | ((df.arm != "exact_sr") & (df.step == step))]
    t = 100 * d.groupby(["arm", "N", "h"]).rel_err.median().unstack("h")
    mf = 100 * df[df.N == df.N.max()].groupby("h").mf_rel_err.first().to_frame().T
    mf.index = pd.MultiIndex.from_tuples([("product state", df.N.max())], names=["arm", "N"])
    return pd.concat([mf, t])

rel_table(300).style.format("{:.2g}", na_rep="")'''),
    md(r'''### B.4 N = 64 per seed — trapped or still improving?
`absm` = ⟨|m|⟩ and `dw` = mean number of domain walls of the trained network (global-flip evaluator).
References: `dw_GS` = ground-state domain walls, N(1 - ⟨σᶻσᶻ⟩)/2 from free fermions; `m_inf` = (1 - h²)^(1/8),
the thermodynamic-limit order parameter (h < 1), a rough guide to ⟨|m|⟩ at N = 64.
Compare like with like: a run with ground-state-like `dw` but `absm` far below `m_inf` has not ordered
(domains of both signs); `dw` well above `dw_GS` means too many domain walls (h = 0.5).'''),
    code(r'''def dw_gs(N, h):
    k = np.pi * (2 * np.arange(N) + 1) / N
    zz = np.mean((1 - h * np.cos(k)) / np.sqrt(1 + h * h - 2 * h * np.cos(k)))  # -dE0/dJ per bond
    return N * (1 - zz) / 2

def per_seed(N=64, arm="gibbs"):
    d = df[(df.arm == arm) & (df.N == N)].assign(rel_pct=lambda x: 100 * x.rel_err).pivot_table(
        index=["h", "seed"], columns="step", values=["err", "rel_pct", "absm", "dw"])
    hs = d.index.get_level_values("h")
    d["dw_GS"] = [dw_gs(N, h) for h in hs]
    d["m_inf"] = [(1 - h * h) ** 0.125 if h < 1 else 0.0 for h in hs]
    d["mf_err"] = [mf_err(N, h) for h in hs]
    d["mf_rel_pct"] = [100 * mf_rel(N, h) for h in hs]
    return d  # err: per spin; rel_pct: relative error in %

per_seed(64)'''),
    md(r'''### B.5 Where does a trained network beat the product state?
Fraction of seeds with error below the product-state error, per N and h, at each SR step (Gibbs).'''),
    code(r'''def beat_fraction(arm="gibbs"):
    d = df[df.arm == arm].dropna(subset=["err"]).assign(beats=lambda x: x.err < x.mf_err)
    t = d.groupby(["N", "step", "h"]).beats.agg(lambda b: f"{b.sum()}/{b.size}").unstack("h")
    return t

beat_fraction("gibbs")'''),
    md(r'''### B.6 The review's h = 0.5 claim (stored benchmark runs)
From `scripts/validation/mean_field_baseline.py` → `results/mean_field_baseline/summary.json`.'''),
    code(r'''mb = json.loads((REPO / "results" / "mean_field_baseline" / "summary.json").read_text())
lv = pd.DataFrame(mb["reference_levels"]).set_index(["h", "N"])
e0n = lv.E0_per_spin.abs()
for k in ("all_up", "x_pol", "eps", "1pct"):  # relative error next to every per-spin error
    lv[f"rel_{k}"] = lv[f"err_{k}"] / e0n
display(lv)  # err_*: per spin; rel_*: relative (fraction)
runs = pd.DataFrame(mb["runs_vs_mean_field"]).drop(columns="file").set_index(["runs", "estimate", "N"])
e0n_runs = runs.index.get_level_values("N").map(lambda N: e0n.loc[(0.5, N)])
for k in ("median_err", "min_err", "max_err"):
    runs[k.replace("err", "rel_pct")] = 100 * runs[k] / np.asarray(e0n_runs)
runs  # *_err: per spin; *_rel_pct: relative error in %; median_over_mf is the same for both'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    if args.execute:
        import tempfile

        from jupyter_client import KernelManager
        from nbclient import NotebookClient

        # IPC transport: the kernel talks over Unix domain sockets, so no TCP port is opened, not even on
        # localhost. No notebook server is started; open the result in VS Code or the exported HTML.
        with tempfile.TemporaryDirectory() as tmp:
            km = KernelManager(kernel_name="python3", transport="ipc", ip=str(Path(tmp) / "kernel"))
            NotebookClient(nb, km=km, timeout=900, resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

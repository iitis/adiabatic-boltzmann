# Handoff — referee point 5 (statistically fair benchmarking)

State as of 2026-07-14. Everything below is in this repo; commit + pull on the
other machine to continue.

## Done: J1-J2 convergence figure (was `fig_convergence_best`)

- **New script:** `scripts/j1j2/j1j2_convergence_median.py` — single reproducible
  source for the figure. Protocol:
  - config per (N, J2) cell re-derived from the pooled Optuna tuning studies
    (`ite_run.global_best_per_combo_sampler`, gibbs, lowest tuning rel_error) —
    verified identical to the configs of the existing production sweeps;
  - evaluation seeds disjoint from tuning seeds {1, 42, 123};
  - exact J2 match (old script's `round(J2,1)` mixed J2=0.45 probes into the 0.5 panel);
  - median E/N trajectory + IQR band over ~20-28 seeds per panel;
  - append-only divergence ledger `results/j1j2_convergence_median/summary.json`
    (survives `--plot-only`; failures = censored observations, shown as `n=a/b`).
- **Key finding:** gibbs config crashes deterministically for most seeds in the
  frustrated regime — (N=8, J2=0.7): 14/22 diverged; (N=8, J2=0.9): 18/21.
- **Report integrated:** `../adiabatic-boltzmann-report/report.tex` Figure 10
  (`figures/fig_convergence_median.pdf`), caption states full protocol, body text
  now says "median relative error rises sharply beyond J2/J1 = 0.5". Compiles clean
  (pdflatex -shell-escape, 2 passes).
- Old `scripts/viz/plot_j1j2_convergence_curves.py` removed (git rm).

Regenerate anytime:
```bash
python scripts/j1j2/j1j2_convergence_median.py              # top-up missing seeds + plot
python scripts/j1j2/j1j2_convergence_median.py --plot-only  # plot from existing data
cp plots/j1j2/fig_convergence_median.pdf ../adiabatic-boltzmann-report/figures/
```

## In progress: parallel-embedding benchmark (replaces best-of-3-seeds figure)

- **New script:** `scripts/exper/parallel_embedding_bench.py`. Pre-registered
  protocol in module constants:
  - arms n_parallel {1,3,5}; 12 seeds `[0,2,3,4,5,6,8,9,10,11,12,13]` in declared
    order (excludes tuning-adjacent 1/42/123); per-seed arm order shuffled
    (META_SEED=7) so QPU drift can't confound arms; paired seeds enable per-seed
    ratio tests;
  - primary endpoint: cumulative QPU sampling time to *sustained* eps=1% over a
    full 10-iteration window; crashes/non-converged right-censored, Kaplan-Meier
    median + bootstrap CI; crash rate with Wilson CI; secondary: rel. error at 5 s;
  - budget guard: 11 min allocation, measured as `time.json` deltas from a baseline
    stored in the ledger at first live run; projects next-run cost, hard-aborts;
    truncation drops trailing (pre-declared) seeds = unbiased;
  - append-only ledger written after every run:
    `results/parallel_embedding_bench/ledger{_rehearsal}.json`; restarts skip
    completed (seed, arm) pairs.

### Where it stands

Rehearsal (full pipeline, zero QPU spend) iterated through samplers:
1. ClassicalSampler metropolis — no mixing (~14/990 unique), 0 converged, event
   path unexercised;
2. ClassicalSampler gibbs — 100% NaN@3 (QUBO couplings explode on this config);
3. **current:** `DimodSampler(method="simulated_annealing")` (neal — same QUBO
   pipeline as QPU, classical, no time.json writes). Rehearsal 3 was started and
   **stopped mid-run at user request**; `ledger_rehearsal.json` may be partial.

### Next steps (in order)

```bash
# 1. finish the rehearsal (safe, no QPU; resumes/skips completed pairs)
python scripts/exper/parallel_embedding_bench.py --rehearse

# expect: some converged events, KM medians + paired ratios printed,
# figure plots/parallel_embedding/parallel_embedding_bench_rehearsal.{pdf,png}
# marked "REHEARSAL"

# 2. inspect schedule + cost projection
python scripts/exper/parallel_embedding_bench.py --dry-run
# last check: ~8.2 min projected QPU of 11 min allocation

# 3. THE ONE LIVE QPU SESSION (needs D-Wave credentials on the machine)
python scripts/exper/parallel_embedding_bench.py --live

# 4. figure from live ledger
python scripts/exper/parallel_embedding_bench.py --plot-only --live-ledger
```

Then: swap the new figure into the report (currently
`figures/parallel_embedding/parallel_embedding_np_best_qpu_time.pdf`,
caption "best of 3 seeds each" at report.tex ~line 507) and rewrite that caption
to state the protocol (mirror Figure 10's caption style).

## Still open from referee point 5 (not started)

- ITE figure (`scripts/viz/plot_ite.py`): drop in-sample `select_best_configs()`,
  consume declared configs from ite_run summaries instead.
- TTC (`scripts/viz/plot_ttc.py`): censored aggregation (currently drops
  non-converged runs from panels); stop pooling hyperparameter configs.
- Stale `plots/tte/` figures: regenerate or delete.
- QPU multi-seed runs for the single-seed `dimod/zephyr` / `pegasus_fast` TTC/ITE
  curves — needs its own QPU budget decision.
- Stopping rule text for the report: sustained-window criterion is implemented in
  the new scripts; document it wherever TTC/ITE figures land.

## House rules discovered/confirmed this session

- Never generate paper-figure data via `scripts/main.py` — dedicated in-process
  runner per figure (ite_run.py pattern).
- Never hand-patch result artifacts (summaries, ledgers) — fix script, delete
  artifact, re-run end-to-end. On error: report and ask first.
- One JAX process at a time (single GPU).

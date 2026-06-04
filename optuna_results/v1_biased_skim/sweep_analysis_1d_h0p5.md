# VeloxQstandard SA Gibbs sweep — analysis (1d, h=0.5)

Setup: start_temp=1 (pinned), scale_ising=False, compress=False, beta_x=1, JAX x64.
Optuna searched: num_sweeps_per_step + learning_rate + regularization + n_hidden.
Objective stored = signed gap (mean last-5 energy - E_exact); negative => energy below exact => sampling bias.

## N=16 (50 trials)
- Most faithful (min |err|): trial 29, |err|=0.0019, sweeps=500, lr=0.0217, n_hidden=51
- Most biased (min signed): trial 34, signed=-0.6340, sweeps=1000

| sweeps | n_converged | mean_signed | best\|err\| |
|---|---|---|---|
| 10 | 12 | -0.3149 | 0.3188 |
| 100 | 1 | +0.6745 | 0.6745 |
| 500 | 2 | -0.1751 | 0.0019 |
| 1000 | 5 | -0.0149 | 0.2704 |
| 2000 | 7 | -0.2502 | 0.1747 |
| 5000 | 3 | -0.0872 | 0.1874 |
| 10000 | 2 | +0.2790 | 0.3757 |
| 50000 | 0 (of 3) | — diverged — | — |

## N=24 (50 trials)
- Most faithful (min |err|): trial 17, |err|=0.0484, sweeps=100, lr=0.0569, n_hidden=83
- Most biased (min signed): trial 45, signed=-0.5099, sweeps=10000

| sweeps | n_converged | mean_signed | best\|err\| |
|---|---|---|---|
| 10 | 3 | +0.2688 | 0.0989 |
| 100 | 9 | -0.0075 | 0.0484 |
| 500 | 3 | +0.4275 | 0.3563 |
| 1000 | 0 (of 2) | — diverged — | — |
| 2000 | 1 | -0.2377 | 0.2377 |
| 5000 | 0 (of 2) | — diverged — | — |
| 10000 | 7 | -0.3328 | 0.0671 |
| 50000 | 1 | +0.6670 | 0.6670 |

## Caveat
Per-sweep buckets are tiny and confounded because the VMC params were searched (user choice).
For a clean bias-vs-sweeps curve: fix lr~0.02-0.05 / n_hidden~size / reg~1e-3 and sweep only
num_sweeps_per_step with multiple seeds each.

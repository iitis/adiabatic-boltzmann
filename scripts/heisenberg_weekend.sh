#!/bin/bash
# Heisenberg J1-J2 weekend benchmark: sweep → TTE → plot
# Usage: bash scripts/heisenberg_weekend.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "=========================================="
echo " Phase 1a: sweep N=8,12  J2=0.7,0.9"
echo "=========================================="
python3 scripts/hparam_optuna.py \
  --hamiltonian heisenberg_j1j2_1d \
  --N 8 12 \
  --J2 0.7 0.9 \
  --sampling-methods gibbs exchange simulated_annealing \
  --ansatz-types rbm \
  --n-trials 100 --iterations 100 --seeds 1 42 \
  --study-name heisenberg_weekend

echo "=========================================="
echo " Phase 1b: sweep N=16,24,32  J2=0.1–0.9"
echo "=========================================="
python3 scripts/hparam_optuna.py \
  --hamiltonian heisenberg_j1j2_1d \
  --N 16 24 32 \
  --J2 0.1 0.3 0.5 0.7 0.9 \
  --sampling-methods gibbs exchange simulated_annealing \
  --ansatz-types rbm \
  --n-trials 100 --iterations 100 --seeds 1 42 \
  --study-name heisenberg_weekend

echo "=========================================="
echo " Phase 2: TTE  (30 seeds × 5 N × 5 J2)"
echo "=========================================="
python3 scripts/tts_run.py \
  --hamiltonian heisenberg_j1j2_1d \
  --N 8 12 16 24 32 \
  --sampling-methods gibbs exchange simulated_annealing \
  --n-seeds 30 --iterations 300 \
  --epsilon 0.01 0.001 0.0001

echo "=========================================="
echo " Phase 3: plot"
echo "=========================================="
python3 scripts/viz/plot_tte.py --model heisenberg_j1j2_1d

echo "Done."

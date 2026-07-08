#!/usr/bin/env bash
# Multi-seed FPGA/VeloxQ-SA production sweep for TFIM N=64 and N=128.
# N=32 is intentionally excluded (out of scope for this round).
#
# N=64 already has an Optuna-tuned config (results/hparam_search/tfim_1d/
# veloxq_tfim/N64_h0.5/) — no new trials are run for it. N=128 has none yet,
# so this runs a fresh search for it first (same search space/budget as the
# N=32/64 study, via hparam_veloxq_tfim_n128.py) before using per-size tuned
# hyperparameters instead of the fixed lr=0.13/reg=0.005 generalization
# sweep (which collapses at N>=32 because reg=0.005 is ~4 orders of
# magnitude larger than the per-size Optuna optimum).
#
# Runs three stages in order:
#   1. hparam_veloxq_tfim_n128.py             (Optuna search for N=128 only;
#      resumable — skips trials already completed if rerun)
#   2. run_fpga_best.py --no-generalize --num-sweeps 100   (production sweep, N=64+128)
#   3. run_fpga_best.py --no-generalize --num-sweeps 2000  (production sweep, N=64+128)
#
# Usage:
#   scripts/fpga/run_n64_128_sweep.sh
#   scripts/fpga/run_n64_128_sweep.sh --n-trials 100 --n-seeds 30
#
# Any extra arguments are forwarded to both the hparam search and the
# production sweep stages where the flag name applies to each.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python3
fi

N_TRIALS=60
N_SEEDS=20
ITERATIONS=100
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --n-trials) N_TRIALS="$2"; shift 2 ;;
        --n-seeds) N_SEEDS="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=============================================================="
echo "Stage 1/3: Optuna hyperparameter search for N=128"
echo "(N=64 already has tuned data — no new trials run for it)"
echo "=============================================================="
"$PYTHON" scripts/hparam/hparam_veloxq_tfim_n128.py \
    --n-trials "$N_TRIALS" \
    "${EXTRA_ARGS[@]}"

HPARAM_ROOT="results/hparam_search/tfim_1d/veloxq_tfim"
SIZES=()
for N in 64 128; do
    if compgen -G "$HPARAM_ROOT/N${N}_h*/index.jsonl" > /dev/null; then
        SIZES+=("$N")
    else
        echo "WARNING: no hparam trials for N=$N under $HPARAM_ROOT — skipping." >&2
    fi
done
if [ ${#SIZES[@]} -eq 0 ]; then
    echo "No sizes have hparam data. Aborting." >&2
    exit 1
fi
echo "Sizes with hparam data: ${SIZES[*]}"

echo
echo "=============================================================="
echo "Stage 2/3: production sweep, num_sweeps=100"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes "${SIZES[@]}" --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 100 --output-dir results/sweeps100_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Stage 3/3: production sweep, num_sweeps=2000"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes "${SIZES[@]}" --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 2000 --output-dir results/sweeps2000_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Done. Results under results/sweeps{100,2000}_v2/tfim_1d/{${SIZES[*]}}/"
echo "Plot with:"
echo "  python scripts/viz/plot_ite.py --model tfim_1d --h 0.5 \\"
echo "      --methods fpga/fpga@sweeps100_v2 fpga/fpga@sweeps2000_v2 \\"
echo "                velox/simulated_annealing@sweeps100_v2 velox/simulated_annealing@sweeps2000_v2 \\"
echo "      --suffix _fpga_velox_v2"
echo "=============================================================="

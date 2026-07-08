#!/usr/bin/env bash
# Multi-seed FPGA/VeloxQ-SA production sweep for TFIM, using per-size tuned
# hyperparameters already found by the existing Optuna search
# (results/hparam_search/tfim_1d/veloxq_tfim/) rather than the fixed
# lr=0.13/reg=0.005 generalization sweep (which collapses at N>=32 because
# reg=0.005 is ~4 orders of magnitude larger than the per-size optimum).
#
# Runs NO new Optuna trials — it only reads the top-K already-completed
# trials from results/hparam_search/tfim_1d/veloxq_tfim/N{size}_h{h}/ and
# re-runs that config across --n-seeds seeds on both backends. Sizes with no
# existing hparam data (currently N=128) are skipped with a warning rather
# than run with untuned/fixed hyperparameters.
#
# Runs two stages in order:
#   1. run_fpga_best.py --no-generalize --num-sweeps 100   (production sweep)
#   2. run_fpga_best.py --no-generalize --num-sweeps 2000  (production sweep)
#
# Usage:
#   scripts/fpga/run_n32_64_128_sweep.sh
#   scripts/fpga/run_n32_64_128_sweep.sh --n-seeds 30
#
# Any extra arguments are forwarded to both stages.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python3
fi

HPARAM_ROOT="results/hparam_search/tfim_1d/veloxq_tfim"
CANDIDATE_SIZES=(32 64 128)
SIZES=()
for N in "${CANDIDATE_SIZES[@]}"; do
    if compgen -G "$HPARAM_ROOT/N${N}_h*/index.jsonl" > /dev/null; then
        SIZES+=("$N")
    else
        echo "WARNING: no existing hparam trials for N=$N under $HPARAM_ROOT — skipping." >&2
    fi
done

if [ ${#SIZES[@]} -eq 0 ]; then
    echo "No sizes have existing hparam data. Run scripts/hparam/hparam_veloxq_tfim.py first." >&2
    exit 1
fi

N_SEEDS=20
ITERATIONS=100
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --n-seeds) N_SEEDS="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "Sizes with existing hparam data: ${SIZES[*]}"

echo
echo "=============================================================="
echo "Stage 1/2: production sweep, num_sweeps=100"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes "${SIZES[@]}" --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 100 --output-dir results/sweeps100_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Stage 2/2: production sweep, num_sweeps=2000"
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

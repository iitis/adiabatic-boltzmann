#!/usr/bin/env bash
# Multi-seed FPGA/VeloxQ-SA production sweep for TFIM N=64 only, using the
# per-size tuned hyperparameters already found by the existing Optuna search
# (results/hparam_search/tfim_1d/veloxq_tfim/N64_h0.5/) rather than the fixed
# lr=0.13/reg=0.005 generalization sweep (which collapses at N>=32 because
# reg=0.005 is ~4 orders of magnitude larger than the per-size optimum).
#
# Runs NO new Optuna trials — it only reads the top-K already-completed
# trials for N=64 and re-runs that config across --n-seeds seeds on both
# backends (fpga, veloxq_sa).
#
# Runs two stages in order:
#   1. run_fpga_best.py --no-generalize --num-sweeps 100   (production sweep)
#   2. run_fpga_best.py --no-generalize --num-sweeps 2000  (production sweep)
#
# Usage:
#   scripts/fpga/run_n64_sweep.sh
#   scripts/fpga/run_n64_sweep.sh --n-seeds 30
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
if ! compgen -G "$HPARAM_ROOT/N64_h*/index.jsonl" > /dev/null; then
    echo "ERROR: no existing hparam trials for N=64 under $HPARAM_ROOT." >&2
    echo "Run scripts/hparam/hparam_veloxq_tfim.py --N 64 first." >&2
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

echo "=============================================================="
echo "Stage 1/2: production sweep, N=64, num_sweeps=100"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes 64 --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 100 --output-dir results/sweeps100_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Stage 2/2: production sweep, N=64, num_sweeps=2000"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes 64 --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 2000 --output-dir results/sweeps2000_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Done. Results under results/sweeps{100,2000}_v2/tfim_1d/64/"
echo "Plot with:"
echo "  python scripts/viz/plot_ite.py --model tfim_1d --h 0.5 \\"
echo "      --methods fpga/fpga@sweeps100_v2 fpga/fpga@sweeps2000_v2 \\"
echo "                velox/simulated_annealing@sweeps100_v2 velox/simulated_annealing@sweeps2000_v2 \\"
echo "      --suffix _fpga_velox_n64_v2"
echo "=============================================================="

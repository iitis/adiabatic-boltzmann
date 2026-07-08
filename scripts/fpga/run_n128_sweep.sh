#!/usr/bin/env bash
# Multi-seed FPGA/VeloxQ-SA production sweep for TFIM N=128 only. Mirrors
# run_n64_sweep.sh, but N=128 has no existing Optuna search, so this requires
# scripts/hparam/hparam_veloxq_tfim_n128.py to have been run first.
#
# Runs two stages in order:
#   1. run_fpga_best.py --no-generalize --num-sweeps 100   (production sweep)
#   2. run_fpga_best.py --no-generalize --num-sweeps 2000  (production sweep)
#
# Usage:
#   python scripts/hparam/hparam_veloxq_tfim_n128.py   # once, first
#   scripts/fpga/run_n128_sweep.sh
#   scripts/fpga/run_n128_sweep.sh --n-seeds 30
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
if ! compgen -G "$HPARAM_ROOT/N128_h*/index.jsonl" > /dev/null; then
    echo "ERROR: no existing hparam trials for N=128 under $HPARAM_ROOT." >&2
    echo "Run scripts/hparam/hparam_veloxq_tfim_n128.py first." >&2
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
echo "Stage 1/2: production sweep, N=128, num_sweeps=100"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes 128 --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 100 --output-dir results/sweeps100_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Stage 2/2: production sweep, N=128, num_sweeps=2000"
echo "=============================================================="
"$PYTHON" scripts/fpga/run_fpga_best.py --no-generalize \
    --sizes 128 --backends fpga veloxq_sa \
    --top-k 1 --n-seeds "$N_SEEDS" --iterations "$ITERATIONS" \
    --num-sweeps 2000 --output-dir results/sweeps2000_v2 \
    "${EXTRA_ARGS[@]}"

echo
echo "=============================================================="
echo "Done. Results under results/sweeps{100,2000}_v2/tfim_1d/128/"
echo "Plot with:"
echo "  python scripts/viz/plot_ite.py --model tfim_1d --h 0.5 \\"
echo "      --methods fpga/fpga@sweeps100_v2 fpga/fpga@sweeps2000_v2 \\"
echo "                velox/simulated_annealing@sweeps100_v2 velox/simulated_annealing@sweeps2000_v2 \\"
echo "      --suffix _fpga_velox_n128_v2"
echo "=============================================================="

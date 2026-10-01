#!/bin/bash
# Pegasus rerun under the P4 protocol (3-draw CEM calibration + visible-PL tracking),
# same settings as the Zephyr P4 runs in results/cem_zephyr_protocol_qpu/:
# N = 8 16 32 64, seeds 0-19, one background process per size.
#
# Budget: hard cap of 600 s (10 min) on time.json in the repo root, the ground-truth
# QPU counter. Checked before each run and before every QPU call (see
# cem_zephyr_protocol_qpu.py), so the cap cannot be exceeded by more than one call
# per process (~40 ms). Expected use is ~290 s.
#
# Writes only to results/cem_pegasus_protocol_qpu/ (append-only, finished seeds are
# skipped), so running it again resumes and never touches existing results.
#
# Usage (no arguments, bash script, already backgrounds itself): bash scripts/exper/cem_pegasus_p4_launch.sh
# Uses $PYTHON, else .venv/bin/python, else the python on PATH (e.g. an active conda env).
set -euo pipefail
cd "$(dirname "$0")/../.."
CAP_S=600
SEEDS=0-19
SIZES="8 16 32 64"
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=$(command -v python || true)
fi
O=results/cem_pegasus_protocol_qpu

[ -n "$PY" ] && [ -x "$PY" ] || { echo "python not found (set PYTHON=...)"; exit 1; }
echo "python: $PY"
[ -f time.json ] || { echo "time.json missing: it is the QPU budget counter, refusing to start"; exit 1; }
if pgrep -f "^[^ ]*python[^ ]* scripts/exper/cem_zephyr_protocol_qpu.py .*--device pegasus" > /dev/null; then
  echo "Pegasus P4 runs are already running:"; pgrep -af "^[^ ]*python[^ ]* scripts/exper/cem_zephyr_protocol_qpu.py .*--device pegasus"; exit 1
fi
used=$("$PY" -c "import json; print(json.load(open('time.json'))['time_ms'] / 1000)")
echo "time.json: ${used} s used, cap ${CAP_S} s"

mkdir -p $O
n=$(wc -w <<< "$SIZES")
for N in $SIZES; do
  JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} nohup "$PY" scripts/exper/cem_zephyr_protocol_qpu.py P4cemPL $N $SEEDS \
    --calib 3 --fb cem+pl --ci 1 --alpha 0.5 --device pegasus --cap-s $CAP_S --n-procs $n \
    >> $O/P4cemPL_N${N}.log 2>&1 &
  echo "N=$N pid $! -> $O/P4cemPL_N${N}.log"
done
echo "monitor: tail -f $O/*.log ; budget: cat time.json"

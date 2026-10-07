#!/bin/bash
# (1) MC evaluator on all N<=16 networks, compared with enumeration in make_v3.py;
# (2) second, independent N=64 evaluation with 5x longer burn-in and a different seed stream.
cd "$(dirname "$0")/../.."
P=.venv/bin/python; O=results/audit_eval3
for d in pegasus zephyr; do for N in 8 16; do
  f=results/cem_${d}_protocol_qpu/P4cemPL_N$N.jsonl; [ -f $f ] && $P scripts/exper/audit_evaluate.py $f $O/${d}_P4_N$N.jsonl &
done; done
for d in pegasus zephyr; do $P scripts/exper/audit_evaluate.py results/cem_${d}_protocol_qpu/P4cemPL_N64.jsonl $O/${d}_P4_N64_long.jsonl --burn 5000 --seed-offset 1 & done
wait

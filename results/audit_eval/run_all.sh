#!/bin/bash
# Re-evaluate all final networks with scripts/exper/audit_evaluate.py (global-flip Metropolis).
cd "$(dirname "$0")/../.."
P=.venv/bin/python; O=results/audit_eval
for N in 64 32 16 8; do
  $P scripts/exper/audit_evaluate.py results/cem_pegasus_protocol_qpu/P4cemPL_N$N.jsonl $O/pegasus_P4_N$N.jsonl
  $P scripts/exper/audit_evaluate.py results/cem_zephyr_protocol_qpu/P4cemPL_N$N.jsonl $O/zephyr_P4_N$N.jsonl
done
for f in results/audit_ablation_qpu/{fixed,calib_only,pl_only,cem_step}_N*.jsonl; do
  $P scripts/exper/audit_evaluate.py $f $O/ablation_$(basename $f)
done

#!/bin/bash
# Final-network evaluation with an own Monte Carlo seed per network (N>=32; N<=16 uses enumeration).
cd "$(dirname "$0")/../.."
P=.venv/bin/python; O=results/audit_eval2
for N in 64 32; do
  $P scripts/exper/audit_evaluate.py results/cem_pegasus_protocol_qpu/P4cemPL_N$N.jsonl $O/pegasus_P4_N$N.jsonl
  $P scripts/exper/audit_evaluate.py results/cem_zephyr_protocol_qpu/P4cemPL_N$N.jsonl $O/zephyr_P4_N$N.jsonl
  for v in mh mh_flip; do $P scripts/exper/audit_evaluate.py results/audit_classical/${v}_N$N.jsonl $O/classical_${v}_N$N.jsonl; done
done
for m in fixed calib_only pl_only cem_step auto; do
  $P scripts/exper/audit_evaluate.py results/audit_ablation_qpu/${m}_N64.jsonl $O/ablation_${m}_N64.jsonl
done

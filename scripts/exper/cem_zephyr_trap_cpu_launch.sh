#!/bin/bash
# Launch one config of cem_zephyr_trap_gpu.py on CPU, split into chunks of seeds (one process per chunk).
# Usage: cem_zephyr_trap_cpu_launch.sh NAME FIRST LAST CHUNK key=val ...
cd "$(dirname "$0")/../.."
name=$1 lo=$2 hi=$3 ch=$4; shift 4
O=results/cem_zephyr_trap_gpu
for ((s=lo; s<=hi; s+=ch)); do e=$((s+ch-1<hi ? s+ch-1 : hi))
  JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    nohup ${PYTHON:-python} scripts/exper/cem_zephyr_trap_gpu.py $O/${name}_s$s.jsonl $s-$e "$@" > $O/${name}_s$s.log 2>&1 &
done

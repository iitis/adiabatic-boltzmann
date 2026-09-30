#!/bin/bash
# Batch 1 of GPU-only Zephyr-trap follow-ups (see cem_zephyr_trap_gpu.py). N=32 unless stated.
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
P=${PYTHON:-python}; S=scripts/exper/cem_zephyr_trap_gpu.py; O=results/cem_zephyr_trap_gpu
run() { name=$1; shift; $P $S $O/$name.jsonl 0-19 "$@" > $O/$name.log 2>&1 & }
run A1_exact             sampler=exact
run A2_exact_lr04        sampler=exact lr=0.04 iters=200
run A3_exact_ns1000      sampler=exact ns=1000
run B0_synth_bhw1        beta_hw=1
run C0_synth_bhw6        beta_hw=6
run C1_bhw6_freeze30     beta_hw=6 freeze=30
run C2_bhw6_alpha1       beta_hw=6 alpha=1.0
run C3_bhw6_warmup40     beta_hw=6 warmup=40
run C4_bhw6_ci1a1_frz10  beta_hw=6 ci=1 alpha=1.0 freeze=10
run D1_bhw6_bxi6_ice015  beta_hw=6 bxi=6 ice=0.015
run D2_bhw6_bxi6_ice03   beta_hw=6 bxi=6 ice=0.03
wait

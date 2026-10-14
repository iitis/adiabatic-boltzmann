#!/bin/bash
# Phase 1 of mf_vs_h.py (~1 h on one GPU): 3 seeds, 10 h values, checkpoints at 100 and 300 SR steps
# (exact_sr ceiling: 100..3000). Extend later by rerunning with more --seeds / longer --steps.
SELF=$(readlink -f "$0")  # absolute: the detached re-exec below runs after the cd
cd "$(dirname "$SELF")/../.."
O=${MF_VS_H_OUT:-results/mf_vs_h}  # same default/override as mf_vs_h.py
# Detach from the terminal (survives SSH disconnects / closing the shell): re-exec under setsid + nohup.
# Set MF_FOREGROUND=1 to run attached instead.
if [ -z "$MF_DETACHED" ] && [ -z "$MF_FOREGROUND" ]; then
  mkdir -p "$O/logs"
  LOG=$O/logs/launcher_$(basename "$SELF" .sh)_$(date +%Y%m%d_%H%M%S).log
  MF_DETACHED=1 setsid nohup "$SELF" "$@" > "$LOG" 2>&1 < /dev/null &
  echo "detached: pid $! (session leader), log $LOG"
  exit 0
fi
P=${PYTHON:-python}; L=$O/logs; mkdir -p $L
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LO=0.5,0.8,0.9,0.95,1.0; HI=1.05,1.1,1.25,1.5,2.0
run() { $P scripts/exper/mf_vs_h.py "$@" > $L/$(echo "$@" | tr ' ,' '__' | tr -d '-').log 2>&1 & }
run --arm exact_sr --N 8  --seeds 0-2 --steps 100,300,1000,3000
run --arm exact_sr --N 16 --seeds 0-2 --steps 100,300,1000,3000
run --arm exact --N 8  --seeds 0-2 --steps 100,300
run --arm exact --N 16 --seeds 0-2 --steps 100,300 --hs 0.5,0.8,0.9
run --arm exact --N 16 --seeds 0-2 --steps 100,300 --hs 0.95,1.0,1.05
run --arm exact --N 16 --seeds 0-2 --steps 100,300 --hs 1.1,1.25,1.5,2.0
run --arm gibbs --N 8  --seeds 0-2 --steps 100,300
run --arm gibbs --N 16 --seeds 0-2 --steps 100,300
run --arm gibbs --N 32 --seeds 0-2 --steps 100,300 --hs $LO
run --arm gibbs --N 32 --seeds 0-2 --steps 100,300 --hs $HI
run --arm gibbs --N 64 --seeds 0-2 --steps 100,300 --hs $LO
run --arm gibbs --N 64 --seeds 0-2 --steps 100,300 --hs $HI
wait

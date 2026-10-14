#!/bin/bash
# Persistent-Gibbs arm of mf_vs_h.py on CPU, one single-threaded process per (N, h), with the
# N > 16 Metropolis evaluations deferred to one GPU process (many CUDA contexts on one GPU time-slice
# badly: 12 processes ran ~10x slower than one).
#   scripts/exper/mf_vs_h_gibbs_cpu.sh SEEDS STEPS [NS] [HS]
#   scripts/exper/mf_vs_h_gibbs_cpu.sh 0-2 100,300
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
SEEDS=$1; STEPS=$2; NS=${3:-8,16,32,64}; HS=${4:-0.5,0.8,0.9,0.95,1.0,1.05,1.1,1.25,1.5,2.0}
P=${PYTHON:-python}; L=$O/logs; mkdir -p $L; rm -f $O/TRAINING_DONE
XLA_PYTHON_CLIENT_PREALLOCATE=false $P scripts/exper/mf_vs_h.py --evaluate-pending 30 > $L/evaluate_pending.log 2>&1 &
EVAL=$!
for N in ${NS//,/ }; do
  for h in ${HS//,/ }; do
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $P scripts/exper/mf_vs_h.py --arm gibbs --N $N --hs $h --seeds $SEEDS --steps $STEPS --defer-eval \
      > $L/gibbs_N${N}_h${h}.log 2>&1 &
  done
done
wait $(jobs -p | grep -v "^$EVAL$")
touch $O/TRAINING_DONE
wait $EVAL

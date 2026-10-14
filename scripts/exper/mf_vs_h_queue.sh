#!/bin/bash
# Job-queue launcher for mf_vs_h.py: CPU training jobs from a file, at most NPROC at once, each a
# single-threaded process with deferred N > 16 evaluation; plus GPU evaluators, one per glob in
# MF_EVAL_PATTERNS (disjoint globs, so each <file>.eval.jsonl has a single writer). Detaches itself
# (setsid + nohup) so it survives SSH disconnects; MF_FOREGROUND=1 runs attached.
# Each training job is pinned to its own core (taskset, by xargs slot): JAX's CPU backend starts ~200
# Eigen threads per process regardless of XLA_FLAGS, and 30 unpinned jobs drove the load to ~130 on 36 cores.
#   scripts/exper/mf_vs_h_queue.sh JOBFILE [NPROC]
# JOBFILE: one mf_vs_h.py argument list per line ('#' comments allowed), e.g.
#   --arm gibbs --N 64 --hs 1.0 --seeds 0-2 --steps 100,300,1000,3000
SELF=$(readlink -f "$0")
JOBS=$(readlink -f "$1"); NPROC=${2:-30}
cd "$(dirname "$SELF")/../.."
O=${MF_VS_H_OUT:-results/mf_vs_h}
if [ -z "$MF_DETACHED" ] && [ -z "$MF_FOREGROUND" ]; then
  mkdir -p "$O/logs"
  LOG=$O/logs/launcher_$(basename "$JOBS" .txt)_$(date +%Y%m%d_%H%M%S).log
  MF_DETACHED=1 setsid nohup "$SELF" "$JOBS" "$NPROC" > "$LOG" 2>&1 < /dev/null &
  echo "detached: pid $! (session leader), log $LOG"
  exit 0
fi
P=${PYTHON:-python}; L=$O/logs; mkdir -p $L; rm -f $O/TRAINING_DONE
echo "$(date) start: $(grep -cv '^\s*\(#\|$\)' "$JOBS") jobs, $NPROC parallel"
EVALS=()
for pat in ${MF_EVAL_PATTERNS:-*.jsonl}; do
  # evaluators (GPU) share the cores left over by the training slots
  XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c "$NPROC-$(($(nproc) - 1))" \
    $P scripts/exper/mf_vs_h.py --evaluate-pending 30 --eval-pattern "$pat" \
    > "$L/evaluate_pending_$(echo "$pat" | tr -c 'A-Za-z0-9_\n' '_').log" 2>&1 &
  EVALS+=($!)
done
export P L
grep -v '^\s*\(#\|$\)' "$JOBS" | xargs -P "$NPROC" --process-slot-var=SLOT -I{} bash -c '
  args="{}"; log="$L/job_$(echo "$args" | tr -c "A-Za-z0-9.\n" "_" | tr -s _).log"
  JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c "$SLOT" $P scripts/exper/mf_vs_h.py $args --defer-eval >> "$log" 2>&1 && exit 0
  # The python3.14 interpreter occasionally segfaults (seen at start-up); runs resume from their last
  # checkpoint, so retry before giving up.
  for try in 2 3; do
    echo "RETRY $try: $args"
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      taskset -c "$SLOT" $P scripts/exper/mf_vs_h.py $args --defer-eval >> "$log" 2>&1 && exit 0
  done
  echo "FAILED: $args (see $log)"'
touch $O/TRAINING_DONE
echo "$(date) training done; waiting for evaluators"
wait "${EVALS[@]}"
echo "$(date) all done"

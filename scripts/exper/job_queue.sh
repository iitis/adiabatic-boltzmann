#!/bin/bash
# Generic detached job queue: runs the commands in JOBFILE (one shell command per line, '#' comments allowed)
# with at most NPROC at once, in file order. Job k-th slot is pinned to CORES_PER_JOB cores starting at
# CORE0 + slot*CORES_PER_JOB (taskset), so queues on the same machine don't oversubscribe the CPU.
# Detaches itself with setsid + nohup (no controlling terminal: survives SSH disconnects and the parent shell);
# MF_FOREGROUND=1 runs attached. A failed job is retried twice (runs are resumable; the python3.14
# interpreter has crashed sporadically). Logs: LOGDIR/queue_<jobfile>.log and one log per job.
#   scripts/exper/job_queue.sh JOBFILE NPROC CORES_PER_JOB CORE0 LOGDIR
SELF=$(readlink -f "$0")
JOBS=$(readlink -f "$1"); NPROC=$2; CPJ=$3; CORE0=$4; LOGDIR=$5
cd "$(dirname "$SELF")/../.."
mkdir -p "$LOGDIR"; LOGDIR=$(readlink -f "$LOGDIR")
QLOG=$LOGDIR/queue_$(basename "$JOBS" .txt).log
if [ -z "$Q_DETACHED" ] && [ -z "$MF_FOREGROUND" ]; then
  Q_DETACHED=1 setsid nohup "$SELF" "$JOBS" "$NPROC" "$CPJ" "$CORE0" "$LOGDIR" >> "$QLOG" 2>&1 < /dev/null &
  echo "detached: pid $! (session leader), queue log $QLOG"
  exit 0
fi
echo "$(date '+%F %T') start: $(grep -cv '^\s*\(#\|$\)' "$JOBS") jobs, $NPROC parallel, $CPJ cores each from $CORE0"
export CPJ CORE0 LOGDIR
grep -v '^\s*\(#\|$\)' "$JOBS" | xargs -P "$NPROC" --process-slot-var=SLOT -d '\n' -I{} bash -c '
  cmd="{}"; lo=$((CORE0 + SLOT * CPJ)); hi=$((lo + CPJ - 1))
  log="$LOGDIR/job_$(echo "$cmd" | sed "s#.*scripts/##" | tr -c "A-Za-z0-9.\n" "_" | tr -s _ | cut -c1-150).log"
  for try in 1 2 3; do
    echo "$(date "+%F %T") [slot $SLOT cores $lo-$hi try $try] $cmd" >> "$log"
    OMP_NUM_THREADS=$CPJ OPENBLAS_NUM_THREADS=$CPJ MKL_NUM_THREADS=$CPJ \
      taskset -c "$lo-$hi" bash -c "$cmd" >> "$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { echo "$(date "+%F %T") OK: $cmd"; exit 0; }
    echo "$(date "+%F %T") exit $rc (try $try): $cmd"
  done
  echo "$(date "+%F %T") FAILED: $cmd (see $log)"'
echo "$(date '+%F %T') all jobs finished"

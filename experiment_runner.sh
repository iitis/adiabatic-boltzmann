#!/bin/bash
# run_benchmark.sh
# Runs each experiment as an isolated Python subprocess.
# On failure, retries once. On second failure, aborts the sweep.
# D-Wave QPU time is tracked cumulatively in time.json across all sessions.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKIP_FIRST_N=0
OUTPUT_DIR="results/"
LOG_FILE="benchmark.log"
SCRIPT="src/single_experiment.py"
MODEL="2d"
SIZES=(4 6 8)

N_NH_STEPS=2 # 25%, 50%, 75%, 100% of n_visible
LEARNING_RATES=(0.1 0.01)
SAMPLERS=("custom:metropolis")
RBMS=("pegasus")
H_VALUES=(0.5)
SEEDS=(1 42)
ITERATIONS=300
DWAVE_BUDGET_MS=1200000 # 20 minutes in milliseconds
TIME_FILE="time.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
  local msg="$1"
  echo "$msg" | tee -a "$LOG_FILE"
}

read_qpu_time_ms() {
  if [ ! -f "$TIME_FILE" ]; then
    echo "0"
    return
  fi
  python3 -c "
import json
try:
    print(int(json.load(open('$TIME_FILE')).get('time_ms', 0)))
except Exception:
    print(0)
"
}

format_min() {
  # Format milliseconds as minutes with 2 decimal places
  python3 -c "print(f'{$1 / 60000:.2f}')"
}

is_dwave() {
  local method=$1
  [[ "$method" == "pegasus" || "$method" == "zephyr" ]]
}

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

log "========================================================"
log "Benchmark started : $(date)"
log "Script            : $SCRIPT"
log "Output dir        : $OUTPUT_DIR"
log "Log file          : $LOG_FILE"
log "Skip first        : $SKIP_FIRST_N"
log "QPU budget        : $(format_min $DWAVE_BUDGET_MS) min"
log "========================================================"

used_ms=$(read_qpu_time_ms)
remaining_ms=$((DWAVE_BUDGET_MS - used_ms))
log "QPU time used so far : $(format_min $used_ms) min"
log "QPU time remaining   : $(format_min $((remaining_ms > 0 ? remaining_ms : 0))) min"

if [ "$used_ms" -ge "$DWAVE_BUDGET_MS" ] 2>/dev/null; then
  log ""
  log "[QPU BUDGET] Budget already exceeded before run started."
  log "             D-Wave experiments will be skipped."
fi

TOTAL=$((${#SIZES[@]} * N_NH_STEPS * ${#H_VALUES[@]} * ${#LEARNING_RATES[@]} * ${#SAMPLERS[@]} * ${#RBMS[@]} * ${#SEEDS[@]}))
log "Total experiments : $TOTAL"
log "========================================================"

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

DONE=0
SKIPPED=0
FAILED=0

for size in "${SIZES[@]}"; do

  # Compute n_visible and the 4 n_hidden steps (25%, 50%, 75%, 100% of n_visible)
  if [ "$MODEL" == "2d" ]; then
    n_visible=$((size * size))
  else
    n_visible=$size
  fi
  N_HIDDEN_VALUES=()
  for k in $(seq 1 $N_NH_STEPS); do
    N_HIDDEN_VALUES+=($((n_visible * k / N_NH_STEPS)))
  done

  log ""
  log "Size=$size  n_visible=$n_visible  n_hidden sweep: ${N_HIDDEN_VALUES[*]}"

  for n_hidden in "${N_HIDDEN_VALUES[@]}"; do
    local_alpha=$(python3 -c "print(f'{$n_hidden/$n_visible:.2f}')")

    for rbm in "${RBMS[@]}"; do
      for h in "${H_VALUES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for sampler_method in "${SAMPLERS[@]}"; do
            for seed in "${SEEDS[@]}"; do

              sampler="${sampler_method%%:*}"
              method="${sampler_method##*:}"
              idx=$((DONE + SKIPPED + FAILED + 1))

              # ── Skip first N ──────────────────────────────────────────────────
              if [ "$SKIPPED" -lt "$SKIP_FIRST_N" ]; then
                SKIPPED=$((SKIPPED + 1))
                log "[skip $SKIPPED/$SKIP_FIRST_N] size=$size nh=$n_hidden rbm=$rbm h=$h lr=$lr $sampler/$method seed=$seed"
                continue
              fi

              # ── QPU budget check ───────────────────────────────────────────────
              if is_dwave "$method"; then
                used_ms=$(read_qpu_time_ms)
                if [ "$used_ms" -ge "$DWAVE_BUDGET_MS" ] 2>/dev/null; then
                  log ""
                  log "[QPU BUDGET] $(format_min $used_ms) min used >= $(format_min $DWAVE_BUDGET_MS) min limit."
                  log "             Aborting remaining D-Wave experiments."
                  break 7
                fi
                qpu_info="  QPU used=$(format_min $used_ms)min"
              else
                qpu_info=""
              fi

              log ""
              log "[$idx/$TOTAL] size=$size nh=$n_hidden (α=$local_alpha) rbm=$rbm h=$h lr=$lr $sampler/$method seed=$seed$qpu_info"

              # ── Run with one retry ────────────────────────────────────────────
              success=0
              for attempt in 1 2; do
                if [ "$attempt" -eq 2 ]; then
                  log "  Retrying (attempt $attempt/2)..."
                fi

                python3 "$SCRIPT" \
                  --size "$size" \
                  --lr "$lr" \
                  --sampler "$sampler" \
                  --method "$method" \
                  --seed "$seed" \
                  --output-dir "$OUTPUT_DIR" \
                  --model "$MODEL" \
                  --n-hidden "$n_hidden" \
                  --h "$h" \
                  --iterations "$ITERATIONS" \
                  --rbm "$rbm" \
                  2>&1 | tee -a "$LOG_FILE"

                exit_code=${PIPESTATUS[0]}

                if [ "$exit_code" -eq 0 ]; then
                  success=1
                  break
                else
                  log "  Attempt $attempt failed (exit code $exit_code)"
                fi
              done

              if [ "$success" -eq 0 ]; then
                log "  Both attempts failed — skipping this experiment, continuing sweep."
                FAILED=$((FAILED + 1))
                continue
              fi

              DONE=$((DONE + 1))

            done
          done
        done
      done
    done
  done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

final_used_ms=$(read_qpu_time_ms)

log ""
log "========================================================"
log "Benchmark finished : $(date)"
log "Completed          : $DONE / $TOTAL"
log "Skipped (--skip)   : $SKIPPED"
log "Failed (skipped)   : $FAILED"
log "Total QPU time     : $(format_min $final_used_ms) min"
log "========================================================"

#!/usr/bin/env bash
# Unattended run of one robosuite task: build the offline dataset, fit the world
# model alone, clone the demos as a baseline, then train policies through the
# world model until the time budget is used up.
#
#   scripts/run_manip_overnight.sh <lift|square> [run-name]
#
# Everything goes to logs/<task>/<run-name>/ (default: a timestamp); the dataset
# to logs/<task>/data/. Stages that already produced their output are skipped,
# so a rerun with the same name continues. REPORT.md there is rewritten after
# every stage. Tunables (environment variables):
#   TOTAL_HOURS   wall-clock budget for the whole script        (default 10)
#   MAIN_HOURS    cap on the main policy run                    (default 5)
#   EXTRA_HOURS   cap on each follow-up policy run              (default 2)
#   EXTRA_RUNS    follow-up runs, in order, while budget remains (default
#                 "no_offline no_demo_starts seed1"; "" for none)
#   ONLY_DATA     1 = stop after building the dataset (CPU only, no GPU)
#   GPU_GB        per-process GPU memory cap in GiB             (default 8)
#   WORKERS       simulator processes for data collection       (default 16)
#   NICE          niceness of every process started here        (default 10)
#   BRANCH_STEPS  exploration transitions to collect (default 800000 for lift,
#                 1600000 for square)
#   WM_UPDATES    world-model updates in stage 1                (default 2000)
#   POLICY_ARGS   extra arguments for every train_policy call   (default none)
#   DEMOS         demonstration files (default: the task's ph and mh sets in
#                 DEMO_DIR, which must then be set)
set -u
cd "$(dirname "$0")/.."

TASK="${1:?usage: run_manip_overnight.sh <lift|square> [run-name]}"
RUN_NAME="${2:-$(date +%Y-%m-%d_%H-%M-%S)}"
TOTAL_HOURS="${TOTAL_HOURS:-10}"
MAIN_HOURS="${MAIN_HOURS:-5}"
EXTRA_HOURS="${EXTRA_HOURS:-2}"
EXTRA_RUNS="${EXTRA_RUNS-no_offline no_demo_starts seed1}"
ONLY_DATA="${ONLY_DATA:-0}"
GPU_GB="${GPU_GB:-8}"
WORKERS="${WORKERS:-16}"
NICE="${NICE:-10}"
WM_UPDATES="${WM_UPDATES:-2000}"
POLICY_ARGS="${POLICY_ARGS:-}"
case "$TASK" in
  lift) DEFAULT_BRANCH_STEPS=800000 ;;
  square) DEFAULT_BRANCH_STEPS=1600000 ;;
  *) echo "unknown task '$TASK'" >&2; exit 2 ;;
esac
BRANCH_STEPS="${BRANCH_STEPS:-$DEFAULT_BRANCH_STEPS}"

PY="nice -n $NICE .venv-lift/bin/python"
export PYTHONPATH=src
if [ -z "${DEMOS:-}" ]; then
  DEMO_DIR="${DEMO_DIR:?set DEMO_DIR to the folder with the robomimic low_dim datasets, or DEMOS to the files}"
  DEMOS=$($PY -c "from supermimic.manip.tasks import TASKS; print(' '.join('$DEMO_DIR/' + f for f in TASKS['$TASK'].demo_files))")
fi
DATA="${DATA:-logs/$TASK/data}"
RUN="logs/$TASK/$RUN_NAME"
mkdir -p "$RUN" "$DATA"
START=$(date +%s)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/overnight.log"; }
hours_left() { $PY -c "import time; print(max(0.0, $TOTAL_HOURS - (time.time() - $START) / 3600))"; }
min() { $PY -c "print(min($1, $2))"; }
summarize() { $PY -m supermimic.manip.scripts.summarize --data "$DATA" --run "$RUN" >>"$RUN/overnight.log" 2>&1; }

# One policy run in $1, capped at $2 hours, from the world model $3, with the
# remaining args; retried from its last checkpoint if the process dies.
policy_run() {
  local out="$1" cap="$2" wm="$3"; shift 3
  if [ -f "$out/policy_report.json" ]; then log "skip $out (finished)"; return 0; fi
  mkdir -p "$out"
  local attempt
  for attempt in 1 2 3; do
    local hours; hours=$(min "$cap" "$(hours_left)")
    log "policy run $out: attempt $attempt, up to $hours h"
    # shellcheck disable=SC2086
    if $PY -m supermimic.manip.scripts.train_policy --data "$DATA" \
        --wm "$wm" --out "$out" --hours "$hours" \
        --gpu-memory-gb "$GPU_GB" --resume "$@" $POLICY_ARGS \
        >>"$out/train.log" 2>&1; then
      return 0
    fi
    log "policy run $out failed (see $out/train.log)"
    sleep 30
  done
  return 1
}

log "run $RUN: task $TASK, budget $TOTAL_HOURS h, GPU cap $GPU_GB GiB, nice $NICE"

# demos_recorded.npz marks a dataset in the current layout.
if [ ! -f "$DATA/meta.json" ] || [ ! -f "$DATA/demos_recorded.npz" ]; then
  log "stage 0: building the dataset in $DATA"
  # shellcheck disable=SC2086
  $PY -m supermimic.manip.scripts.prepare_data --task "$TASK" --out "$DATA" \
    --demos $DEMOS --num-workers "$WORKERS" --branch-steps "$BRANCH_STEPS" \
    >>"$RUN/prepare_data.log" 2>&1 \
    || { log "stage 0 failed (see $RUN/prepare_data.log)"; exit 1; }
else
  log "stage 0: reusing $DATA"
fi
if [ "$ONLY_DATA" = "1" ]; then
  summarize
  log "ONLY_DATA=1: stopping after the dataset"
  exit 0
fi

if [ ! -f "$RUN/wm_eval.json" ]; then
  log "stage 1: world model alone"
  $PY -m supermimic.manip.scripts.train_wm --data "$DATA" --out "$RUN" \
    --updates "$WM_UPDATES" --gpu-memory-gb "$GPU_GB" >>"$RUN/train_wm.log" 2>&1 \
    || { log "stage 1 failed (see $RUN/train_wm.log)"; summarize; exit 1; }
else
  log "stage 1: reusing $RUN/world_model.pt"
fi
summarize

if [ ! -f "$RUN/bc/bc_report.json" ]; then
  log "baseline: behaviour cloning"
  mkdir -p "$RUN/bc"
  $PY -m supermimic.manip.scripts.train_bc --data "$DATA" --out "$RUN/bc" \
    --gpu-memory-gb "$GPU_GB" >>"$RUN/bc/train.log" 2>&1 \
    || log "baseline failed (see $RUN/bc/train.log); continuing"
  summarize
fi

log "stage 2: main policy run"
policy_run "$RUN/main" "$MAIN_HOURS" "$RUN/world_model.pt" --seed 0
summarize

for extra in $EXTRA_RUNS; do
  left=$(hours_left)
  if [ "$($PY -c "print(int($left < 0.5))")" = "1" ]; then
    log "skip $extra: $left h left"
    continue
  fi
  case "$extra" in
    seed1) policy_run "$RUN/seed1" "$EXTRA_HOURS" "$RUN/world_model.pt" --seed 1 ;;
    # Real episodes and imagined rollouts start only where the policy itself
    # has been; demos and branches still train the world model.
    no_demo_starts) policy_run "$RUN/no_demo_starts" "$EXTRA_HOURS" \
      "$RUN/world_model.pt" --seed 0 --bank-fraction 0 --start-mix 1 0 0 ;;
    # No offline data at all: an untrained world model that only ever sees the
    # policy's own rollouts.
    no_offline) policy_run "$RUN/no_offline" "$EXTRA_HOURS" scratch --seed 0 \
      --bank-fraction 0 --start-mix 1 0 0 --wm-mix 1 0 0 ;;
    *) log "unknown extra run '$extra'" ;;
  esac
  summarize
done

log "done after $($PY -c "import time; print(round((time.time() - $START) / 3600, 2))") h; report: $RUN/REPORT.md"

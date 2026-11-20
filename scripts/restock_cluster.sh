#!/bin/bash
# Restock on a Slurm cluster: submit the data, training or scoring job.
#
#   scripts/restock_cluster.sh data
#   scripts/restock_cluster.sh train <run-name> [train.py arguments]
#   scripts/restock_cluster.sh score <run-name> [score.py arguments]
#   scripts/restock_cluster.sh diagnose <run-name> [diagnose.py arguments]
#
# SBATCH_EXTRA adds sbatch options, e.g. --dependency=afterany:<job> to chain runs.
#
# The learner runs in condaenv/rl (torch 2.10); the simulator workers run in
# condaenv/genesis, the task's own simulator environment, and import the task
# code read-only from its repository. Training holds one GPU.
set -euo pipefail
LAB=${LAB:?set LAB to the directory that holds condaenv/, the task repository and this checkout}
ROOT=${ROOT:-$LAB/supermimic}
REPO=${REPO:-$LAB/ht_2026-11-13/gp_rl}
TAPES=${TAPES:-$LAB/ht_2026-11-13/demos/dHfull_all_ht_rns10h_img}
DATA=${DATA:-$ROOT/logs/restock/data}
PY=$LAB/condaenv/rl/bin/python
SIM_PY=$LAB/condaenv/genesis/bin/python
ENVS=${ENVS:-24}
# Nodes that are not the 64-physical-core class the tapes were recorded on.
EXCL64=${EXCL64:-$LAB/gp_dp_e2e/.excl64.txt}
SMT_EXCL=pax006,pax012,pax044,pax036,pax037,pax038,pax039,pax040,pax041,pax043,pax045,pax046,pax056,pax066

mode=$1; shift
mkdir -p "$ROOT/logs/restock/slurm"
common=(--chdir="$ROOT" -o "$ROOT/logs/restock/slurm/%x_%j.out"
  --export=ALL,PYTHONPATH=src,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1)
sim=(--repo "$REPO" --sim-python "$SIM_PY")

# Episode outcomes depend on the CPU class; the tapes reproduce on the
# 64-physical-core nodes, so data and scoring run there.
exclude=$(printf '%s,%s' "$(cat "$EXCL64")" "$SMT_EXCL" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd,)

case $mode in
  data)
    sbatch "${common[@]}" -J sm_restock_data -p batch,preempt -N 1 -c 10 --mem=28G -t 01:30:00 \
      --exclude="$exclude" --wrap \
      "grep -m1 'model name' /proc/cpuinfo; hostname; $PY -m supermimic.restock.scripts.prepare_data --tapes $TAPES --out $DATA ${sim[*]} --workers 8 $*"
    ;;
  train)
    run=$1; shift
    # The job imports a copy of the code taken now, so a later change to the
    # tree cannot reach a run that is still queued.
    snap=$ROOT/logs/restock/runs/$run/src
    rm -rf "$snap"; mkdir -p "$snap"
    cp -r "$ROOT/src/supermimic" "$snap/"
    find "$snap" -name __pycache__ -prune -exec rm -rf {} +
    sbatch ${SBATCH_EXTRA:-} "${common[@]}" --export=ALL,PYTHONPATH="$snap",PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1 \
      -J "sm_restock_$run" -p ${PARTITION:-gpu} --gres=gpu:1 -N 1 -c $((ENVS + 2)) \
      --mem=$((ENVS * 2 + 16))G -t ${TIME:-06:00:00} --wrap \
      "nvidia-smi --query-gpu=name --format=csv,noheader; $PY -m supermimic.restock.scripts.train --data $DATA --out $ROOT/logs/restock/runs/$run ${sim[*]} --num-envs $ENVS $*"
    ;;
  score)
    run=$1; shift
    sbatch ${SBATCH_EXTRA:-} "${common[@]}" -J "sm_restock_score_$run" -p batch,preempt -N 1 -c 4 --mem=12G -t 02:00:00 \
      --exclude="$exclude" --wrap \
      "grep -m1 'model name' /proc/cpuinfo; $PY -m supermimic.restock.scripts.score --run $ROOT/logs/restock/runs/$run --data $DATA ${sim[*]} $*"
    ;;
  diagnose)
    run=$1; shift
    sbatch ${SBATCH_EXTRA:-} "${common[@]}" -J "sm_restock_diag_$run" -p batch,preempt -N 1 -c 6 --mem=16G -t 01:00:00 \
      --exclude="$exclude" --wrap \
      "$PY -m supermimic.restock.scripts.diagnose --run $ROOT/logs/restock/runs/$run --data $DATA ${sim[*]} $*"
    ;;
  *) echo "usage: $0 data | train <run> [args] | score <run> [args] | diagnose <run> [args]" >&2; exit 2 ;;
esac

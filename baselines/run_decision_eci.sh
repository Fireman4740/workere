#!/usr/bin/env bash
# Full Laya run: every intra-sentence pair of the three -workere datasets, plus the dev splits
# the threshold is tuned on, plus dir3 on the gold pairs for direction.
#   run_decision_eci.sh <checkpoint> <variant>
# Needs `laya` importable (PYTHONPATH) and a python with torch + transformers + datasets.
set -euo pipefail
CKPT=$1 VAR=$2
PY=${PY:-/home/lenal/project/agentere/.venv/bin/python}
O=/home/lenal/project/workere/logs/decision_eci/full
HERE=$(dirname "$0")
export USE_TF=0
mkdir -p "$O"
run() { "$PY" "$HERE/decision_eci.py" predict --backend laya --checkpoint "$CKPT" --bf16 --batch-size 256 --forward-batch 32 "$@"; }
run --variant "$VAR" --dataset esc   --split dev   --out "$O/${CKPT}_${VAR}_esc_dev.jsonl"
run --variant "$VAR" --dataset maven --split dev   --out "$O/${CKPT}_${VAR}_maven_dev.jsonl"
run --variant "$VAR" --dataset esc   --split train --out "$O/${CKPT}_${VAR}_esc_train.jsonl"
run --variant "$VAR" --dataset ctb   --split train --out "$O/${CKPT}_${VAR}_ctb_train.jsonl"
run --variant "$VAR" --dataset maven --split test  --out "$O/${CKPT}_${VAR}_maven_test.jsonl"
if [ "$VAR" != dir3 ]; then
  for ds in "esc train" "ctb train" "maven test"; do
    set -- $ds
    run --variant dir3 --gold-only --dataset "$1" --split "$2" --out "$O/${CKPT}_dir3gold_$1_$2.jsonl"
  done
fi
echo "finished $CKPT $VAR"

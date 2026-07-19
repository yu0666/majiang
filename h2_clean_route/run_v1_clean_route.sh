#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/py10/bin/python3}"
GPU="${GPU:-1}"
RUN_ID="${RUN_ID:-20260715_v1_clean}"
RUN_DIR="${RUN_DIR:-$ROOT/V1_Clean_${RUN_ID}}"

BASE_MODEL="${BASE_MODEL:-models/qwen/Qwen2___5-1___5B-Instruct}"
V1_DATA_DIR="$RUN_DIR/v1_clean"
V1_DATA_FILE="$V1_DATA_DIR/v1_clean_sft.jsonl"
V1_ADAPTER="$ROOT/qwen-v1-clean-sft-${RUN_ID}"
V1_MERGED="$ROOT/models/Qwen-Mahjong-V1-Clean-SFT-${RUN_ID}-Merged"
EVAL_GAMES="${EVAL_GAMES:-200}"
SEEDS=(${SEEDS:-2026071801 2026072801 2026073801})

mkdir -p "$RUN_DIR/logs"
export PYTHONHASHSEED=0

stage() {
  local name="$1"
  shift
  local marker="$RUN_DIR/.done_${name}"
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] skip=$name"
    return
  fi
  echo "[$(date -Is)] start=$name"
  "$@" >"$RUN_DIR/logs/${name}.log" 2>&1
  touch "$marker"
  echo "[$(date -Is)] done=$name"
}

stage build_v1_clean_data "$PY" "$SCRIPT_DIR/build_v1_clean_sft_dataset.py" \
  --output-dir "$V1_DATA_DIR" \
  --rule-file sft_data_elite.jsonl \
  --rule-cap 30000 \
  --selfplay-file sft_data_selfplay_v3.jsonl \
  --selfplay-cap -1 \
  --dedupe \
  --eval-fraction 0.02

stage train_v1_clean env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_v1_mixed_sft.py \
  --model-path "$BASE_MODEL" \
  --data-file "$V1_DATA_FILE" \
  --output-dir "$V1_ADAPTER" \
  --epochs 2 \
  --learning-rate 8e-5 \
  --save-steps 100

stage merge_v1_clean "$PY" merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$V1_ADAPTER" \
  --output-dir "$V1_MERGED" \
  --device-map cpu

stage eval_v1_clean env CUDA_VISIBLE_DEVICES="$GPU" "$PY" rerun_v2_e2_ladder_3seeds.py \
  --output-dir "$RUN_DIR/eval_v1_clean" \
  --model-path "$V1_MERGED" \
  --no-adapter \
  --seeds "${SEEDS[@]}" \
  --games "$EVAL_GAMES" \
  --force

touch "$RUN_DIR/COMPLETE"
echo "[$(date -Is)] V1-clean route complete: $RUN_DIR"


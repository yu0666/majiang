#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/py10/bin/python3}"
RUN_ID="${RUN_ID:-20260714}"
RUN_DIR="${RUN_DIR:-$ROOT/Aligned_V1V2_${RUN_ID}}"
DATA_DIR="$RUN_DIR/data"
RULE_GAMES="${RULE_GAMES:-1400}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-1000}"
MASK_GAMES_PER_SEED="${MASK_GAMES_PER_SEED:-200}"
EVAL_GAMES="${EVAL_GAMES:-200}"

V1_ADAPTER="$ROOT/qwen-v1-aligned-sft-${RUN_ID}"
V1_MERGED="$ROOT/models/Qwen-Mahjong-V1-Aligned-SFT-${RUN_ID}-Merged"
V2_ADAPTER="$ROOT/qwen-v2-aligned-grpo-${RUN_ID}"
MIXED_DIR="$RUN_DIR/v1_mixed"
MIXED_FILE="$MIXED_DIR/v1_mixed_aligned.jsonl"

mkdir -p "$DATA_DIR" "$RUN_DIR/logs"
export PYTHONHASHSEED=0

echo "[$(date -Is)] stage=data_generation"
"$PY" generate_policy_sft_data.py \
  --teacher rule \
  --games "$RULE_GAMES" \
  --seed 2026071401 \
  --output-file "$DATA_DIR/rule_aligned.jsonl" \
  >"$RUN_DIR/logs/rule_data.log" 2>&1 &
PID_RULE=$!

"$PY" generate_mask_sft_data.py \
  --output-dir "$DATA_DIR/mask_l2" \
  --output-file mask_l2_aligned.jsonl \
  --seeds 2026071401 2026072401 2026073401 \
  --games-per-seed "$MASK_GAMES_PER_SEED" \
  --assistant-format json \
  --metadata minimal \
  >"$RUN_DIR/logs/mask_data.log" 2>&1 &
PID_MASK=$!

CUDA_VISIBLE_DEVICES=1 "$PY" generate_policy_sft_data.py \
  --teacher local_qwen \
  --games "$SELFPLAY_GAMES" \
  --seed 2026071501 \
  --model-path models/Qwen-Mahjong-V1-Mixed-SFT-Merged \
  --temperature 0.7 \
  --max-new-tokens 16 \
  --output-file "$DATA_DIR/selfplay_aligned.jsonl" \
  >"$RUN_DIR/logs/selfplay_data.log" 2>&1 &
PID_SELFPLAY=$!

wait "$PID_RULE"
wait "$PID_MASK"
wait "$PID_SELFPLAY"

echo "[$(date -Is)] stage=build_mixed_dataset"
"$PY" build_v1_mixed_sft_dataset.py \
  --output-dir "$MIXED_DIR" \
  --output-file "$(basename "$MIXED_FILE")" \
  --summary-file v1_mixed_aligned_summary.json \
  --rule-file "$DATA_DIR/rule_aligned.jsonl" \
  --rule-cap 30000 \
  --selfplay-file "$DATA_DIR/selfplay_aligned.jsonl" \
  --selfplay-cap -1 \
  --mask-file "$DATA_DIR/mask_l2/mask_l2_aligned.jsonl" \
  --mask-cap -1 \
  --mask-repeat 2 \
  --dedupe \
  >"$RUN_DIR/logs/build_mixed.log" 2>&1

echo "[$(date -Is)] stage=v1_sft"
CUDA_VISIBLE_DEVICES=1 "$PY" train_v1_mixed_sft.py \
  --data-file "$MIXED_FILE" \
  --output-dir "$V1_ADAPTER" \
  --epochs 2 \
  --learning-rate 8e-5 \
  --save-steps 100 \
  >"$RUN_DIR/logs/train_v1.log" 2>&1

echo "[$(date -Is)] stage=merge_v1"
"$PY" merge_lora_adapter.py \
  --base-model models/qwen/Qwen2___5-1___5B-Instruct \
  --adapter "$V1_ADAPTER" \
  --output-dir "$V1_MERGED" \
  --device-map cpu \
  >"$RUN_DIR/logs/merge_v1.log" 2>&1

echo "[$(date -Is)] stage=v2_grpo"
CUDA_VISIBLE_DEVICES=1 "$PY" train_v2_grpo.py \
  --model-path "$V1_MERGED" \
  --data-file "$MIXED_FILE" \
  --output-dir "$V2_ADAPTER" \
  --max-steps 300 \
  --dataset-limit 6000 \
  --learning-rate 3e-6 \
  --beta 0.03 \
  --save-steps 50 \
  --logging-steps 5 \
  >"$RUN_DIR/logs/train_v2.log" 2>&1

echo "[$(date -Is)] stage=evaluate_v1"
CUDA_VISIBLE_DEVICES=1 "$PY" rerun_v2_e2_ladder_3seeds.py \
  --output-dir "$RUN_DIR/eval_v1" \
  --model-path "$V1_MERGED" \
  --no-adapter \
  --seeds 2026071401 2026072401 2026073401 \
  --games "$EVAL_GAMES" \
  --force \
  >"$RUN_DIR/logs/eval_v1.log" 2>&1

echo "[$(date -Is)] stage=evaluate_v2"
CUDA_VISIBLE_DEVICES=1 "$PY" rerun_v2_e2_ladder_3seeds.py \
  --output-dir "$RUN_DIR/eval_v2" \
  --model-path "$V1_MERGED" \
  --adapter-path "$V2_ADAPTER/best_grpo_adapter" \
  --seeds 2026071401 2026072401 2026073401 \
  --games "$EVAL_GAMES" \
  --force \
  >"$RUN_DIR/logs/eval_v2.log" 2>&1

echo "[$(date -Is)] stage=complete"
touch "$RUN_DIR/COMPLETE"

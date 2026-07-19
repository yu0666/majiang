#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/py10/bin/python3}"
GPU="${GPU:-1}"
RUN_ID="${RUN_ID:-20260715_h2_gate_v1}"
RUN_DIR="${RUN_DIR:-$ROOT/H2_Gate_Route_${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

POLICY_MODEL="${POLICY_MODEL:-models/Qwen-Mahjong-V1-Aligned-SFT-20260714-Merged}"
POLICY_ADAPTER="${POLICY_ADAPTER-qwen-v2-aligned-grpo-20260714/best_grpo_adapter}"
GATE_BASE_MODEL="${GATE_BASE_MODEL:-models/Qwen-Mahjong-V2-Aligned-GRPO-20260714-Merged}"

TARGET_STATES="${TARGET_STATES:-500}"
ORACLE_CHUNK_SIZE="${ORACLE_CHUNK_SIZE:-50}"
ORACLE_MAX_GAMES="${ORACLE_MAX_GAMES:-80}"
ROLLOUTS_PER_MODE="${ROLLOUTS_PER_MODE:-4}"
GATE_SFT_EPOCHS="${GATE_SFT_EPOCHS:-2}"
GATE_GRPO_STEPS="${GATE_GRPO_STEPS:-150}"
EVAL_GAMES="${EVAL_GAMES:-500}"
SEEDS=(${SEEDS:-2026071701 2026072701 2026073701})

GATE_ORACLE="$RUN_DIR/gate_oracle"
GATE_DATA="$RUN_DIR/gate_data"
GATE_SFT="$ROOT/qwen-h2-gate-sft-${RUN_ID}"
GATE_SFT_MERGED="$ROOT/models/Qwen-Mahjong-H2-Gate-SFT-${RUN_ID}-Merged"
GATE_GRPO="$ROOT/qwen-h2-gate-grpo-${RUN_ID}"
EVAL_DIR="$RUN_DIR/evaluation"

mkdir -p "$RUN_DIR/logs" "$GATE_DATA"
export PYTHONHASHSEED=0

REWARD_ARGS=(
  --return-clip 0
  --hu-bonus 5
  --fan-bonus 3
  --dealin-penalty 20
  --tail-alpha 0.2
  --tail-risk-weight 0.5
  --catastrophic-loss-threshold 200
  --catastrophic-loss-penalty 40
)

run_stage() {
  local name="$1"
  shift
  local marker="$RUN_DIR/.done_${name}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $name: $*"
    return
  fi
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] skip=$name"
    return
  fi
  echo "[$(date -Is)] start=$name"
  "$@" >"$RUN_DIR/logs/${name}.log" 2>&1
  touch "$marker"
  echo "[$(date -Is)] done=$name"
}

run_stage gate_oracle env CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT_DIR/collect_gate_rollouts_chunked.py" \
  --output-dir "$GATE_ORACLE" \
  --target-states "$TARGET_STATES" \
  --chunk-size "$ORACLE_CHUNK_SIZE" \
  --max-games-per-chunk "$ORACLE_MAX_GAMES" \
  --seed 2026071701 \
  --rollout-seed 817100 \
  --rollouts-per-mode "$ROLLOUTS_PER_MODE" \
  --threat-fold-threshold 0.7 \
  --model-path "$POLICY_MODEL" \
  --adapter-path "$POLICY_ADAPTER" \
  --max-new-tokens 64

run_stage gate_data "$PY" build_gate_training_data.py \
  --input "$GATE_ORACLE/gate_rollout_states.jsonl" \
  --output-dir "$GATE_DATA" \
  --reward-scale 20 \
  --min-reward-range 2 \
  "${REWARD_ARGS[@]}"

run_stage gate_sft env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_sft.py \
  --model-path "$GATE_BASE_MODEL" \
  --data-file "$GATE_DATA/gate_sft.jsonl" \
  --output-dir "$GATE_SFT" \
  --min-examples 200 \
  --epochs "$GATE_SFT_EPOCHS" \
  --learning-rate 2e-5 \
  --save-steps 50

run_stage gate_sft_merge "$PY" merge_lora_adapter.py \
  --base-model "$GATE_BASE_MODEL" \
  --adapter "$GATE_SFT" \
  --output-dir "$GATE_SFT_MERGED" \
  --device-map cpu

run_stage gate_grpo env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_grpo.py \
  --model-path "$GATE_SFT_MERGED" \
  --data-file "$GATE_DATA/gate_grpo.jsonl" \
  --output-dir "$GATE_GRPO" \
  --max-steps "$GATE_GRPO_STEPS" \
  --learning-rate 5e-6 \
  --beta 0.05 \
  --save-steps 25

run_stage h2_eval env CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT_DIR/evaluate_h2_gate_ladder.py" \
  --output-dir "$EVAL_DIR" \
  --seeds "${SEEDS[@]}" \
  --games "$EVAL_GAMES" \
  --policy-model "$POLICY_MODEL" \
  --policy-adapter "$POLICY_ADAPTER" \
  --gate-model "$GATE_SFT_MERGED" \
  --gate-adapter "$GATE_GRPO/best_grpo_adapter" \
  --force

if [[ "$DRY_RUN" != "1" ]]; then
  touch "$RUN_DIR/COMPLETE"
  echo "[$(date -Is)] H2 gate route complete: $RUN_DIR"
fi

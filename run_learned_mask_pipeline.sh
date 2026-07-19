#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/py10/bin/python3}"
RUN_ID="${RUN_ID:-20260715_riskgate_v1}"
RUN_DIR="${RUN_DIR:-$ROOT/Learned_MASK_${RUN_ID}}"
GPU="${GPU:-1}"
SEEDS=(2026082101 2026083101 2026084101)
EVAL_GAMES="${EVAL_GAMES:-500}"

V1_MODEL="$ROOT/models/Qwen-Mahjong-V1-Aligned-SFT-20260714-Merged"
V2_ADAPTER="$ROOT/qwen-v2-aligned-grpo-20260714/best_grpo_adapter"
V2_MERGED="$ROOT/models/Qwen-Mahjong-V2-Aligned-GRPO-20260714-Merged"

RR_ORACLE="$RUN_DIR/reranker_oracle"
RR_DATA="$RUN_DIR/reranker_data"
RR_SFT="$ROOT/qwen-v2-risk-reranker-sft-${RUN_ID}"
RR_SFT_MERGED="$ROOT/models/Qwen-Mahjong-V2-Risk-Reranker-SFT-${RUN_ID}-Merged"
RR_GRPO="$ROOT/qwen-v2-risk-reranker-grpo-${RUN_ID}"

GATE_ORACLE="$RUN_DIR/gate_oracle"
GATE_DATA="$RUN_DIR/gate_data"
GATE_SFT="$ROOT/qwen-v2-mask-gate-sft-${RUN_ID}"
GATE_SFT_MERGED="$ROOT/models/Qwen-Mahjong-V2-MASK-Gate-SFT-${RUN_ID}-Merged"
GATE_GRPO="$ROOT/qwen-v2-mask-gate-grpo-${RUN_ID}"

mkdir -p "$RUN_DIR/logs" "$RR_DATA" "$GATE_DATA"
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

stage reranker_oracle env CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_candidate_oracle.py \
  --output-dir "$RR_ORACLE" \
  --seed 2026071601 --rollout-seed 816100 \
  --max-games 240 --target-exploit 160 --target-safe 120 --target-deceive 120 \
  --rollouts-per-action 16 --max-candidates 6 --augment-modes \
  --continuation rule_mask --opponent-style responsive \
  --threat-fold-threshold 0.7 --defender-threat-model blend \
  --defender-tell-weight 0.3 --defender-tell-window 6 \
  --mask-forced-deceive off --mask-deceive-style threat \
  --mask-threat-gate-threshold 0.7 --mask-threat-response-model blend \
  --mask-threat-response-tell-weight 0.3 --mask-threat-require-real-target \
  --mask-threat-target-max-shanten 1 --mask-threat-target-signal mc \
  --mask-threat-target-prob-threshold 0.78 --mask-threat-max-start-shanten 2 \
  --backend local_qwen --model-path "$V1_MODEL" --adapter-path "$V2_ADAPTER"

stage reranker_sft_data "$PY" build_reranker_sft_dataset.py \
  "$RR_ORACLE/candidate_oracle_states.jsonl" \
  --output "$RR_DATA/reranker_sft.jsonl" --accept-best --min-rollouts 16 \
  --completion-format action_only \
  "${REWARD_ARGS[@]}"

stage reranker_sft env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_reranker_sft.py \
  --model-path "$V2_MERGED" --data-file "$RR_DATA/reranker_sft.jsonl" \
  --output-dir "$RR_SFT" --min-examples 300 --min-modes 3 \
  --epochs 2 --learning-rate 2e-5 --save-steps 50

stage reranker_sft_merge "$PY" merge_lora_adapter.py \
  --base-model "$V2_MERGED" --adapter "$RR_SFT" \
  --output-dir "$RR_SFT_MERGED" --device-map cpu

stage reranker_grpo_data "$PY" build_reranker_grpo_dataset.py \
  --input "$RR_ORACLE/candidate_oracle_states.jsonl" \
  --output "$RR_DATA/reranker_grpo.jsonl" --completion-format action_only \
  --no-include-reference --permutations-per-state 2 --reward-scale 20 \
  --min-reward-range 2 "${REWARD_ARGS[@]}"

stage reranker_grpo env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_reranker_grpo.py \
  --model-path "$RR_SFT_MERGED" --data-file "$RR_DATA/reranker_grpo.jsonl" \
  --output-dir "$RR_GRPO" --max-steps 100 --learning-rate 5e-6 \
  --beta 0.05 --save-steps 20

stage gate_oracle env CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_gate_mode_oracle.py \
  --output-dir "$GATE_ORACLE" --target-states 400 \
  --seed 2026072601 --rollout-seed 826100 --rollouts-per-mode 8 \
  --max-games 240 --opponent-style responsive \
  --backend local_qwen --model-path "$V1_MODEL" --adapter-path "$V2_ADAPTER" \
  "${REWARD_ARGS[@]}"

stage gate_data "$PY" build_gate_training_dataset.py \
  --input "$GATE_ORACLE/gate_mode_oracle.jsonl" --output-dir "$GATE_DATA" \
  --reward-scale 20 --min-reward-range 2

stage gate_sft env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_sft.py \
  --model-path "$V2_MERGED" --data-file "$GATE_DATA/gate_sft.jsonl" \
  --output-dir "$GATE_SFT" --min-examples 200 --epochs 2 \
  --learning-rate 2e-5 --save-steps 50

stage gate_sft_merge "$PY" merge_lora_adapter.py \
  --base-model "$V2_MERGED" --adapter "$GATE_SFT" \
  --output-dir "$GATE_SFT_MERGED" --device-map cpu

stage gate_grpo env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_grpo.py \
  --model-path "$GATE_SFT_MERGED" --data-file "$GATE_DATA/gate_grpo.jsonl" \
  --output-dir "$GATE_GRPO" --max-steps 100 --learning-rate 5e-6 \
  --beta 0.05 --save-steps 20

stage evaluation env CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_unified_mask_comparison.py \
  --output-dir "$RUN_DIR/evaluation" --seeds "${SEEDS[@]}" --games "$EVAL_GAMES" \
  --model-path "$V1_MODEL" --adapter-path "$V2_ADAPTER" \
  --gate-model-path "$GATE_SFT_MERGED" \
  --gate-adapter-path "$GATE_GRPO/best_grpo_adapter" \
  --reranker-model-path "$RR_SFT_MERGED" \
  --reranker-adapter-path "$RR_GRPO/best_grpo_adapter" --force

touch "$RUN_DIR/COMPLETE"
echo "[$(date -Is)] pipeline=complete output=$RUN_DIR"

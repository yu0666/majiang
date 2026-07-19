#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/py10/bin/python3}"
GPU="${GPU:-1}"
RUN_ID="${RUN_ID:-20260716}"
RUN_DIR="${RUN_DIR:-$ROOT/MASK_Full_${RUN_ID}}"
LOG_DIR="$RUN_DIR/logs"
BASE_MODEL="${BASE_MODEL:-models/Qwen-Mahjong-V1-Aligned-SFT-20260714-Merged}"
POLICY_ADAPTER="${POLICY_ADAPTER:-qwen-v2-aligned-grpo-20260714/best_grpo_adapter}"

RERANK_ORACLE="$RUN_DIR/reranker_oracle"
RERANK_DATA="$RUN_DIR/reranker_data"
RERANK_SFT_ADAPTER="$ROOT/qwen-reranker-risk-sft-${RUN_ID}"
RERANK_SFT_MERGED="$ROOT/models/Qwen-Mahjong-Reranker-Risk-SFT-${RUN_ID}-Merged"
RERANK_GRPO="$ROOT/qwen-reranker-risk-grpo-${RUN_ID}"

GATE_ORACLE="$RUN_DIR/gate_oracle"
GATE_DATA="$RUN_DIR/gate_data"
GATE_SFT_ADAPTER="$ROOT/qwen-mask-gate-sft-${RUN_ID}"
GATE_SFT_MERGED="$ROOT/models/Qwen-Mahjong-Gate-SFT-${RUN_ID}-Merged"
GATE_GRPO="$ROOT/qwen-mask-gate-grpo-${RUN_ID}"

mkdir -p "$LOG_DIR" "$RERANK_DATA" "$GATE_DATA"
rm -f "$RUN_DIR/FAILED" "$RUN_DIR/COMPLETE"
export PYTHONHASHSEED=0

stage() {
  printf '[%s] stage=%s\n' "$(date -Is)" "$1" | tee -a "$RUN_DIR/pipeline.log"
}

fail() {
  status=$?
  printf '[%s] failed status=%s\n' "$(date -Is)" "$status" | tee -a "$RUN_DIR/pipeline.log"
  touch "$RUN_DIR/FAILED"
  exit "$status"
}
trap fail ERR

stage reranker_counterfactual_collection
CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_candidate_oracle.py \
  --output-dir "$RERANK_ORACLE" \
  --seed 2026071601 \
  --rollout-seed 816000 \
  --max-games 500 \
  --target-exploit 240 \
  --target-safe 180 \
  --target-deceive 180 \
  --rollouts-per-action 12 \
  --max-candidates 6 \
  --augment-modes \
  --continuation rule_mask \
  --threat-fold-threshold 0.7 \
  --defender-threat-model blend \
  --defender-tell-weight 0.3 \
  --mask-forced-deceive off \
  --mask-deceive-style threat \
  --mask-threat-gate-threshold 0.7 \
  --mask-threat-response-model blend \
  --mask-threat-response-tell-weight 0.3 \
  --mask-threat-require-real-target \
  --mask-threat-target-max-shanten 1 \
  --mask-threat-target-signal mc \
  --mask-threat-target-prob-threshold 0.78 \
  --backend local_qwen \
  --model-path "$BASE_MODEL" \
  --adapter-path "$POLICY_ADAPTER" \
  >"$LOG_DIR/reranker_oracle.log" 2>&1

stage reranker_risk_data
"$PY" build_reranker_sft_dataset.py \
  "$RERANK_ORACLE/candidate_oracle_states.jsonl" \
  --output "$RERANK_DATA/reranker_risk_sft.jsonl" \
  --min-rollouts 12 \
  --confidence 0.80 \
  --min-mean-advantage 2 \
  --min-ci-lower -5 \
  --min-anchor-advantage 1 \
  >"$LOG_DIR/build_reranker_sft.log" 2>&1

"$PY" build_reranker_grpo_dataset.py \
  --input "$RERANK_ORACLE/candidate_oracle_states.jsonl" \
  --output "$RERANK_DATA/reranker_risk_grpo.jsonl" \
  --min-reward-range 2 \
  --reward-scale 20 \
  --permutations-per-state 3 \
  --completion-format action_only \
  --no-include-reference \
  >"$LOG_DIR/build_reranker_grpo.log" 2>&1

stage reranker_sft
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_reranker_sft.py \
  --model-path "$BASE_MODEL" \
  --data-file "$RERANK_DATA/reranker_risk_sft.jsonl" \
  --output-dir "$RERANK_SFT_ADAPTER" \
  --min-examples 100 \
  --min-modes 2 \
  --epochs 2 \
  --learning-rate 2e-5 \
  --save-steps 50 \
  >"$LOG_DIR/train_reranker_sft.log" 2>&1

stage reranker_sft_merge
"$PY" merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$RERANK_SFT_ADAPTER" \
  --output-dir "$RERANK_SFT_MERGED" \
  --device-map cpu \
  >"$LOG_DIR/merge_reranker_sft.log" 2>&1

stage reranker_grpo
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_reranker_grpo.py \
  --model-path "$RERANK_SFT_MERGED" \
  --data-file "$RERANK_DATA/reranker_risk_grpo.jsonl" \
  --output-dir "$RERANK_GRPO" \
  --max-steps 100 \
  --learning-rate 5e-6 \
  --beta 0.05 \
  --save-steps 20 \
  >"$LOG_DIR/train_reranker_grpo.log" 2>&1

stage learned_gate_counterfactual_collection
CUDA_VISIBLE_DEVICES="$GPU" "$PY" collect_gate_rollouts.py \
  --output-dir "$GATE_ORACLE" \
  --model-path "$BASE_MODEL" \
  --adapter-path "$POLICY_ADAPTER" \
  --seed 2026071601 \
  --rollout-seed 916000 \
  --max-states 400 \
  --max-games 300 \
  --rollouts-per-mode 8 \
  >"$LOG_DIR/gate_oracle.log" 2>&1

stage learned_gate_data
"$PY" build_gate_training_data.py \
  --input "$GATE_ORACLE/gate_rollout_states.jsonl" \
  --output-dir "$GATE_DATA" \
  --min-reward-range 2 \
  --reward-scale 20 \
  >"$LOG_DIR/build_gate_data.log" 2>&1

stage learned_gate_sft
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_sft.py \
  --model-path "$BASE_MODEL" \
  --data-file "$GATE_DATA/gate_sft.jsonl" \
  --output-dir "$GATE_SFT_ADAPTER" \
  --min-examples 100 \
  --epochs 2 \
  --learning-rate 2e-5 \
  >"$LOG_DIR/train_gate_sft.log" 2>&1

stage learned_gate_sft_merge
"$PY" merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$GATE_SFT_ADAPTER" \
  --output-dir "$GATE_SFT_MERGED" \
  --device-map cpu \
  >"$LOG_DIR/merge_gate_sft.log" 2>&1

stage learned_gate_grpo
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_grpo.py \
  --model-path "$GATE_SFT_MERGED" \
  --data-file "$GATE_DATA/gate_grpo.jsonl" \
  --output-dir "$GATE_GRPO" \
  --max-steps 80 \
  --learning-rate 5e-6 \
  --beta 0.05 \
  >"$LOG_DIR/train_gate_grpo.log" 2>&1

stage serial_3seed_500game_evaluation
CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_mask_full_comparison.py \
  --output-dir "$RUN_DIR/evaluation" \
  --model-path "$BASE_MODEL" \
  --adapter-path "$POLICY_ADAPTER" \
  --gate-model-path "$GATE_SFT_MERGED" \
  --gate-adapter-path "$GATE_GRPO/best_grpo_adapter" \
  --reranker-model-path "$RERANK_SFT_MERGED" \
  --reranker-adapter-path "$RERANK_GRPO/best_grpo_adapter" \
  --seeds 2026071601 2026072601 2026073601 \
  --games 500 \
  --force \
  >"$LOG_DIR/evaluation.log" 2>&1

stage complete
touch "$RUN_DIR/COMPLETE"

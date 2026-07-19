#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/py10/bin/python3}"
GPU="${GPU:-1}"
RUN_ID="${RUN_ID:-20260716_h2_full_learned}"
RUN_DIR="${RUN_DIR:-$ROOT/H2_Full_Learned_${RUN_ID}}"
LOG_DIR="$RUN_DIR/logs"
MARKER_DIR="$RUN_DIR/markers"

BASE_MODEL="${BASE_MODEL:-models/Qwen-Mahjong-V1-Clean-SFT-20260715_v1_clean-Merged}"
POLICY_ADAPTER="${POLICY_ADAPTER-}"
OLD_GATE_MODEL="${OLD_GATE_MODEL:-models/Qwen-Mahjong-H2-Gate-SFT-20260715_h2_gate_v1clean-Merged}"
OLD_GATE_ADAPTER="${OLD_GATE_ADAPTER:-qwen-h2-gate-grpo-20260715_h2_gate_v1clean/best_grpo_adapter}"
DEFENDER_MODEL="${DEFENDER_MODEL:-Defender_danger_model/danger_model.pth}"

GATE_ORACLE="$RUN_DIR/gate_oracle"
GATE_DATA="$RUN_DIR/gate_data"
GATE_SFT_ADAPTER="$ROOT/qwen-h2-gate-sft-${RUN_ID}"
GATE_SFT_MERGED="$ROOT/models/Qwen-Mahjong-H2-Gate-SFT-${RUN_ID}-Merged"
GATE_GRPO="$ROOT/qwen-h2-gate-grpo-${RUN_ID}"

RERANK_ORACLE="$RUN_DIR/reranker_oracle"
RERANK_DATA="$RUN_DIR/reranker_data"
RERANK_SFT_ADAPTER="$ROOT/qwen-h2-reranker-sft-${RUN_ID}"
RERANK_SFT_MERGED="$ROOT/models/Qwen-Mahjong-H2-Reranker-SFT-${RUN_ID}-Merged"
RERANK_GRPO="$ROOT/qwen-h2-reranker-grpo-${RUN_ID}"

SEEDS=(${SEEDS:-2026071601 2026072601 2026073601})
EVAL_GAMES="${EVAL_GAMES:-500}"
EVAL_PARALLEL_WORKERS="${EVAL_PARALLEL_WORKERS:-1}"
EVAL_FORCE="${EVAL_FORCE:-0}"
GATE_TARGET_STATES="${GATE_TARGET_STATES:-500}"
GATE_CHUNK_SIZE="${GATE_CHUNK_SIZE:-50}"
GATE_MAX_GAMES_PER_CHUNK="${GATE_MAX_GAMES_PER_CHUNK:-100}"
GATE_ROLLOUTS_PER_MODE="${GATE_ROLLOUTS_PER_MODE:-6}"

RERANK_MAX_GAMES="${RERANK_MAX_GAMES:-600}"
RERANK_TARGET_EXPLOIT="${RERANK_TARGET_EXPLOIT:-260}"
RERANK_TARGET_SAFE="${RERANK_TARGET_SAFE:-200}"
RERANK_TARGET_DECEIVE="${RERANK_TARGET_DECEIVE:-200}"
RERANK_ROLLOUTS_PER_ACTION="${RERANK_ROLLOUTS_PER_ACTION:-12}"

mkdir -p "$LOG_DIR" "$MARKER_DIR" "$GATE_DATA" "$RERANK_DATA"
rm -f "$RUN_DIR/FAILED" "$RUN_DIR/COMPLETE"
export PYTHONHASHSEED=0

log_stage() {
  printf '[%s] stage=%s\n' "$(date -Is)" "$1" | tee -a "$RUN_DIR/pipeline.log"
}

run_stage() {
  local name="$1"
  shift
  local marker="$MARKER_DIR/$name.done"
  if [[ -f "$marker" && "${FORCE_STAGE:-0}" != "1" ]]; then
    log_stage "skip_${name}"
    return
  fi
  log_stage "$name"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] %q ' "$@" | tee -a "$RUN_DIR/pipeline.log"
    printf '\n' | tee -a "$RUN_DIR/pipeline.log"
    return
  fi
  "$@"
  touch "$marker"
}

fail() {
  local status=$?
  printf '[%s] failed status=%s\n' "$(date -Is)" "$status" | tee -a "$RUN_DIR/pipeline.log"
  touch "$RUN_DIR/FAILED"
  exit "$status"
}
trap fail ERR

run_stage gate_counterfactual_collection \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" h2_gate_route/collect_gate_rollouts_chunked.py \
    --output-dir "$GATE_ORACLE" \
    --target-states "$GATE_TARGET_STATES" \
    --chunk-size "$GATE_CHUNK_SIZE" \
    --max-games-per-chunk "$GATE_MAX_GAMES_PER_CHUNK" \
    --seed 2026071701 \
    --rollout-seed 817100 \
    --rollouts-per-mode "$GATE_ROLLOUTS_PER_MODE" \
    --threat-fold-threshold 0.7 \
    --defender-threat-model learned \
    --defender-learned-model-path "$DEFENDER_MODEL" \
    --model-path "$BASE_MODEL" \
    --adapter-path "$POLICY_ADAPTER" \
    >"$LOG_DIR/gate_oracle.log" 2>&1

run_stage gate_data \
  "$PY" build_gate_training_data.py \
    --input "$GATE_ORACLE/gate_rollout_states.jsonl" \
    --output-dir "$GATE_DATA" \
    --min-reward-range 2 \
    --reward-scale 20 \
    >"$LOG_DIR/build_gate_data.log" 2>&1

run_stage gate_sft \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_sft.py \
    --model-path "$BASE_MODEL" \
    --data-file "$GATE_DATA/gate_sft.jsonl" \
    --output-dir "$GATE_SFT_ADAPTER" \
    --min-examples 120 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --save-steps 50 \
    >"$LOG_DIR/train_gate_sft.log" 2>&1

run_stage gate_sft_merge \
  "$PY" merge_lora_adapter.py \
    --base-model "$BASE_MODEL" \
    --adapter "$GATE_SFT_ADAPTER" \
    --output-dir "$GATE_SFT_MERGED" \
    --device-map cpu \
    >"$LOG_DIR/merge_gate_sft.log" 2>&1

run_stage gate_grpo \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_gate_grpo.py \
    --model-path "$GATE_SFT_MERGED" \
    --data-file "$GATE_DATA/gate_grpo.jsonl" \
    --output-dir "$GATE_GRPO" \
    --max-steps 120 \
    --learning-rate 5e-6 \
    --beta 0.05 \
    --save-steps 20 \
    >"$LOG_DIR/train_gate_grpo.log" 2>&1

run_stage reranker_counterfactual_collection \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" run_candidate_oracle.py \
    --output-dir "$RERANK_ORACLE" \
    --seed 2026071801 \
    --rollout-seed 818100 \
    --max-games "$RERANK_MAX_GAMES" \
    --target-exploit "$RERANK_TARGET_EXPLOIT" \
    --target-safe "$RERANK_TARGET_SAFE" \
    --target-deceive "$RERANK_TARGET_DECEIVE" \
    --rollouts-per-action "$RERANK_ROLLOUTS_PER_ACTION" \
    --max-candidates 6 \
    --augment-modes \
    --continuation rule_mask \
    --threat-fold-threshold 0.7 \
    --defender-threat-model learned \
    --defender-learned-model-path "$DEFENDER_MODEL" \
    --mask-forced-deceive off \
    --mask-deceive-style threat \
    --mask-threat-max-result-shanten 0 \
    --mask-threat-max-shanten-regret 0 \
    --mask-threat-min-ukeire-ratio 1.0 \
    --mask-threat-gate-threshold 0.7 \
    --mask-threat-gate-margin 0.12 \
    --mask-threat-min-delta 0.03 \
    --mask-threat-gate-mode cross \
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

run_stage reranker_data \
  "$PY" build_reranker_sft_dataset.py \
    "$RERANK_ORACLE/candidate_oracle_states.jsonl" \
    --output "$RERANK_DATA/reranker_sft.jsonl" \
    --min-rollouts "$RERANK_ROLLOUTS_PER_ACTION" \
    --confidence 0.80 \
    --min-mean-advantage 2 \
    --min-ci-lower -5 \
    --min-anchor-advantage 1 \
    >"$LOG_DIR/build_reranker_sft.log" 2>&1

run_stage reranker_grpo_data \
  "$PY" build_reranker_grpo_dataset.py \
    --input "$RERANK_ORACLE/candidate_oracle_states.jsonl" \
    --output "$RERANK_DATA/reranker_grpo.jsonl" \
    --min-reward-range 2 \
    --reward-scale 20 \
    --permutations-per-state 3 \
    --completion-format action_only \
    --no-include-reference \
    >"$LOG_DIR/build_reranker_grpo.log" 2>&1

run_stage reranker_sft \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_reranker_sft.py \
    --model-path "$BASE_MODEL" \
    --data-file "$RERANK_DATA/reranker_sft.jsonl" \
    --output-dir "$RERANK_SFT_ADAPTER" \
    --min-examples 120 \
    --min-modes 2 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --save-steps 50 \
    >"$LOG_DIR/train_reranker_sft.log" 2>&1

run_stage reranker_sft_merge \
  "$PY" merge_lora_adapter.py \
    --base-model "$BASE_MODEL" \
    --adapter "$RERANK_SFT_ADAPTER" \
    --output-dir "$RERANK_SFT_MERGED" \
    --device-map cpu \
    >"$LOG_DIR/merge_reranker_sft.log" 2>&1

run_stage reranker_grpo \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_reranker_grpo.py \
    --model-path "$RERANK_SFT_MERGED" \
    --data-file "$RERANK_DATA/reranker_grpo.jsonl" \
    --output-dir "$RERANK_GRPO" \
    --max-steps 120 \
    --learning-rate 5e-6 \
    --beta 0.05 \
    --save-steps 20 \
    >"$LOG_DIR/train_reranker_grpo.log" 2>&1

EVAL_COMMAND=(
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" h2_full_learned_route/evaluate_h2_six_variants.py
    --output-dir "$RUN_DIR/evaluation" \
    --model-path "$BASE_MODEL" \
    --adapter-path "$POLICY_ADAPTER" \
    --old-gate-model-path "$OLD_GATE_MODEL" \
    --old-gate-adapter-path "$OLD_GATE_ADAPTER" \
    --gate-model-path "$GATE_SFT_MERGED" \
    --gate-adapter-path "$GATE_GRPO/best_grpo_adapter" \
    --reranker-model-path "$RERANK_SFT_MERGED" \
    --reranker-adapter-path "$RERANK_GRPO/best_grpo_adapter" \
    --defender-learned-model-path "$DEFENDER_MODEL" \
    --seeds "${SEEDS[@]}" \
    --games "$EVAL_GAMES" \
    --parallel-workers "$EVAL_PARALLEL_WORKERS"
)
if [[ "$EVAL_FORCE" == "1" ]]; then
  EVAL_COMMAND+=(--force)
fi
run_stage six_variant_evaluation "${EVAL_COMMAND[@]}" >"$LOG_DIR/evaluation.log" 2>&1

log_stage complete
touch "$RUN_DIR/COMPLETE"

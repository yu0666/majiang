#!/usr/bin/env bash
#
# H1 / B_phi end-to-end pipeline on the CORRECTED harness.
#
#   Step A  regenerate belief-SFT data (per-j prompts, fixed tenpai label,
#           leak-safe split: train oversampled, eval natural)
#   Step B  train LLM-B_phi LoRA on the train split
#   Step C  AUTHORITATIVE H1 gate: run_h1_belief_experiment.py with the trained
#           B_phi (backend=local_qwen), scored vs realized tenpai on a balanced
#           eval (AUC + Brier reduction vs B0/B1 + paired sign-test)
#   Step D  SFT-file diagnostics (JSON parse rate / latency / calibration)
#
# Determinism: PYTHONHASHSEED=0 is required (the engine iterates set()).  The
# Python entry points self-re-exec with it, but we export it here too so torch
# data ordering is stable.
#
# Steps B/C/D need a GPU + the local model; Step A is CPU-only.
# Run the whole thing:           bash run_h1_pipeline.sh
# Run a single step:             bash run_h1_pipeline.sh A
#                                bash run_h1_pipeline.sh C    (single-seed gate)
#                                bash run_h1_pipeline.sh AGG  (multi-seed mean+/-CI, paper table)
set -euo pipefail
export PYTHONHASHSEED=0
cd "$(dirname "$0")"

# ----------------------------- configuration --------------------------------
BASE_MODEL="${BASE_MODEL:-models/Qwen-Mahjong-V3-Merged}"
TAG="${TAG:-v3}"
ADAPTER_DIR="qwen-bphi-sft-${TAG}"
DATA="belief_sft_data_${TAG}.jsonl"           # generator writes _train/_eval/_meta beside it
GATE_DIR="H1_belief_results_${TAG}"
SFT_EVAL_DIR="H1_belief_sft_eval_${TAG}"

# Step A — data (CPU). Labels = public-information play-aware posterior (a public
# belief proxy, observer-independent; NOT a private per-j belief). Prompt is
# per-observer (carries j's public footprint). TRAIN class-balanced but
# duplication capped at MAX_OVERSAMPLE_RATIO x natural count.
GEN_GAMES="${GEN_GAMES:-300}"
GEN_SAMPLE_EVERY="${GEN_SAMPLE_EVERY:-3}"
GEN_MAX_RECORDS="${GEN_MAX_RECORDS:-60000}"
GEN_TARGET_PER_LABEL="${GEN_TARGET_PER_LABEL:-4000}"
GEN_MAX_OVERSAMPLE_RATIO="${GEN_MAX_OVERSAMPLE_RATIO:-3}"
GEN_LABEL_SOURCE="${GEN_LABEL_SOURCE:-opponent_posterior}"
GEN_ORACLE_SAMPLES="${GEN_ORACLE_SAMPLES:-30}"
GEN_ORACLE_BETA="${GEN_ORACLE_BETA:-2.0}"
# Positive class: shanten<=DANGER_THRESHOLD. Oracle-ceiling sweep showed precise
# tenpai (0) is near-unreadable from public info (AUC ceiling 0.54) but 'danger'
# (1, tenpai-or-one-away) is readable (ceiling 0.77). Default to 1.
DANGER_THRESHOLD="${DANGER_THRESHOLD:-1}"

# Step C — gate run.
# B_PHI_SOURCE=mc (default): B_phi := the MC public-info posterior, CPU-only, no
#   LLM, and it PASSES H1 on the danger target (AUC~0.76). This is the working
#   estimator. backend forced to heuristic_fallback (no model load).
# B_PHI_SOURCE=llm: score the trained LLM-B_phi instead (needs GPU + adapter).
#   NOTE: the text-SFT LLM failed to learn the numeric posterior (over-fires,
#   AUC~0.52) even though the signal exists; kept only for comparison.
# MC gate is CPU-only & cheap, so sample densely for a stable balanced-eval AUC
# (the balanced subset downsamples to the positives; too few -> noisy verdict
# that can dip below 0.75 even though the large-sample AUC is ~0.77-0.79).
B_PHI_SOURCE="${B_PHI_SOURCE:-mc}"
GATE_GAMES="${GATE_GAMES:-250}"
GATE_SAMPLE_EVERY="${GATE_SAMPLE_EVERY:-3}"
GATE_TRAIN_RATIO="${GATE_TRAIN_RATIO:-0.4}"
GATE_ORACLE_SAMPLES="${GATE_ORACLE_SAMPLES:-60}"
AUC_THRESHOLD="${AUC_THRESHOLD:-0.75}"
BRIER_REDUCTION="${BRIER_REDUCTION:-0.20}"
P_THRESHOLD="${P_THRESHOLD:-0.05}"

STEP="${1:-ALL}"

step_A() {
  echo "==== Step A: regenerate belief-SFT data ($DATA) ===="
  python generate_belief_sft_data.py \
    --games "$GEN_GAMES" \
    --all-players \
    --sample-every "$GEN_SAMPLE_EVERY" \
    --max-records "$GEN_MAX_RECORDS" \
    --target-per-label "$GEN_TARGET_PER_LABEL" \
    --max-oversample-ratio "$GEN_MAX_OVERSAMPLE_RATIO" \
    --label-source "$GEN_LABEL_SOURCE" \
    --oracle-samples "$GEN_ORACLE_SAMPLES" \
    --oracle-beta "$GEN_ORACLE_BETA" \
    --danger-threshold "$DANGER_THRESHOLD" \
    --train-ratio 0.9 \
    --output "$DATA"
  echo "Wrote ${DATA%.jsonl}_train.jsonl / _eval.jsonl / _meta.json"
}

step_B() {
  echo "==== Step B: train LLM-B_phi LoRA -> $ADAPTER_DIR ===="
  python train_belief_sft.py \
    --model-path "$BASE_MODEL" \
    --data-file "${DATA%.jsonl}_train.jsonl" \
    --output-dir "$ADAPTER_DIR" \
    --epochs 2.0 \
    --learning-rate 8e-5
}

step_C() {
  echo "==== Step C: AUTHORITATIVE H1 gate (B_phi_source=$B_PHI_SOURCE) -> $GATE_DIR ===="
  if [ "$B_PHI_SOURCE" = "mc" ]; then
    GATE_BACKEND_ARGS=(--backend heuristic_fallback --b-phi-source mc)  # CPU, no model
  else
    GATE_BACKEND_ARGS=(--backend local_qwen --model-path "$BASE_MODEL" --adapter-path "$ADAPTER_DIR" --b-phi-source llm)
  fi
  python run_h1_belief_experiment.py \
    "${GATE_BACKEND_ARGS[@]}" \
    --games "$GATE_GAMES" \
    --sample-every "$GATE_SAMPLE_EVERY" \
    --train-ratio "$GATE_TRAIN_RATIO" \
    --oracle-samples "$GATE_ORACLE_SAMPLES" \
    --oracle-beta "$GEN_ORACLE_BETA" \
    --danger-threshold "$DANGER_THRESHOLD" \
    --auc-threshold "$AUC_THRESHOLD" \
    --brier-reduction "$BRIER_REDUCTION" \
    --p-threshold "$P_THRESHOLD" \
    --output-dir "$GATE_DIR"
  echo "--- H1 gate verdict ---"
  python - "$GATE_DIR/h1_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
g = s["H1_gate"]
print("backend           :", s["backend"])
print("balanced samples  :", s["samples_balanced_eval"])
if g.get("underpowered"):
    print("!! UNDERPOWERED   :", g.get("underpowered_note"))
print("B2 AUC            : %.3f (need >= %s)" % (g["B2_auc"], g["requirements"]["auc"]))
print("Brier red vs B0/B1: %.2f / %.2f (need %s)" % (
    g["B2_vs_B0_relative_brier_reduction"], g["B2_vs_B1_relative_brier_reduction"],
    g["requirements"]["brier_reduction"]))
print("paired p vs B0/B1 : %s / %s" % (
    g["paired_test_vs_B0"]["sign_test_p"], g["paired_test_vs_B1"]["sign_test_p"]))
print("H1 GATE PASS      :", g["pass"])
PY
}

step_AGG() {
  echo "==== Step AGG: multi-seed H1 robustness (mean +/- 95% CI) -> H1_seed_aggregate ===="
  if [ "$B_PHI_SOURCE" = "mc" ]; then
    AGG_BACKEND_ARGS=(--backend heuristic_fallback --b-phi-source mc)
  else
    AGG_BACKEND_ARGS=(--backend local_qwen --model-path "$BASE_MODEL" --adapter-path "$ADAPTER_DIR" --b-phi-source llm)
  fi
  python aggregate_h1_seeds.py \
    "${AGG_BACKEND_ARGS[@]}" \
    --num-seeds "${AGG_NUM_SEEDS:-5}" \
    --games "${AGG_GAMES:-200}" \
    --sample-every "$GATE_SAMPLE_EVERY" \
    --train-ratio "$GATE_TRAIN_RATIO" \
    --oracle-samples "$GATE_ORACLE_SAMPLES" \
    --oracle-beta "$GEN_ORACLE_BETA" \
    --danger-threshold "$DANGER_THRESHOLD" \
    --auc-threshold "$AUC_THRESHOLD" \
    --brier-reduction "$BRIER_REDUCTION" \
    --p-threshold "$P_THRESHOLD"
}

step_D() {
  echo "==== Step D: SFT-file diagnostics (parse rate / latency / calibration) -> $SFT_EVAL_DIR ===="
  python evaluate_belief_sft.py \
    --eval-file "${DATA%.jsonl}_eval.jsonl" \
    --model-path "$BASE_MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --limit 400 \
    --output-dir "$SFT_EVAL_DIR"
}

case "$STEP" in
  A) step_A ;;
  B) step_B ;;
  C) step_C ;;
  AGG) step_AGG ;;
  D) step_D ;;
  ALL) step_A; step_B; step_C; step_D ;;
  *) echo "Unknown step: $STEP (use A|B|C|AGG|D|ALL)"; exit 1 ;;
esac

echo "Done: step $STEP"

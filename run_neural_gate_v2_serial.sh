#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONHASHSEED=0

OUT_DIR="${OUT_DIR:-Neural_Gate_20260723_v2}"
PYTHON="${PYTHON:-./py10/bin/python3}"

echo "[1/2] Collect continuous_v2 teacher data -> ${OUT_DIR}/teacher"
"${PYTHON}" collect_neural_gate_teacher.py \
  --output-dir "${OUT_DIR}/teacher" \
  --teacher-gate-policy continuous_v2 \
  --backend heuristic_fallback \
  --opponent-style neural \
  --neural-opponent-model-path Neural_opponent_model/neural_opponent_policy.pth \
  --neural-opponent-device cpu \
  --target-exploit 800 \
  --target-safe 400 \
  --target-deceive 400 \
  --target-states 1600 \
  --max-games 3000 \
  --sample-stride 1 \
  --seed 2026072301

echo "[2/2] Train neural gate -> ${OUT_DIR}/model"
"${PYTHON}" train_neural_gate.py \
  --input "${OUT_DIR}/teacher/teacher_gate_states.jsonl" \
  --output-dir "${OUT_DIR}/model" \
  --epochs 60 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --min-examples 500 \
  --min-reward-range 0 \
  --reward-pg-weight 0.0 \
  --normalization-mode raw \
  --device cpu \
  --seed 3422

echo "Done."
echo "Model: ${OUT_DIR}/model/neural_gate_policy.pth"

#!/bin/bash
# PPO training launch script for MASK parameter optimization
# Usage: ./run_ppo_experiment.sh [options]

set -e

# Default configuration
ITERATIONS=500
GAMES_PER_ITER=64
LR=3e-4
SEED=42
SAVE_DIR="ppo_checkpoints"
DEVICE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations) ITERATIONS="$2"; shift 2;;
        --games-per-iter) GAMES_PER_ITER="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --save-dir) SAVE_DIR="$2"; shift 2;;
        --gpu) DEVICE="cuda:$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

echo "============================================"
echo "PPO Training for MASK Parameters"
echo "============================================"
echo "Iterations: $ITERATIONS"
echo "Games per iteration: $GAMES_PER_ITER"
echo "Learning rate: $LR"
echo "Seed: $SEED"
echo "Save directory: $SAVE_DIR"
echo "Device: $DEVICE"
echo "============================================"
echo ""

# Run training
python train_ppo_mask.py \
    --iterations "$ITERATIONS" \
    --games-per-iter "$GAMES_PER_ITER" \
    --lr "$LR" \
    --seed "$SEED" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE"

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $SAVE_DIR"
echo "Training log: $SAVE_DIR/training_log.jsonl"

# H2 Full Learned-Defender Route

This route retrains the learned MASK gate and the candidate reranker, then
evaluates six variants against the learned defender:

- `L0`: base policy, no opponent modeling.
- `L1`: base policy with reactive `z`.
- `L2_rule_gate`: MASK with the rule three-state gate.
- `L2_learned_gate`: MASK with the previous learned gate.
- `L2_retrained_gate`: MASK with the newly retrained learned gate.
- `L2_retrained_gate_reranker`: newly retrained learned gate plus dedicated reranker.

Default evaluation:

```bash
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
tmux new -s h2_full_learned
GPU=1 RUN_ID=20260716_h2_full_learned h2_full_learned_route/run_h2_full_learned_pipeline.sh
```

The final summary is written to:

```text
H2_Full_Learned_<RUN_ID>/evaluation/h2_six_variant_learned_defender_summary.json
```

Useful controls:

- `EVAL_GAMES=500`: games per seed per variant.
- `SEEDS="2026071601 2026072601 2026073601"`: evaluation seeds.
- `GATE_TARGET_STATES=500`: learned-gate rollout states.
- `RERANK_TARGET_EXPLOIT=260 RERANK_TARGET_SAFE=200 RERANK_TARGET_DECEIVE=200`: reranker rollout quotas.
- `DRY_RUN=1`: print the resolved commands without running them.
- `FORCE_STAGE=1`: rerun stages even if marker files exist.

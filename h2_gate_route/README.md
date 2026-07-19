# H2 Gate Route

This directory is an isolated route for the H2 pre-experiment:

- L0: base policy, no opponent modeling.
- L1: reactive z, passive opponent adaptation.
- L2-rule: MASK with rule-based exploit/safe/deceive gate.
- L2-learned: MASK with a separately trained three-state gate.

The scripts here do not modify existing project files. They only call the
existing data collection, SFT, GRPO, merge, and Gate1 evaluation entry points.

## Opponent for H2

Use the belief-responsive opponent as the main H2 opponent:

```text
opponent-style=responsive
defender-threat-model=blend
defender-tell-weight=0.3
defender-tell-window=6
```

This opponent is the correct one for Gate1 because it updates its defense from
public signals. Basic non-belief and robust anti-shaping opponents should be
used later as sanity and robustness checks, not mixed into the H2 main result.

## Training Flow

1. Collect paired mode rollouts for `exploit`, `safe`, and `deceive`.
2. Convert those rollouts into gate SFT and gate GRPO data.
3. Train Gate-SFT to imitate the best/rule-compatible mode.
4. Merge the Gate-SFT LoRA.
5. Train Gate-GRPO with actual settlement, fan reward, deal-in penalty, and tail risk.
6. Evaluate L0/L1/L2-rule/L2-learned on the same seeds and opponent.

The gate only selects mode. The policy/reranker is unchanged in this route so
the comparison isolates whether a learned three-state gate improves L2.

## Smoke Run

```bash
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
DRY_RUN=1 bash h2_gate_route/run_h2_gate_route.sh
```

## Full Run

```bash
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
GPU=1 RUN_ID=20260715_h2_gate_v1 bash h2_gate_route/run_h2_gate_route.sh
```

Important environment variables:

- `RUN_ID`: output suffix.
- `RUN_DIR`: full output directory. Defaults to `H2_Gate_Route_${RUN_ID}`.
- `GPU`: CUDA device id.
- `POLICY_MODEL`: policy merged model used for L0/L1/L2 decisions.
- `POLICY_ADAPTER`: policy adapter used during L0/L1/L2 decisions.
- `GATE_BASE_MODEL`: merged model used to train the standalone gate.
- `EVAL_GAMES`: games per seed per variant. Default `500`.
- `TARGET_STATES`: gate oracle states. Default `500`.
- `GATE_GRPO_STEPS`: GRPO steps. Default `150`.

Final evaluation summary:

```text
${RUN_DIR}/evaluation/h2_gate_ladder_summary.json
```

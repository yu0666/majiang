# H2 Clean V1 Route

This route builds a clean shared backbone for H2.

Clean means the V1 SFT data contains only:

- RuleBot / expert-style action data.
- Self-play action data.

It intentionally excludes MASK, B_phi, z_j(t), gate labels, and deceive teacher
data. This keeps the H2 ladder interpretation clean:

```text
L0 = clean base policy
L1 = clean base + reactive z
L2 = clean base + B_phi/MASK/gate
```

## Full Run

```bash
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
GPU=1 RUN_ID=20260715_v1_clean bash h2_clean_route/run_v1_clean_route.sh
```

Outputs:

```text
V1_Clean_${RUN_ID}/v1_clean/v1_clean_sft.jsonl
qwen-v1-clean-sft-${RUN_ID}
models/Qwen-Mahjong-V1-Clean-SFT-${RUN_ID}-Merged
V1_Clean_${RUN_ID}/eval_v1_clean/v2_e2_ladder_3seeds_summary.json
```


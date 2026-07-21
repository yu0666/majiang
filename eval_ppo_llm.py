"""Evaluate PPO-trained MASK parameters with LLM base model.

Uses the full MASK framework (MASKLLMAgent) with PPO-learned parameters,
backed by Qwen-Mahjong LLM for decision-making.

Usage:
  python eval_ppo_llm.py \
    --policy-path ppo_run_seed42/best_policy.pt \
    --belief-surrogate belief_surrogate.pt \
    --model-path models/Qwen-Mahjong-V1-Mixed-SFT-Merged \
    --seeds 50260627 50261627 50262627 50263627 50264627 \
           50265627 50266627 50267627 50268627 50269627 \
    --games 100 \
    --gpu 0 \
    --output ppo_llm_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from collect_belief_data import extract_opponent_public_features
from game import MahjongGame, bot_decide_exchange, bot_decide_missing_suit, bot_decide_response, bot_decide_turn_action, parse_console_tile
from mask_llm import MASKLLMAgent, PublicOpponentTracker, RiskGate, _clip
from ppo_agent import MASKPolicyNet, PARAM_NAMES, unscale_params
from ppo_features import extract_state_features
from prompt_builder import get_legal_actions
from train_belief_surrogate import BeliefSurrogate


def init_game_eval(seed: int) -> MahjongGame:
    random.seed(seed)
    game = MahjongGame(
        game_id=f"eval_{seed}",
        player_names=["P0", "P1", "P2", "P3"],
        bots=[False, True, True, True],
    )
    game.start_game()
    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))
    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))
    return game


def get_ppo_params_at_step(
    policy: MASKPolicyNet,
    game: MahjongGame,
    z_tracker: PublicOpponentTracker,
    risk_gate: RiskGate,
    belief_surrogate: Optional[BeliefSurrogate],
    device: str,
) -> Dict[str, float]:
    """Run PPO policy network to get current parameters."""
    z_state = z_tracker.summary()
    beliefs = {}
    if belief_surrogate is not None:
        belief_surrogate.eval()
        with torch.no_grad():
            for target_pid in (1, 2, 3):
                feats = extract_opponent_public_features(game, target_pid, 0)
                feat_tensor = torch.tensor(feats, dtype=torch.float32, device=device)
                prob = belief_surrogate(feat_tensor).item()
                beliefs[f"P{target_pid}"] = {"tenpai_prob": prob, "tenpai_confidence": prob}
    else:
        from belief_oracle import opponent_view_posterior
        for target_pid in (1, 2, 3):
            beliefs[f"P{target_pid}"] = opponent_view_posterior(game, target_pid, 0, num_samples=5)

    gate = risk_gate.compute(game, 0, z_state, beliefs)
    state_features = extract_state_features(game, 0, z_state, beliefs, gate)
    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)

    with torch.no_grad():
        params_scaled, _, _ = policy.get_action(state_tensor, deterministic=True)
        params_actual = unscale_params(params_scaled)

    # Convert tensor to dict
    params_dict = {}
    for i, name in enumerate(PARAM_NAMES):
        params_dict[name] = params_actual[i].item()
    return params_dict


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None):
    """Execute a player action (hu/gang/discard). Returns discarded tile or None."""
    player = game.players[pid]
    action = (action or "").strip()
    discarded_tile = None

    if action == "h" and player.can_hu():
        win_tile = drawn_tile if drawn_tile else (player.hand_tiles[-1] if player.hand_tiles else None)
        if win_tile is not None:
            game.hu(pid, win_tile, True)
            game.check_game_over()
        return None

    if action == "g":
        gang_info = game.can_self_gang(pid)
        if gang_info.get("can_gang"):
            game.gang(pid, gang_info["gang_tiles"][0])
            return None

    if action.startswith("d "):
        tile = parse_console_tile(action[2:])
        if tile and game.discard_tile(pid, tile):
            discarded_tile = tile

    if discarded_tile is None:
        legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
        if legal_discards:
            tile = parse_console_tile(legal_discards[0][2:])
            if tile and game.discard_tile(pid, tile):
                discarded_tile = tile

    return discarded_tile


def resolve_responses_eval(
    game: MahjongGame, discarded_tile, acting_pid: int, agent: MASKLLMAgent,
) -> Dict[str, Any]:
    """Resolve responses from other players after a discard."""
    responses = game.check_responses(discarded_tile, acting_pid)
    responded = False
    agent_won_by_discard = False

    for rid, acts in responses.items():
        if responded:
            break

        player = game.players[rid]
        if rid == 0:
            # P0 (LLM agent) responds
            valid = get_legal_actions(game, rid, response_actions=acts)
            response_action = "h" if "h" in valid else agent.decide(game, valid)
        else:
            response_action = bot_decide_response(player, acts)

        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            responded = True
            agent_won_by_discard = rid == 0
        elif response_action == "g" and "gang" in acts:
            game.gang(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True
        elif response_action == "p" and "peng" in acts:
            game.peng(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True

    return {
        "responded": responded,
        "agent_won_by_discard": agent_won_by_discard,
    }


def play_one_game_eval(
    policy: MASKPolicyNet,
    agent: MASKLLMAgent,
    seed: int,
    belief_surrogate: Optional[BeliefSurrogate],
    device: str,
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Play one game with PPO params + LLM base."""
    game = init_game_eval(seed)
    start_balance = game.players[0].balance

    z_tracker = PublicOpponentTracker([1, 2, 3])
    risk_gate = RiskGate()

    steps = 0
    skip_draw = True

    while not game.is_game_over and steps < max_steps:
        steps += 1
        if game.deck.remaining_count() == 0 or sum(1 for p in game.players if p.is_hu) >= 3:
            game.check_game_over()
            break

        pid = game.current_player_id
        player = game.players[pid]
        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        drawn_tile = None
        if not skip_draw:
            drawn_tile = game.draw_tile(pid)
            if not drawn_tile:
                game.check_game_over()
                break
        else:
            skip_draw = False

        if pid == 0:
            # Update tracker with latest game history
            z_tracker.update_from_game(game)
            # Get PPO params for this step
            ppo_params = get_ppo_params_at_step(
                policy, game, z_tracker, risk_gate, belief_surrogate, device
            )
            agent.ppo_params = ppo_params

            legal_actions = get_legal_actions(game, 0)
            action = agent.decide(game, legal_actions)
        else:
            action = bot_decide_turn_action(player, game)

        discarded_tile = execute_action(game, pid, action, drawn_tile)
        if game.is_game_over:
            break

        if discarded_tile is None:
            # hu or gang happened
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        # Resolve responses
        response_info = resolve_responses_eval(game, discarded_tile, pid, agent)

        if game.is_game_over:
            break

        if response_info["responded"]:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    end_balance = game.players[0].balance
    net_score = end_balance - start_balance
    is_hu = game.players[0].is_hu

    return {
        "seed": seed,
        "net_score": net_score,
        "is_hu": is_hu,
        "steps": steps,
        "final_balance": end_balance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--belief-surrogate", type=str, default=None)
    parser.add_argument("--model-path", required=True, help="Path to Qwen-Mahjong model")
    parser.add_argument("--seeds", nargs="+", type=int, default=[50260627, 50261627, 50262627, 50263627, 50264627,
                                                                  50265627, 50266627, 50267627, 50268627, 50269627])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Override: cpu or cuda:X")
    parser.add_argument("--output", type=str, default="ppo_llm_eval.json")
    args = parser.parse_args()

    device = args.device if args.device else f"cuda:{args.gpu}"
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    # Load PPO policy
    print(f"Loading PPO policy from {args.policy_path}")
    ckpt = torch.load(args.policy_path, map_location=device, weights_only=False)
    policy = MASKPolicyNet().to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Load belief surrogate
    belief_surrogate = None
    if args.belief_surrogate and os.path.exists(args.belief_surrogate):
        bckpt = torch.load(args.belief_surrogate, map_location=device, weights_only=False)
        belief_surrogate = BeliefSurrogate(input_dim=bckpt.get("input_dim", 18)).to(device)
        belief_surrogate.eval()
        print(f"Loaded belief surrogate")

    # Setup LLM
    print(f"Loading LLM from {args.model_path}")
    from llm_backends import build_llm_callable
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend="local_qwen",
        repo_dir=repo_dir,
        model_path=args.model_path,
        max_new_tokens=64,
        temperature=0.0,
    )

    # Create MASKLLMAgent with continuous_v7 gate policy
    agent = MASKLLMAgent(
        player_id=0,
        llm=llm,
        gate_policy="continuous_v7",
        use_mc_belief=True,
        mc_oracle_samples=5,  # Fast for evaluation
        deceive_style="threat",
        ppo_params={},  # Will be updated dynamically per step
    )

    # Run evaluation
    all_results = []
    t0 = time.time()
    for seed in args.seeds:
        seed_results = []
        for g in range(args.games):
            game_seed = seed * 10000 + g
            result = play_one_game_eval(
                policy, agent, game_seed,
                belief_surrogate=belief_surrogate,
                device=device,
            )
            seed_results.append(result)
            if (g + 1) % 10 == 0:
                avg = statistics.mean(r["net_score"] for r in seed_results)
                print(f"  Seed {seed}: {g+1}/{args.games} games, avg_net={avg:+.1f}")

        avg_net = statistics.mean(r["net_score"] for r in seed_results)
        hu_rate = sum(1 for r in seed_results if r["is_hu"]) / len(seed_results)
        print(f"  Seed {seed} DONE: avg_net={avg_net:+.1f}, hu_rate={hu_rate:.2%}")
        all_results.extend(seed_results)

    elapsed = time.time() - t0

    # Aggregate
    avg_net = statistics.mean(r["net_score"] for r in all_results)
    hu_rate = sum(1 for r in all_results if r["is_hu"]) / len(all_results)
    sd = statistics.stdev(r["net_score"] for r in all_results) if len(all_results) > 1 else 0

    output = {
        "config": {
            "policy_path": args.policy_path,
            "model_path": args.model_path,
            "belief_surrogate": args.belief_surrogate,
            "seeds": args.seeds,
            "games_per_seed": args.games,
            "total_games": len(all_results),
            "gate_policy": "continuous_v7",
            "elapsed_seconds": elapsed,
        },
        "aggregate": {
            "avg_net": avg_net,
            "sd_net": sd,
            "hu_rate": hu_rate,
            "n": len(all_results),
        },
        "per_seed": {
            str(seed): {
                "avg_net": statistics.mean(r["net_score"] for r in all_results if r["seed"] // 10000 == seed),
                "n": sum(1 for r in all_results if r["seed"] // 10000 == seed),
            }
            for seed in args.seeds
        },
    }

    print(f"\n=== Results ===")
    print(f"Total games: {len(all_results)}")
    print(f"Avg net score: {avg_net:+.2f} (sd={sd:.2f})")
    print(f"Hu rate: {hu_rate:.2%}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(all_results):.2f}s/game)")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

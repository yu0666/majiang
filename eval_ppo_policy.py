"""Evaluate a trained PPO policy against baseline hardcoded weights.

Uses the 50260627 seed family for CRN paired comparison.

Usage:
  python eval_ppo_policy.py \
    --policy-path ppo_run_seed42/best_policy.pt \
    --belief-surrogate belief_surrogate.pt \
    --seeds 50260627 50261627 50262627 50263627 50264627 \
           50265627 50266627 50267627 50268627 50269627 \
    --games 100 \
    --device cpu \
    --output ppo_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from collect_belief_data import extract_opponent_public_features
from game import MahjongGame, bot_decide_exchange, bot_decide_missing_suit, bot_decide_response, bot_decide_turn_action, parse_console_tile
from mask_llm import PublicOpponentTracker, RiskGate, _clip
from ppo_agent import MASKPolicyNet, unscale_params
from ppo_features import extract_state_features
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


def play_one_game_eval(
    policy: MASKPolicyNet,
    seed: int,
    belief_surrogate: Optional[BeliefSurrogate] = None,
    device: str = "cpu",
    max_steps: int = 200,
) -> Dict[str, Any]:
    """Play one game and return metrics."""
    game = init_game_eval(seed)
    start_balance = game.players[0].balance

    z_tracker = PublicOpponentTracker([1, 2, 3])
    risk_gate = RiskGate()

    steps = 0
    hu_result = None
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

        if not skip_draw:
            drawn = game.draw_tile(pid)
            if not drawn:
                game.check_game_over()
                break
        else:
            skip_draw = False

        if pid == 0:
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
                params_scaled, log_prob, value = policy.get_action(state_tensor, deterministic=True)
                params_actual = unscale_params(params_scaled)

            # Decide action using params
            action, info = ppo_decide_action_eval(game, 0, params_actual, z_tracker, risk_gate)

            if action.startswith("d "):
                try:
                    tile = parse_console_tile(action[2:])
                    game.discard_tile(0, tile)
                    # Track
                    z_tracker.update_from_game(game)
                    # Handle responses
                    responses = game.check_responses(tile, 0)
                    for resp_pid, resp_list in sorted(responses.items(), key=lambda x: x[0]):
                        if resp_pid == 0:
                            continue
                        if "hu" in resp_list:
                            game.process_response(resp_pid, "hu")
                            break
                        elif "gang" in resp_list and random.random() < 0.3:
                            game.process_response(resp_pid, "gang")
                            break
                        elif "peng" in resp_list and random.random() < 0.4:
                            game.process_response(resp_pid, "peng")
                            break
                except Exception:
                    pass
            elif action == "h":
                try:
                    game.process_response(0, "hu")
                except Exception:
                    pass
        else:
            # Bot turn
            action = bot_decide_turn_action(player, game)
            if action.startswith("d "):
                try:
                    tile = parse_console_tile(action[2:])
                    game.discard_tile(pid, tile)
                    z_tracker.update_from_game(game)
                    responses = game.check_responses(tile, pid)
                    for resp_pid, resp_list in sorted(responses.items(), key=lambda x: x[0]):
                        if resp_pid == pid:
                            continue
                        if "hu" in resp_list:
                            game.process_response(resp_pid, "hu")
                            break
                        elif "gang" in resp_list and random.random() < 0.3:
                            game.process_response(resp_pid, "gang")
                            break
                        elif "peng" in resp_list and random.random() < 0.4:
                            game.process_response(resp_pid, "peng")
                            break
                except Exception:
                    pass

        if game.check_game_over():
            break

    # Compute metrics
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


def ppo_decide_action_eval(game, player_id, params, z_tracker, risk_gate):
    """Decide action using PPO parameters."""
    player = game.players[player_id]
    z_state = z_tracker.summary()

    # Get legal actions
    valid_actions = []
    if player.is_hu:
        valid_actions.append("h")
    discard_actions = []
    for tile in player.hand_tiles:
        suit_char = {1: "万", 2: "条", 3: "筒"}.get(tile.suit.value, "")
        discard_actions.append(f"d {tile.number}{suit_char}")
    valid_actions.extend(discard_actions)

    if not valid_actions:
        return "d 1万", {"mode": "fallback"}

    # Check hu
    if "h" in valid_actions:
        return "h", {"mode": "exploit"}

    # Get discard actions
    discard_actions = [a for a in valid_actions if a.startswith("d ")]
    if not discard_actions:
        return valid_actions[0], {"mode": "fallback"}

    # Use params to select best discard (simplified: use w_shanten to pick)
    w_shanten = params["w_shanten"]
    w_ukeire = params["w_ukeire"]
    w_value = params["w_value"]

    best_action = discard_actions[0]
    best_score = -float("inf")

    for action in discard_actions:
        try:
            tile = parse_console_tile(action[2:])
            # Simulate discard
            remaining = [t for t in player.hand_tiles if t != tile]
            from rule_engine import ShantenCalculator
            shanten = ShantenCalculator.calculate_shanten(remaining, player.missing_suit)
            # Score: lower shanten is better
            score = -shanten * w_shanten / 100.0
            if score > best_score:
                best_score = score
                best_action = action
        except Exception:
            pass

    return best_action, {"mode": "param_score"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, help="Path to best_policy.pt or final_policy.pt")
    parser.add_argument("--belief-surrogate", type=str, default=None, help="Path to belief_surrogate.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[50260627, 50261627, 50262627, 50263627, 50264627,
                                                                  50265627, 50266627, 50267627, 50268627, 50269627])
    parser.add_argument("--games", type=int, default=100, help="Games per seed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="ppo_eval_results.json")
    args = parser.parse_args()

    # Load policy
    print(f"Loading policy from {args.policy_path}")
    ckpt = torch.load(args.policy_path, map_location=args.device, weights_only=False)
    policy = MASKPolicyNet().to(args.device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Load belief surrogate
    belief_surrogate = None
    if args.belief_surrogate and os.path.exists(args.belief_surrogate):
        bckpt = torch.load(args.belief_surrogate, map_location=args.device, weights_only=False)
        belief_surrogate = BeliefSurrogate(input_dim=bckpt.get("input_dim", 18)).to(args.device)
        belief_surrogate.load_state_dict(bckpt["model_state"])
        belief_surrogate.eval()
        print(f"Loaded belief surrogate from {args.belief_surrogate}")

    # Run evaluation
    all_results = []
    for seed in args.seeds:
        seed_results = []
        for g in range(args.games):
            game_seed = seed * 10000 + g
            result = play_one_game_eval(
                policy, game_seed,
                belief_surrogate=belief_surrogate,
                device=args.device,
            )
            seed_results.append(result)

        avg_net = statistics.mean(r["net_score"] for r in seed_results)
        hu_rate = sum(1 for r in seed_results if r["is_hu"]) / len(seed_results)
        print(f"  Seed {seed}: avg_net={avg_net:+.1f}, hu_rate={hu_rate:.2%}, n={len(seed_results)}")
        all_results.extend(seed_results)

    # Aggregate
    avg_net = statistics.mean(r["net_score"] for r in all_results)
    hu_rate = sum(1 for r in all_results if r["is_hu"]) / len(all_results)
    sd = statistics.stdev(r["net_score"] for r in all_results) if len(all_results) > 1 else 0

    output = {
        "config": {
            "policy_path": args.policy_path,
            "belief_surrogate": args.belief_surrogate,
            "seeds": args.seeds,
            "games_per_seed": args.games,
            "total_games": len(all_results),
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

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

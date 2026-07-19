"""Evaluate four neural opponent policies in self-play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from experiment_trace import ensure_deterministic_hashing, write_json
from game import (
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
)
from neural_opponent_policy import load_policy
from run_gate1_experiments import execute_action


def init_selfplay_game(seed: int, game_id: str) -> MahjongGame:
    import random

    random.seed(seed)
    game = MahjongGame(game_id, ["N0", "N1", "N2", "N3"], bots=[False, False, False, False])
    game.start_game()
    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))
    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))
    return game


def play_one_game(args: argparse.Namespace, game_index: int) -> Dict[str, Any]:
    seed = args.seed + game_index * 1009
    game = init_selfplay_game(seed, f"NeuralSelfPlay_{game_index}")
    policies = {
        pid: load_policy(args.model_path, observer_pid=pid, device=args.device)
        for pid in range(4)
    }
    start_balances = [player.balance for player in game.players]
    hu_counts = [0, 0, 0, 0]
    self_draw_counts = [0, 0, 0, 0]
    discard_win_counts = [0, 0, 0, 0]
    dealin_counts = [0, 0, 0, 0]

    skip_draw = True
    steps = 0
    while not game.is_game_over and steps < args.max_steps:
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

        was_hu = player.is_hu
        action = policies[pid].turn(player, game, step=steps)
        discarded_tile = execute_action(game, pid, action, drawn_tile)
        if action == "h" and not was_hu and player.is_hu:
            hu_counts[pid] += 1
            self_draw_counts[pid] += 1

        if game.is_game_over:
            break

        if discarded_tile is None:
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        responses = game.check_responses(discarded_tile, pid)
        responded = False
        for rid, acts in responses.items():
            if responded:
                break
            responder = game.players[rid]
            if rid in policies:
                response_action = policies[rid].response(responder, acts, game)
            else:
                response_action = bot_decide_response(responder, acts)

            if response_action == "h" and "hu" in acts:
                game.hu(rid, discarded_tile, False, pid)
                game.check_game_over()
                hu_counts[rid] += 1
                discard_win_counts[rid] += 1
                dealin_counts[pid] += 1
                responded = True
            elif response_action == "g" and "gang" in acts:
                game.gang(rid, discarded_tile, pid)
                game.current_player_id = rid
                responded = True
            elif response_action == "p" and "peng" in acts:
                game.peng(rid, discarded_tile, pid)
                game.current_player_id = rid
                responded = True

        if game.is_game_over:
            break
        if responded:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    final_balances = [player.balance for player in game.players]
    nets = [final_balances[i] - start_balances[i] for i in range(4)]
    return {
        "game_index": game_index,
        "seed": seed,
        "steps": steps,
        "wall_exhausted": game.deck.remaining_count() == 0,
        "nets": nets,
        "final_balances": final_balances,
        "hu": [bool(player.is_hu) for player in game.players],
        "hu_counts": hu_counts,
        "self_draw_counts": self_draw_counts,
        "discard_win_counts": discard_win_counts,
        "dealin_counts": dealin_counts,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    games = len(rows)
    by_player = []
    for pid in range(4):
        nets = [row["nets"][pid] for row in rows]
        by_player.append(
            {
                "pid": pid,
                "avg_net": sum(nets) / games if games else 0.0,
                "hu_rate": sum(1 for row in rows if row["hu"][pid]) / games if games else 0.0,
                "self_draws": sum(row["self_draw_counts"][pid] for row in rows),
                "discard_wins": sum(row["discard_win_counts"][pid] for row in rows),
                "dealin_rate": sum(row["dealin_counts"][pid] for row in rows) / games if games else 0.0,
            }
        )
    all_nets = [value for row in rows for value in row["nets"]]
    total_hu = sum(sum(1 for value in row["hu"] if value) for row in rows)
    total_dealin = sum(sum(row["dealin_counts"]) for row in rows)
    return {
        "games": games,
        "avg_steps": sum(row["steps"] for row in rows) / games if games else 0.0,
        "wall_exhaustion_rate": sum(1 for row in rows if row["wall_exhausted"]) / games if games else 0.0,
        "overall": {
            "avg_net_per_seat": sum(all_nets) / len(all_nets) if all_nets else 0.0,
            "hu_rate_per_seat": total_hu / (games * 4) if games else 0.0,
            "dealin_rate_per_seat": total_dealin / (games * 4) if games else 0.0,
            "avg_hu_players_per_game": total_hu / games if games else 0.0,
        },
        "by_player": by_player,
    }


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("Neural_opponent_model/neural_opponent_policy.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("Neural_opponent_selfplay_100games"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026071601)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = []
    for game_index in range(args.games):
        row = play_one_game(args, game_index)
        rows.append(row)
        if (game_index + 1) % 10 == 0 or game_index + 1 == args.games:
            print(f"[selfplay] game={game_index + 1}/{args.games}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "model_path": str(args.model_path),
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "device": args.device,
        },
        "summary": summarize(rows),
    }
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "games.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

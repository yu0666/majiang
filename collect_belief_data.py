"""Collect (public_features → tenpai_prob) data using MC oracle.

Runs many games, at each step for each opponent extracts public features
and labels them with opponent_view_posterior MC sampling.
Output: belief_surrogate_data.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from belief_oracle import opponent_view_posterior
from game import MahjongGame, bot_decide_exchange, bot_decide_missing_suit, bot_decide_turn_action
from rule_engine import ShantenCalculator
from tiles import Suit, Tile


# ── Public feature extraction for one opponent ──────────────────────────

FEATURE_DIM = 18


def extract_opponent_public_features(game: MahjongGame, target_pid: int, observer_pid: int = 0) -> List[float]:
    """Extract public-information features about target_pid, visible to observer_pid.

    18-dim vector:
      [0]  n_open_melds / 4
      [1]  n_discards / 30
      [2]  n_concealed / 13
      [3]  tiles_remaining / 80
      [4-7] missing_suit one-hot (wan, tong, tiao, none)
      [8]  avg_discard_number / 9
      [9]  terminal_discard_ratio
      [10] suit_entropy of discards
      [11] n_peng
      [12] n_gang
      [13] latest_discard_suit_distance (how recent discards cluster)
      [14] has_discarded_missing_suit_tile (bool)
      [15] discard_speed_ratio (discards vs avg)
      [16] open_meld_tile_avg_number / 9
      [17] hand_size_ratio (concealed / 13)
    """
    target = game.players[target_pid]
    feats = []

    # [0] open melds
    feats.append(len(target.open_melds) / 4.0)

    # [1] discards
    feats.append(len(target.discarded_tiles) / 30.0)

    # [2] concealed
    n_hand = len(target.hand_tiles)
    n_conceal = n_hand - 1 if n_hand % 3 == 2 else n_hand  # drop drawn tile
    feats.append(max(0, n_conceal) / 13.0)

    # [3] tiles remaining
    feats.append(game.deck.remaining_count() / 80.0)

    # [4-7] missing suit one-hot
    ms = target.missing_suit
    feats.append(1.0 if ms == Suit.WAN else 0.0)
    feats.append(1.0 if ms == Suit.TONG else 0.0)
    feats.append(1.0 if ms == Suit.TIAO else 0.0)
    feats.append(1.0 if ms is None else 0.0)

    # [8-10] discard statistics
    discards = target.discarded_tiles
    if discards:
        avg_num = sum(t.number for t in discards) / len(discards)
        feats.append(avg_num / 9.0)
        terminal_ratio = sum(1 for t in discards if t.number in (1, 9)) / len(discards)
        feats.append(terminal_ratio)
        # suit entropy
        suit_counts = Counter(t.suit for t in discards)
        total_d = sum(suit_counts.values())
        entropy = -sum((c / total_d) * np.log2(c / total_d + 1e-10) for c in suit_counts.values())
        feats.append(entropy / 2.0)  # normalize (max ~1.58 for 3 suits)
    else:
        feats.extend([0.0, 0.0, 0.0])

    # [11-12] peng / gang counts
    n_peng = sum(1 for meld in target.open_melds if len(meld) == 3)
    n_gang = sum(1 for meld in target.open_melds if len(meld) == 4)
    feats.append(n_peng / 4.0)
    feats.append(n_gang / 4.0)

    # [13] discard recency: std of last 5 discard numbers (lower = more focused)
    recent = discards[-5:] if len(discards) >= 5 else discards
    if len(recent) >= 2:
        nums = [t.number for t in recent]
        feats.append(np.std(nums) / 4.0)
    else:
        feats.append(0.0)

    # [14] has discarded any missing suit tile
    if ms is not None:
        has_ms = any(t.suit == ms for t in discards)
        feats.append(1.0 if has_ms else 0.0)
    else:
        feats.append(0.0)

    # [15] discard speed relative to average
    total_discards_all = sum(len(p.discarded_tiles) for p in game.players)
    avg_discards = total_discards_all / 4.0
    if avg_discards > 0:
        feats.append(len(discards) / (avg_discards * 2.0))
    else:
        feats.append(0.5)

    # [16] open meld average tile number
    if target.open_melds:
        all_meld_nums = [t.number for meld in target.open_melds for t in meld]
        feats.append(sum(all_meld_nums) / len(all_meld_nums) / 9.0)
    else:
        feats.append(0.0)

    # [17] hand size ratio
    feats.append(max(0, n_conceal) / 13.0)

    assert len(feats) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feats)}"
    return feats


# ── Game simulation for data collection ──────────────────────────────────

def init_game(seed: int) -> MahjongGame:
    random.seed(seed)
    game = MahjongGame(
        game_id=f"belief_collect_{seed}",
        player_names=["P0", "P1", "P2", "P3"],
        bots=[False, True, True, True],
    )
    game.start_game()
    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))
    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))
    return game


def play_and_collect(seed: int, mc_samples: int = 20) -> List[Tuple[List[float], float]]:
    """Play one game, collect (features, tenpai_prob) at each step for each opponent."""
    game = init_game(seed)
    data = []
    skip_draw = True  # Dealer doesn't draw on first turn

    for step in range(200):
        # Check game over
        if game.is_game_over:
            break
        if game.deck.remaining_count() == 0 or sum(1 for p in game.players if p.is_hu) >= 3:
            game.check_game_over()
            break

        pid = game.current_player_id
        player = game.players[pid]

        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        # Draw tile
        if not skip_draw:
            drawn = game.draw_tile(pid)
            if not drawn:
                game.check_game_over()
                break
        else:
            skip_draw = False

        # Collect belief data for all 3 opponents (from P0's perspective)
        if step % 3 == 0:  # sample every 3 steps to reduce redundancy
            for target_pid in (1, 2, 3):
                if game.players[target_pid].is_hu:
                    continue
                feats = extract_opponent_public_features(game, target_pid, 0)
                result = opponent_view_posterior(game, target_pid, 0, num_samples=mc_samples)
                tenpai_prob = result["tenpai_prob"]
                data.append((feats, tenpai_prob))

        # Bot decides action
        action = bot_decide_turn_action(player, game)

        # Execute action
        if action.startswith("d "):
            from game import parse_console_tile
            try:
                tile_str = action[2:]
                tile = parse_console_tile(tile_str)
                game.discard_tile(pid, tile)
                # Handle responses
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

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=500, help="Number of games to simulate")
    parser.add_argument("--mc-samples", type=int, default=20, help="MC samples for ground truth")
    parser.add_argument("--seed", type=int, default=50260627, help="Base seed")
    parser.add_argument("--output", type=str, default="belief_surrogate_data.npz", help="Output file")
    args = parser.parse_args()

    print(f"Collecting belief data: {args.num_games} games, mc_samples={args.mc_samples}")
    t0 = time.time()

    all_features = []
    all_labels = []
    total_samples = 0

    for i in range(args.num_games):
        seed = args.seed + i
        game_data = play_and_collect(seed, mc_samples=args.mc_samples)
        for feats, prob in game_data:
            all_features.append(feats)
            all_labels.append(prob)
        total_samples += len(game_data)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Games: {i+1}/{args.num_games} | Samples: {total_samples} | "
                  f"Elapsed: {elapsed:.1f}s | Rate: {(i+1)/elapsed:.1f} games/s")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Total samples: {len(y)}")
    print(f"Feature dim: {X.shape[1]}")
    print(f"Label stats: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    np.savez_compressed(args.output, features=X, labels=y)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

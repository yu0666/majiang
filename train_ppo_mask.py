"""PPO training script for MASK parameter optimization.

Trains a policy network to output 12 parameters that control:
- Risk gate (kappa1, kappa2, kappa3, rho_max)
- Q_base weights (w_shanten, w_ukeire, w_value, w_shape)
- Deception weights (w_b, w_d, w_f, w_tell)

The reward is the agent's net score at the end of each game.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from belief_oracle import opponent_view_posterior
from collect_belief_data import extract_opponent_public_features
from game import MahjongGame, bot_decide_exchange, bot_decide_missing_suit, bot_decide_response, bot_decide_turn_action, parse_console_tile
from train_belief_surrogate import BeliefSurrogate
from mask_llm import MASKLLMAgent, PublicOpponentTracker, RiskGate, _clip, within_shanten
from policy_metrics import discard_progress_metrics
from prompt_builder import get_legal_actions
from ppo_agent import (
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    PARAM_DIM,
    PARAM_NAMES,
    MASKPolicyNet,
    PPOBuffer,
    Trajectory,
    TrajectoryStep,
    unscale_params,
)
from ppo_features import STATE_DIM, extract_state_features
from rule_engine import ShantenCalculator
from tiles import Tile


def _get_potential_fan(player) -> int:
    """Get potential fan from player, handling tuple return."""
    result = player.calculate_potential_fan()
    if isinstance(result, tuple):
        return result[0] if result else 0
    return int(result) if result else 0


def init_game_ppo(seed: int, opponent_style: str = "responsive") -> Tuple[MahjongGame, Dict]:
    """Initialize a game for PPO training."""
    random.seed(seed)
    game = MahjongGame(f"ppo_{seed}", ["Agent", "B1", "B2", "B3"], bots=[False, True, True, True])
    
    # Deal tiles
    game.start_game()
    
    # Exchange tiles (each player selects 3 tiles of same suit to exchange)
    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))
    
    # Set missing suit
    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))
    
    return game, {"seed": seed, "opponent_style": opponent_style}


def ppo_decide_action(
    game: MahjongGame,
    player_id: int,
    params: torch.Tensor,
    z_tracker: PublicOpponentTracker,
    risk_gate: RiskGate,
) -> Tuple[str, Dict[str, Any]]:
    """Make a decision using PPO-generated parameters.
    
    This is a simplified version of _continuous_gate_action that uses
    the PPO parameters instead of hardcoded values.
    
    Args:
        game: Current game state
        player_id: Agent's player ID
        params: 12-dim tensor of parameters (unscaled, actual values)
        z_tracker: Opponent tracking state
        risk_gate: Risk gate computer
    
    Returns:
        action: Chosen action string
        decision_info: Dict with decision details
    """
    # Unpack parameters
    param_dict = {name: params[i].item() for i, name in enumerate(PARAM_NAMES)}
    kappa1 = param_dict["kappa1"]
    kappa2 = param_dict["kappa2"]
    kappa3 = param_dict["kappa3"]
    rho_max = param_dict["rho_max"]
    w_shanten = param_dict["w_shanten"]
    w_ukeire = param_dict["w_ukeire"]
    w_value = param_dict["w_value"]
    w_shape = param_dict["w_shape"]
    w_b = param_dict["w_b"]
    w_d = param_dict["w_d"]
    w_f = param_dict["w_f"]
    
    valid_actions = get_legal_actions(game, player_id)
    player = game.players[player_id]
    
    # Handle hu
    if "h" in valid_actions:
        return "h", {"mode": "exploit", "reason": "take win"}
    
    # Get discard actions
    discard_actions = [a for a in valid_actions if a.startswith("d ")]
    if not discard_actions:
        # Fall back to first valid action
        return valid_actions[0], {"mode": "fallback", "reason": "no discard"}
    
    # Get z_state and beliefs
    z_state = z_tracker.summary()
    beliefs = {}
    for target_pid in (1, 2, 3):
        beliefs[f"P{target_pid}"] = opponent_view_posterior(game, target_pid, 0, num_samples=20)
    
    # Compute gate
    gate = risk_gate.compute(game, player_id, z_state, beliefs)
    
    # Compute alpha (risk appetite)
    rho = float(gate.get("risk_budget", 0.0))
    u = float(gate.get("uncertainty", 0.0))
    score_gap = float(gate.get("score_gap", 0.0))
    tiles_left = float(gate.get("tiles_left", 40.0))
    behind = _clip(-score_gap / 3000.0)
    late = _clip((40.0 - tiles_left) / 40.0)
    risk_appetite = 0.5 * behind + 0.5 * late
    
    # Value gate
    max_potential_fan = 0
    for action in discard_actions:
        value = discard_progress_metrics(game, player_id, action)
        if value and value["shanten"] == 0:
            tile_text = action[2:]
            tile = next((t for t in player.hand_tiles if str(t) == tile_text), None)
            if tile:
                hand_after = player.hand_tiles.copy()
                hand_after.remove(tile)
                pf = _get_potential_fan(player)
                max_potential_fan = max(max_potential_fan, pf)
    value_gate = _clip(max_potential_fan / 3.0)
    
    # Compute alpha with PPO parameters
    logit = kappa1 * risk_appetite - kappa2 * u - kappa3 * (1.0 if rho > rho_max else 0.0)
    alpha = (1.0 / (1.0 + math.exp(-logit))) * value_gate
    
    # Score each discard action
    tell_before = _compute_tell_threat(game, player_id)
    avg_belief_conf = 0.0
    if beliefs:
        confs = [float(b.get("tenpai_confidence", 0.0)) for b in beliefs.values()]
        avg_belief_conf = _clip(sum(confs) / len(confs)) if confs else 0.0
    
    scored = {}
    for action in discard_actions:
        value = discard_progress_metrics(game, player_id, action)
        if value is None:
            continue
        
        tile_text = action[2:]
        tile = next((t for t in player.hand_tiles if str(t) == tile_text), None)
        
        # Q_base
        q_base = (
            -w_shanten * value["shanten"]
            + w_ukeire * value["effective_copies"]
        )
        
        # Add potential fan if tenpai
        potential_fan = 0
        if tile and value["shanten"] == 0:
            potential_fan = _get_potential_fan(player)
        q_base += w_value * potential_fan
        
        # DeltaShape
        tell_after = _compute_tell_threat(game, player_id, extra_tile=tile)
        b_term = (tell_after - tell_before) * (1.0 - avg_belief_conf)
        d_term = 0.0
        if tile and within_shanten(player.hand_tiles, player.missing_suit, 1):
            hand_after = player.hand_tiles.copy()
            hand_after.remove(tile)
            if within_shanten(hand_after, player.missing_suit, 0):
                d_term = (1.0 - avg_belief_conf)
        f_term = tell_after * avg_belief_conf * _clip(value["shanten"] / 3.0)
        delta_shape = w_b * b_term + w_d * d_term + w_f * f_term
        
        score = q_base + alpha * delta_shape
        scored[action] = {
            "score": score,
            "q_base": q_base,
            "delta_shape": delta_shape,
            "shanten": value["shanten"],
            "ukeire": value["effective_copies"],
        }
    
    if not scored:
        return discard_actions[0], {"mode": "fallback", "reason": "no scored actions"}
    
    # Choose best action
    best_action = max(scored, key=lambda a: scored[a]["score"])
    
    # Determine mode
    best_q_only = max(scored, key=lambda a: scored[a]["q_base"])
    if best_action != best_q_only and alpha * scored[best_action]["delta_shape"] > 0:
        mode = "deceive"
    elif rho > rho_max or gate.get("mode_hint") == "safe":
        mode = "safe"
    else:
        mode = "exploit"
    
    return best_action, {
        "mode": mode,
        "alpha": alpha,
        "score": scored[best_action]["score"],
        "params": param_dict,
    }


def _compute_tell_threat(game: MahjongGame, player_id: int, extra_tile: Optional[Tile] = None) -> float:
    """Compute tell threat heuristic."""
    player = game.players[player_id]
    discards = list(player.discarded_tiles)
    if extra_tile:
        discards = discards + [extra_tile]
    if not discards:
        return 0.0
    terminal_count = sum(1 for t in discards if t.number in (1, 9))
    return _clip(terminal_count / max(1, len(discards)))


def play_one_game_ppo(
    policy: MASKPolicyNet,
    seed: int,
    opponent_style: str = "responsive",
    max_steps: int = 200,
    deterministic: bool = False,
    collect_trajectory: bool = True,
    mc_samples: int = 5,
    belief_surrogate: Optional[BeliefSurrogate] = None,
) -> Tuple[float, Optional[Trajectory]]:
    """Play one game using PPO policy for agent decisions.
    
    Follows the same game loop structure as run_gate1_experiments.py.
    If belief_surrogate is provided, use it instead of MC sampling.
    """
    device = next(policy.parameters()).device
    game, game_info = init_game_ppo(seed, opponent_style)
    start_balance = game.players[0].balance
    
    z_tracker = PublicOpponentTracker([1, 2, 3])
    risk_gate = RiskGate()
    
    steps = 0
    trajectory_steps = []
    skip_draw = True  # Dealer doesn't draw on first turn
    
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
        
        # Draw tile
        drawn_tile = None
        if not skip_draw:
            drawn_tile = game.draw_tile(pid)
            if not drawn_tile:
                game.check_game_over()
                break
        else:
            skip_draw = False
        
        if pid == 0:
            # Agent's turn - use PPO policy
            z_state = z_tracker.summary()
            beliefs = {}
            if belief_surrogate is not None:
                # Fast: learned surrogate (microsecond inference)
                belief_surrogate.eval()
                with torch.no_grad():
                    for target_pid in (1, 2, 3):
                        feats = extract_opponent_public_features(game, target_pid, 0)
                        feat_tensor = torch.tensor(feats, dtype=torch.float32, device=device)
                        prob = belief_surrogate(feat_tensor).item()
                        beliefs[f"P{target_pid}"] = {"tenpai_prob": prob, "tenpai_confidence": prob}
            else:
                # Slow: MC sampling
                for target_pid in (1, 2, 3):
                    beliefs[f"P{target_pid}"] = opponent_view_posterior(game, target_pid, 0, num_samples=mc_samples)
            gate = risk_gate.compute(game, 0, z_state, beliefs)
            
            # Extract features and get parameters from policy
            state_features = extract_state_features(game, 0, z_state, beliefs, gate)
            state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                params_scaled, log_prob, value = policy.get_action(state_tensor, deterministic=deterministic)
                params_actual = unscale_params(params_scaled)
            
            # Make decision
            action, decision_info = ppo_decide_action(game, 0, params_actual, z_tracker, risk_gate)
            
            # Record trajectory step
            if collect_trajectory:
                trajectory_steps.append(TrajectoryStep(
                    state=np.array(state_features),
                    action=params_scaled.cpu().numpy(),
                    log_prob=log_prob.item(),
                    reward=0.0,  # Will be filled at end
                    value=value.item(),
                    done=False,
                ))
            
            # Execute action
            discarded_tile = execute_action(game, 0, action, drawn_tile)
            if game.is_game_over:
                break
            
            if discarded_tile is None:
                if action == "g":
                    skip_draw = True
                    continue
                game.next_player()
                skip_draw = False
                continue
            
            # Resolve responses
            response_info = resolve_responses_simple(game, discarded_tile, 0)
            if game.is_game_over:
                break
            if response_info["responded"]:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False
        else:
            # Opponent's turn
            action = bot_decide_turn_action(player, game)
            
            discarded_tile = execute_action(game, pid, action, drawn_tile)
            if game.is_game_over:
                break
            
            # Track opponent action via game history
            z_tracker.update_from_game(game)
            
            if discarded_tile is None:
                if action == "g":
                    skip_draw = True
                    continue
                game.next_player()
                skip_draw = False
                continue
            
            # Resolve responses
            response_info = resolve_responses_simple(game, discarded_tile, pid)
            if game.is_game_over:
                break
            if response_info["responded"]:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False
    
    if not game.is_game_over:
        game.check_game_over()
    
    # Compute final reward
    final_balance = game.players[0].balance
    net_score = final_balance - start_balance
    
    # Build trajectory
    trajectory = None
    if collect_trajectory and trajectory_steps:
        # Set reward for all steps to episode return
        for step in trajectory_steps:
            step.reward = net_score
            step.done = True
        trajectory = Trajectory(
            steps=trajectory_steps,
            episode_return=net_score,
            episode_length=len(trajectory_steps),
        )
    
    return net_score, trajectory


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None) -> Optional[object]:
    """Execute an action in the game. Returns discarded tile or None."""
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

    # Fallback: discard first legal tile
    if discarded_tile is None:
        legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
        if legal_discards:
            tile = parse_console_tile(legal_discards[0][2:])
            if tile and game.discard_tile(pid, tile):
                discarded_tile = tile

    return discarded_tile


def resolve_responses_simple(game: MahjongGame, discarded_tile, acting_pid: int) -> Dict[str, Any]:
    """Simplified response resolution for PPO training."""
    responses = game.check_responses(discarded_tile, acting_pid)
    responded = False
    agent_won = False
    agent_dealt_in = False

    for rid, acts in sorted(responses.items()):
        if responded:
            break
        response_action = bot_decide_response(game.players[rid], acts)

        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            responded = True
            agent_won = (rid == 0)
            agent_dealt_in = (acting_pid == 0 and rid != 0)
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
        "agent_won_by_discard": agent_won,
        "agent_dealt_in": agent_dealt_in,
    }


def ppo_update(
    policy: MASKPolicyNet,
    optimizer: optim.Optimizer,
    buffer: PPOBuffer,
    epochs: int = 4,
    clip_eps: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
) -> Dict[str, float]:
    """Perform PPO update on collected trajectories."""
    data = buffer.get_training_data(policy)
    if not data:
        return {}
    
    states = data["states"]
    actions = data["actions"]
    old_log_probs = data["old_log_probs"]
    advantages = data["advantages"]
    returns = data["returns"]
    
    # Normalize advantages (standard PPO practice)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0
    
    for epoch in range(epochs):
        # Evaluate current policy
        new_log_probs, values, entropy = policy.evaluate_actions(states, actions)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (with clipping to prevent explosion)
        returns_clipped = torch.clamp(returns, -500.0, 500.0)
        value_loss = F.mse_loss(values, returns_clipped)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.mean().item()
        num_updates += 1
    
    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
    }


def train_ppo(
    num_iterations: int = 500,
    games_per_iteration: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    eval_freq: int = 10,
    eval_games: int = 20,
    save_dir: str = "ppo_checkpoints",
    device: str = "cpu",
    seed: int = 42,
    mc_samples: int = 5,
    belief_surrogate_path: Optional[str] = None,
):
    """Main PPO training loop."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create policy and optimizer
    policy = MASKPolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Load belief surrogate if provided
    belief_surrogate = None
    if belief_surrogate_path and os.path.exists(belief_surrogate_path):
        ckpt = torch.load(belief_surrogate_path, map_location=device, weights_only=False)
        belief_surrogate = BeliefSurrogate(input_dim=ckpt.get("input_dim", 18)).to(device)
        belief_surrogate.load_state_dict(ckpt["model_state"])
        belief_surrogate.eval()
        print(f"Loaded belief surrogate from {belief_surrogate_path}")
    
    # Training buffer
    buffer = PPOBuffer()
    
    # Logging
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    log_file = save_path / "training_log.jsonl"
    
    best_avg_return = -float("inf")
    
    print(f"Starting PPO training for {num_iterations} iterations")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print()
    
    for iteration in range(1, num_iterations + 1):
        start_time = time.time()
        buffer.clear()
        
        # Collect trajectories
        returns = []
        for game_idx in range(games_per_iteration):
            game_seed = seed * 10000 + iteration * 1000 + game_idx
            net_score, trajectory = play_one_game_ppo(
                policy,
                seed=game_seed,
                opponent_style="responsive",
                max_steps=200,
                deterministic=False,
                collect_trajectory=True,
                mc_samples=mc_samples,
                belief_surrogate=belief_surrogate,
            )
            returns.append(net_score)
            if trajectory:
                buffer.add(trajectory)
        
        # PPO update
        update_stats = ppo_update(
            policy, optimizer, buffer,
            epochs=ppo_epochs,
            clip_eps=clip_eps,
        )
        
        # Logging
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        elapsed = time.time() - start_time
        
        log_entry = {
            "iteration": iteration,
            "avg_return": avg_return,
            "std_return": std_return,
            "total_steps": buffer.total_steps,
            "elapsed": elapsed,
            **update_stats,
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        if iteration % 10 == 0 or iteration == 1:
            print(f"Iter {iteration:4d} | Avg: {avg_return:7.2f} +/- {std_return:6.2f} | "
                  f"Steps: {buffer.total_steps:5d} | Time: {elapsed:5.1f}s")
            if update_stats:
                print(f"           | Policy Loss: {update_stats.get('policy_loss', 0):.4f} | "
                      f"Value Loss: {update_stats.get('value_loss', 0):.4f}")
        
        # Evaluation and checkpointing
        if iteration % eval_freq == 0:
            eval_returns = []
            for eval_idx in range(eval_games):
                eval_seed = seed * 100000 + iteration * 100 + eval_idx
                net_score, _ = play_one_game_ppo(
                    policy,
                    seed=eval_seed,
                    opponent_style="responsive",
                    max_steps=200,
                    deterministic=True,
                    collect_trajectory=False,
                    mc_samples=mc_samples,
                    belief_surrogate=belief_surrogate,
                )
                eval_returns.append(net_score)
            
            eval_avg = np.mean(eval_returns)
            print(f"  [Eval] Iter {iteration}: avg_net = {eval_avg:.2f} (n={eval_games})")
            
            if eval_avg > best_avg_return:
                best_avg_return = eval_avg
                checkpoint_path = save_path / "best_policy.pt"
                torch.save({
                    "iteration": iteration,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_return": eval_avg,
                }, checkpoint_path)
                print(f"  [Saved] New best policy: {eval_avg:.2f}")
    
    # Save final policy
    final_path = save_path / "final_policy.pt"
    torch.save({
        "iteration": num_iterations,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete. Final policy saved to {final_path}")
    print(f"Best eval return: {best_avg_return:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for MASK parameters")
    parser.add_argument("--iterations", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=64, help="Games per iteration")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda-gae", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--eval-freq", type=int, default=10, help="Evaluation frequency")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of eval games")
    parser.add_argument("--save-dir", type=str, default="ppo_checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mc-samples", type=int, default=5, help="MC samples for belief estimation (lower=faster)")
    parser.add_argument("--belief-surrogate", type=str, default=None, help="Path to belief_surrogate.pt (replaces MC if provided)")
    
    args = parser.parse_args()
    
    train_ppo(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        lr=args.lr,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        mc_samples=args.mc_samples,
        belief_surrogate_path=args.belief_surrogate,
    )

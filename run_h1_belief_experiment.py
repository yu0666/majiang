"""
H1 / E1 pre-experiment: B_phi belief-estimator validity.

Goal:
  Test whether a public-information belief estimator can predict how dangerous
  P0 appears to opponents, using a coarse opponent-view oracle for "P0 is
  tenpai now" as the first runnable label.

Baselines:
  B0_frequency_prior : constant prior from train split.
  B1_public_z        : public-only logistic heuristic from melds/discards/phase.
  B2_B_phi           : LLMBeliefEstimator; with backend=heuristic_fallback it is
                       a public-only B_phi skeleton, with backend=local_qwen it
                       calls the local LLM and parses JSON.

This is not yet the final posterior oracle over concrete hidden hands.  It is
the pre-experiment that replaces the deleted old numeric H1/H2 probes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from belief_oracle import is_tenpai, opponent_view_posterior
from experiment_trace import (
    apply_isotonic,
    balance_by_binary,
    binary_auc,
    brier_score,
    calibration_error,
    ensure_deterministic_hashing,
    fit_isotonic,
    sign_test_p_value,
    write_json,
    write_jsonl,
)
from game import (
    STYLE_BOTS,
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    parse_console_tile,
)
from llm_backends import build_llm_callable
from mask_llm import LLMBeliefEstimator, PublicOpponentTracker
from prompt_builder import get_legal_actions
from rule_engine import ShantenCalculator


RESULTS_DIR = Path("H1_belief_results")
OPPONENT_STYLES = tuple(sorted(STYLE_BOTS.keys())) + ("mixed",)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def init_game(seed: int, opponent_style: str, game_id: str) -> Tuple[MahjongGame, Dict[int, Tuple]]:
    random.seed(seed)
    game = MahjongGame(game_id, ["Agent", "B1", "B2", "B3"], bots=[False, True, True, True])
    if opponent_style == "mixed":
        styles = ["aggressive", "conservative", "random"]
        opponent_funcs = {pid: STYLE_BOTS[styles[pid - 1]] for pid in (1, 2, 3)}
    else:
        opponent_funcs = {pid: STYLE_BOTS.get(opponent_style, STYLE_BOTS["greedy"]) for pid in (1, 2, 3)}

    game.start_game()
    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))
    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))
    return game, opponent_funcs


def choose_min_shanten_action(game: MahjongGame, player_id: int) -> str:
    valid_actions = get_legal_actions(game, player_id)
    player = game.players[player_id]
    if "h" in valid_actions:
        return "h"
    if "g" in valid_actions:
        return "g"

    best_action = valid_actions[0] if valid_actions else "n"
    best_shanten = 99
    for action in valid_actions:
        if not action.startswith("d "):
            continue
        tile_text = action[2:]
        for tile in player.hand_tiles:
            if str(tile) != tile_text:
                continue
            temp = player.hand_tiles.copy()
            temp.remove(tile)
            shanten = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
            if shanten < best_shanten:
                best_shanten = shanten
                best_action = action
            break
    return best_action


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None) -> Optional[object]:
    player = game.players[pid]
    action = (action or "").strip()

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
            return tile

    legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
    if legal_discards:
        tile = parse_console_tile(legal_discards[0][2:])
        if tile and game.discard_tile(pid, tile):
            return tile
    return None


def resolve_responses(game: MahjongGame, discarded_tile, acting_pid: int) -> bool:
    responses = game.check_responses(discarded_tile, acting_pid)
    for rid, acts in responses.items():
        player = game.players[rid]
        response_action = bot_decide_response(player, acts)
        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            return True
        if response_action == "g" and "gang" in acts:
            game.gang(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            return True
        if response_action == "p" and "peng" in acts:
            game.peng(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            return True
    return False


def ground_truth_label(game: MahjongGame, player_id: int = 0, max_shanten: int = 0) -> Dict[str, Any]:
    """Ground truth about the target's real hand (used only for AUC).

    ``true_tenpai`` is the realized positive class: shanten<=max_shanten
    (max_shanten=0 -> precise tenpai; =1 -> 'danger', tenpai-or-one-away).
    Count-agnostic so it is correct on 14-tile post-draw states (the old
    is_ready check forced False there, which crushed the positive rate to ~2%).
    """
    player = game.players[player_id]
    shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
    ready, waits = player.is_ready_with_missing_suit()
    return {
        "true_tenpai": bool(shanten <= max_shanten),
        "true_waits": [str(t) for t in waits],
        "true_shanten": int(shanten),
        "danger_threshold": max_shanten,
    }


def opponent_view_oracle(
    game: MahjongGame, target_pid: int, observer_pid: int, num_samples: int, rng: random.Random,
    beta: float = 2.0, max_shanten: int = 0,
) -> Dict[str, Any]:
    """Per-row oracle: realized tenpai outcome + descriptive opponent-view MC posterior.

    The gate is scored against the realized binary ``true_tenpai`` (a consistent
    way to validate a posterior estimator), while ``mc_opponent_posterior`` is a
    per-observer descriptive signal.  NOTE: the MC posterior uses a *uniform*
    prior over consistent hands, so its absolute level is low (a random hand is
    rarely tenpai); a play-aware prior is future work, hence it is descriptive
    only and not the gate target.
    """
    gt = ground_truth_label(game, target_pid, max_shanten=max_shanten)
    mc = None
    if num_samples > 0:
        post = opponent_view_posterior(
            game, target_pid, observer_pid, num_samples=num_samples, rng=rng, beta=beta, max_shanten=max_shanten,
        )
        mc = float(post["tenpai_prob"])
    return {
        "true_tenpai": gt["true_tenpai"],
        "true_waits": gt["true_waits"],
        "true_shanten": gt["true_shanten"],
        "mc_opponent_posterior": mc,
        "label_scope": "gate target = realized true_tenpai (public-info estimators); mc_opponent_posterior is descriptive per-observer (uniform-prior)",
    }


def public_features(game: MahjongGame, player_id: int, target_id: int, z_summary: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    player = game.players[player_id]
    own_discards = len(player.discarded_tiles)
    meld_count = len(player.open_melds)
    missing_discards = 0
    if player.missing_suit:
        missing_discards = sum(1 for t in player.discarded_tiles if t.suit == player.missing_suit)

    target_state = z_summary.get(target_id, {})
    z_vec = target_state.get("z_public_vector", {})
    return {
        "own_discards": own_discards,
        "own_melds": meld_count,
        "missing_discards": missing_discards,
        "tiles_left": game.deck.remaining_count(),
        "target_z_label": target_state.get("z_label", "unknown"),
        "target_drift_score": target_state.get("drift_score", 0.0),
        "target_peng_rate": z_vec.get("peng_rate", 0.0),
        "target_mid_discard_rate": z_vec.get("mid_discard_rate", 0.0),
        "target_terminal_discard_rate": z_vec.get("terminal_discard_rate", 0.0),
    }


def b1_public_z_score(features: Dict[str, Any]) -> float:
    logit = -1.7
    logit += 0.22 * features["own_discards"]
    logit += 0.75 * features["own_melds"]
    logit += 0.18 * features["missing_discards"]
    logit += 0.035 * max(0, 55 - features["tiles_left"])
    logit += 0.45 * features["target_peng_rate"]
    logit += 0.25 * features["target_mid_discard_rate"]
    logit += 0.15 * features["target_drift_score"]
    return max(0.0, min(1.0, sigmoid(logit)))


def collect_samples(
    games: int,
    seed: int,
    opponent_style: str,
    max_steps: int,
    sample_every: int,
    llm,
    backend: str,
    oracle_samples: int = 60,
    oracle_beta: float = 2.0,
    danger_threshold: int = 0,
    b_phi_source: str = "llm",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    estimator = LLMBeliefEstimator(llm=llm)

    for game_idx in range(games):
        game_seed = seed + game_idx
        game, opponent_funcs = init_game(game_seed, opponent_style, f"H1_{game_seed}")
        tracker = PublicOpponentTracker([1, 2, 3])
        oracle_rng = random.Random(game_seed * 1000 + 7)
        skip_draw = True
        steps = 0

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

            z_summary = tracker.update_from_game(game)
            if pid == 0 and (steps == 1 or steps % sample_every == 0):
                for target_id in (1, 2, 3):
                    # Per-observer oracle: j's belief about P0 from j's info set.
                    label = opponent_view_oracle(game, 0, target_id, oracle_samples, oracle_rng, beta=oracle_beta, max_shanten=danger_threshold)
                    features = public_features(game, 0, target_id, z_summary)
                    if b_phi_source == "mc":
                        # B_phi := the MC public-info posterior (no LLM call).
                        belief = {"tenpai_confidence": float(label.get("mc_opponent_posterior") or 0.0),
                                  "b_phi_source": "mc"}
                        latency_ms = 0.0
                    else:
                        t0 = time.perf_counter()
                        belief = estimator.infer(game, 0, target_id)
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                    rows.append(
                        {
                            "game_id": game.game_id,
                            "seed": game_seed,
                            "step": steps,
                            "t": len(game.history),
                            "j": target_id,
                            "backend": backend,
                            "H_pub": game.get_history_text(k=20),
                            "a_i_history": [r for r in game.history if r.get("pid") == 0],
                            "public_features": features,
                            "b_j_oracle": label,
                            "b_j_hat": belief,
                            "B0_frequency_prior": None,
                            "B1_public_z": b1_public_z_score(features),
                            "B2_B_phi": float(belief.get("tenpai_confidence", 0.5)),
                            "latency_ms": latency_ms,
                        }
                    )

            if pid == 0:
                action = choose_min_shanten_action(game, 0)
            else:
                action = opponent_funcs[pid][0](player, game)

            discarded_tile = execute_action(game, pid, action, drawn_tile)
            if game.is_game_over:
                break
            if discarded_tile is None:
                if action == "g":
                    skip_draw = True
                    continue
                game.next_player()
                skip_draw = False
                continue

            responded = resolve_responses(game, discarded_tile, pid)
            if game.is_game_over:
                break
            if responded:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False

    return rows


def split_train_eval(rows: List[Dict[str, Any]], train_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    cutoff = max(1, int(round(len(seeds) * train_ratio)))
    train_seeds = set(seeds[:cutoff])
    train = [row for row in rows if int(row["seed"]) in train_seeds]
    eval_rows = [row for row in rows if int(row["seed"]) not in train_seeds]
    if not eval_rows:
        return train, train
    return train, eval_rows


def fill_b0_prior(train: List[Dict[str, Any]], eval_rows: List[Dict[str, Any]]) -> float:
    """B0 = marginal tenpai base rate from train (the no-context frequency prior)."""
    if not train:
        prior = 0.5
    else:
        prior = sum(1.0 if r["b_j_oracle"]["true_tenpai"] else 0.0 for r in train) / len(train)
    prior = max(0.01, min(0.99, prior))
    for row in eval_rows:
        row["B0_frequency_prior"] = prior
    return prior


def fit_isotonic_calibrator(train: List[Dict[str, Any]], key: str) -> List[Tuple[float, float]]:
    """Monotonic calibration of a score against the binary tenpai outcome.

    Isotonic (PAV) is order-preserving, so unlike the previous independent-bin
    histogram it does not collapse AUC when it fixes calibration.
    """
    pairs = [
        (max(0.0, min(1.0, float(row[key]))), 1.0 if row["b_j_oracle"]["true_tenpai"] else 0.0)
        for row in train
    ]
    return fit_isotonic(pairs)


def apply_isotonic_calibrator(rows: List[Dict[str, Any]], source_key: str, target_key: str, steps: List[Tuple[float, float]]) -> None:
    for row in rows:
        score = max(0.0, min(1.0, float(row[source_key])))
        row[target_key] = apply_isotonic(steps, score)


def summarize_estimator(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    labels = [1 if row["b_j_oracle"]["true_tenpai"] else 0 for row in rows]
    mc_vals = [row["b_j_oracle"].get("mc_opponent_posterior") for row in rows]
    mc_present = [v for v in mc_vals if v is not None]
    scores = [float(row[key]) for row in rows]
    preds = [1 if score >= 0.5 else 0 for score in scores]
    n = len(rows)
    if n == 0:
        return {"samples": 0}
    # Scored against the realized binary outcome (BSE here == |score - outcome|).
    bse_values = [abs(score - label) for score, label in zip(scores, labels)]
    return {
        "samples": n,
        "positive_rate": sum(labels) / n,
        "mc_posterior_mean": (sum(mc_present) / len(mc_present)) if mc_present else None,
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / n,
        "auc": binary_auc(labels, scores),
        "brier_binary": sum((score - y) ** 2 for y, score in zip(labels, scores)) / n,
        "BSE_abs": sum(bse_values) / n,
        "ECE": calibration_error(labels, scores),
        "latency_ms_avg": (
            sum(float(row.get("latency_ms", 0.0)) for row in rows) / n if key == "B2_B_phi" else 0.0
        ),
    }


def brier_gate(
    b2: Dict[str, Any],
    b0: Dict[str, Any],
    b1: Dict[str, Any],
    auc_threshold: float,
    rel_threshold: float,
    eval_scope: str,
) -> Dict[str, Any]:
    """Proper-scoring gate evaluated on a class-balanced set.

    Brier (a strictly proper scoring rule) replaces absolute error to the soft
    label.  On a balanced eval set the constant base-rate predictor B0 can no
    longer win by predicting near-zero, so beating it actually requires
    discrimination.  The AUC requirement is kept as the discrimination floor.
    """
    b2b = b2.get("brier_binary")
    b0b = b0.get("brier_binary")
    b1b = b1.get("brier_binary")
    auc = b2.get("auc") or 0.0
    if b2b is None or not b0b or not b1b:
        return {"pass": False, "reason": "Insufficient Brier values.", "eval_scope": eval_scope}
    red0 = (b0b - b2b) / b0b
    red1 = (b1b - b2b) / b1b
    return {
        "eval_scope": eval_scope,
        "B2_brier": b2b,
        "B0_brier": b0b,
        "B1_brier": b1b,
        "B2_auc": auc,
        "B2_vs_B0_relative_brier_reduction": red0,
        "B2_vs_B1_relative_brier_reduction": red1,
        "auc_requirement": f">= {auc_threshold}",
        "brier_reduction_requirement": f">= {rel_threshold} against B0/B1",
        "pass": red0 >= rel_threshold and red1 >= rel_threshold and auc >= auc_threshold,
    }


def paired_brier_sign_test(eval_rows: List[Dict[str, Any]], key_b2: str, key_ref: str) -> Dict[str, Any]:
    """Per-sample paired test that B2's squared error beats a reference's.

    delta = (ref - y)^2 - (B2 - y)^2 against the realized outcome y; positive
    when B2 is the better probabilistic predictor.  Two-sided sign-test p-value.
    """
    deltas = []
    for row in eval_rows:
        y = 1.0 if row["b_j_oracle"]["true_tenpai"] else 0.0
        err_b2 = (float(row[key_b2]) - y) ** 2
        err_ref = (float(row[key_ref]) - y) ** 2
        deltas.append(err_ref - err_b2)
    n = len(deltas)
    return {
        "n": n,
        "mean_brier_gain": (sum(deltas) / n) if n else 0.0,
        "win_rate": (sum(1 for d in deltas if d > 0) / n) if n else 0.0,
        "sign_test_p": sign_test_p_value(deltas),
    }


def spec_gate(
    summary: Dict[str, Any],
    eval_rows: List[Dict[str, Any]],
    auc_threshold: float,
    brier_reduction: float,
    p_threshold: float,
    eval_scope: str,
    min_samples: int = 40,
) -> Dict[str, Any]:
    """H1 gate on the balanced eval, scored against the realized tenpai outcome.

    Requirements (imbalance-robust restatement of the spec):
      * AUC(B2) >= auc_threshold              (discrimination floor, spec 0.75)
      * Brier(B2) beats B0 and B1 by >= margin (proper scoring, replaces the
        ill-posed "BSE vs near-constant label" requirement)
      * paired sign-test p < threshold vs B0 and B1 (spec: significance)
    """
    b2, b0, b1 = summary["B2_B_phi"], summary["B0_frequency_prior"], summary["B1_public_z"]
    br2, br0, br1 = b2.get("brier_binary"), b0.get("brier_binary"), b1.get("brier_binary")
    auc = b2.get("auc") or 0.0
    if br2 is None or not br0 or not br1:
        return {"pass": False, "reason": "Insufficient Brier values.", "eval_scope": eval_scope}
    red0 = (br0 - br2) / br0
    red1 = (br1 - br2) / br1
    test0 = paired_brier_sign_test(eval_rows, "B2_B_phi", "B0_frequency_prior")
    test1 = paired_brier_sign_test(eval_rows, "B2_B_phi", "B1_public_z")
    p_ok = (test0["sign_test_p"] is not None and test0["sign_test_p"] < p_threshold
            and test1["sign_test_p"] is not None and test1["sign_test_p"] < p_threshold)
    n_eval = len(eval_rows)
    underpowered = n_eval < min_samples
    return {
        "eval_scope": eval_scope,
        "n_eval": n_eval,
        "underpowered": underpowered,
        "underpowered_note": (
            f"balanced eval has {n_eval} < {min_samples} samples; verdict is statistically weak — raise --games."
            if underpowered else ""
        ),
        "B2_brier": br2, "B0_brier": br0, "B1_brier": br1, "B2_auc": auc,
        "B2_vs_B0_relative_brier_reduction": red0,
        "B2_vs_B1_relative_brier_reduction": red1,
        "paired_test_vs_B0": test0,
        "paired_test_vs_B1": test1,
        "requirements": {
            "brier_reduction": f">= {brier_reduction} vs B0 and B1",
            "auc": f">= {auc_threshold}",
            "paired_sign_test": f"p < {p_threshold} vs B0 and B1",
        },
        "pass": (red0 >= brier_reduction and red1 >= brier_reduction
                 and auc >= auc_threshold and p_ok and not underpowered),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    rows = collect_samples(
        games=args.games,
        seed=args.seed,
        opponent_style=args.opponent_style,
        max_steps=args.max_steps,
        sample_every=args.sample_every,
        llm=llm,
        backend=args.backend,
        oracle_samples=args.oracle_samples,
        oracle_beta=args.oracle_beta,
        danger_threshold=args.danger_threshold,
        b_phi_source=args.b_phi_source,
    )
    train, eval_rows = split_train_eval(rows, args.train_ratio)
    prior = fill_b0_prior(train, eval_rows)
    # Monotonic (isotonic) calibration fit on train only; preserves ranking/AUC.
    b1_calibrator = fit_isotonic_calibrator(train, "B1_public_z")
    b2_calibrator = fit_isotonic_calibrator(train, "B2_B_phi")
    apply_isotonic_calibrator(eval_rows, "B1_public_z", "B1_public_z_calibrated", b1_calibrator)
    apply_isotonic_calibrator(eval_rows, "B2_B_phi", "B2_B_phi_calibrated", b2_calibrator)

    # Class-balanced eval subset: makes Brier/BSE comparisons fair against the
    # base-rate constant B0 instead of letting the 2-3% positive rate dominate.
    balanced_eval, balanced_ok = balance_by_binary(
        eval_rows, lambda r: 1 if r["b_j_oracle"]["true_tenpai"] else 0, seed=args.seed
    )

    def estimator_summaries(rows_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "B0_frequency_prior": summarize_estimator(rows_subset, "B0_frequency_prior"),
            "B1_public_z": summarize_estimator(rows_subset, "B1_public_z"),
            "B1_public_z_calibrated": summarize_estimator(rows_subset, "B1_public_z_calibrated"),
            "B2_B_phi": summarize_estimator(rows_subset, "B2_B_phi"),
            "B2_B_phi_calibrated": summarize_estimator(rows_subset, "B2_B_phi_calibrated"),
        }

    estimator_summary = estimator_summaries(eval_rows)
    balanced_summary = estimator_summaries(balanced_eval) if balanced_ok else {}

    # Gate is decided on the balanced set with RAW B_phi vs raw baselines, scored
    # against the realized tenpai outcome: AUC (discrimination) + Brier reduction
    # (proper scoring) + paired sign-test (significance).  Calibration is fit on
    # the imbalanced train split, so on the balanced eval it shifts the base rate
    # and squashes scores; it is reported as a diagnostic, not the pass criterion.
    gate_source = balanced_summary if balanced_ok else estimator_summary
    gate_eval = balanced_eval if balanced_ok else eval_rows
    gate_scope = "balanced_eval" if balanced_ok else "natural_eval(no_balance_possible)"
    h1_gate = spec_gate(
        gate_source, gate_eval,
        auc_threshold=args.auc_threshold,
        brier_reduction=args.brier_reduction,
        p_threshold=args.p_threshold,
        eval_scope=gate_scope,
    )
    h1_gate_calibrated = brier_gate(
        gate_source["B2_B_phi_calibrated"],
        gate_source["B0_frequency_prior"],
        gate_source["B1_public_z_calibrated"],
        auc_threshold=args.auc_threshold,
        rel_threshold=args.brier_reduction,
        eval_scope=gate_scope,
    )

    summary = {
        "experiment": "H1_B_phi_validity_preexperiment",
        "backend": args.backend,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "opponent_style": args.opponent_style,
        "games": args.games,
        "samples_total": len(rows),
        "samples_train": len(train),
        "samples_eval": len(eval_rows),
        "samples_balanced_eval": len(balanced_eval) if balanced_ok else 0,
        "balanced_eval_available": balanced_ok,
        "B0_prior_from_train": prior,
        "calibration": {
            "type": "train-split isotonic (PAV) calibration against binary tenpai",
            "B1_public_z_steps": b1_calibrator,
            "B2_B_phi_steps": b2_calibrator,
        },
        "gate_config": {
            "primary_gate": "H1_gate (balanced eval): AUC + Brier reduction vs B0/B1 + paired sign-test, scored vs realized tenpai",
            "auc_threshold": args.auc_threshold,
            "brier_reduction": args.brier_reduction,
            "p_threshold": args.p_threshold,
            "oracle_samples": args.oracle_samples,
        },
        "estimators_natural_eval": estimator_summary,
        "estimators_balanced_eval": balanced_summary,
        "H1_gate": h1_gate,
        "H1_gate_calibrated_diagnostic": h1_gate_calibrated,
        "limitations": [
            "Gate target is realized true_tenpai (consistent estimator of the opponent posterior under public info); mc_opponent_posterior is descriptive only and uses a uniform-hand prior, so its absolute level is low (play-aware prior + TopK waits are future work).",
            "backend=heuristic_fallback uses public-only B_phi skeleton; local_qwen or SFT B_phi is needed for paper claims.",
            "Rule bots provide limited opponent-view diversity; LLM course opponents are still needed.",
            "Single experiment seed family; repeat with several --seed values for cross-seed robustness.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "h1_summary.json", summary)
    write_jsonl(args.output_dir / "h1_samples_all.jsonl", rows)
    write_jsonl(args.output_dir / "h1_samples_eval.jsonl", eval_rows)
    if balanced_ok:
        write_jsonl(args.output_dir / "h1_samples_balanced_eval.jsonl", balanced_eval)
    return summary


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--opponent-style", default="mixed", choices=list(OPPONENT_STYLES))
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--oracle-samples", type=int, default=60,
                        help="Monte-Carlo samples for the opponent-view posterior oracle.")
    parser.add_argument("--oracle-beta", type=float, default=2.0,
                        help="Play-aware importance weight exp(-beta*shanten) for the MC posterior.")
    parser.add_argument("--danger-threshold", type=int, default=0,
                        help="Positive class = shanten<=this. 0=precise tenpai; 1='danger' (tenpai-or-one-away).")
    parser.add_argument("--b-phi-source", default="llm", choices=["llm", "mc"],
                        help="B_phi estimator: 'llm' = trained LLM output; 'mc' = the MC public-info posterior directly (no LLM, CPU).")
    parser.add_argument("--auc-threshold", type=float, default=0.75,
                        help="Discrimination floor for B_phi (spec: >= 0.75).")
    parser.add_argument("--brier-reduction", type=float, default=0.20,
                        help="Required relative Brier reduction of B_phi vs B0 and B1 (spec parallel of the 0.20 BSE bar).")
    parser.add_argument("--p-threshold", type=float, default=0.05,
                        help="Paired sign-test significance threshold (spec: p < 0.05).")
    parser.add_argument("--backend", default="heuristic_fallback", choices=["heuristic_fallback", "local_qwen"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved H1 outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()

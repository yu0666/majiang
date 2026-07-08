# import contextlib
# import json
# import os
# import re
# import sys
# import time
# import urllib.error
# import urllib.request

# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# from game import (
#     MahjongGame,
#     bot_decide_exchange,
#     bot_decide_missing_suit,
#     bot_decide_response,
#     bot_decide_turn_action,
#     parse_console_tile,
# )

# try:
#     from openai import OpenAI

#     HAS_OPENAI_SDK = True
# except ImportError:
#     HAS_OPENAI_SDK = False

# try:
#     from local_llm_agent import LocalLLMAgent

#     HAS_LOCAL_LLM = True
# except ImportError:
#     HAS_LOCAL_LLM = False
#     print("Warning: local_llm_agent.py not found, local model evaluation is unavailable.")


# # ================= Evaluation Config =================
# NUM_EPISODES = 500
# NUM_RUNS = 1
# INITIAL_BALANCE = 10000
# LLM_PLAYER_ID = 0
# VERBOSE = False
# ENABLE_XAI = False
# EVAL_MODE = "four_model_table"
# MODEL_TAG = "qwen-grpo"
# COMPARISON_MODEL_TAGS = ["qwen-grpo", "gpt-5.4", "claude-opus-4-6", "gemini-3.1-pro-preview"]
# TABLE_MODEL_TAGS = ["qwen-grpo", "gpt-5.4", "claude-opus-4-6", "gemini-3.1-pro-preview"]
# RESULTS_DIR = "eval_results"
# INCLUDE_BOT_AVG_IN_COMPARISON = True
# PREFLIGHT_CLOUD_MODELS = True
# FAIL_ON_MODEL_ERROR = True
# CLOUD_REQUEST_TIMEOUT = 45
# CLOUD_REQUEST_RETRIES = 3
# CLOUD_RETRY_SLEEP_SECONDS = 2
# # ====================================================


# MODEL_DISPLAY_NAMES = {
#     "qwen": "Qwen",
#     "qwen-sft-bot": "Qwen-SFT-Bot",
#     "qwen-sft-selfplay": "Qwen-SFT-SelfPlay",
#     "qwen-grpo": "Qwen-GRPO",
#     "gpt-5.4": "GPT-5.4",
#     "claude-opus-4-6": "Claude-Opus-4.6",
#     "gemini-3.1-pro-preview": "Gemini-3.1-Pro-Preview",
# }

# MODEL_COLORS = {
#     "qwen": "#666666",
#     "qwen-sft-bot": "#FFB347",
#     "qwen-sft-selfplay": "#FF8844",
#     "qwen-grpo": "#FF4444",
#     "gpt-5.4": "#2A7FFF",
#     "claude-opus-4-6": "#32A852",
#     "gemini-3.1-pro-preview": "#F5A623",
# }

# MODEL_CONFIGS = {
#     "qwen-grpo": {
#         "provider": "local",
#         "display_name": "Qwen-GRPO",
#     },
#     "gpt-5.4": {
#         "provider": "openai",
#         "display_name": "GPT-5.4",
#         "base_url": os.getenv("OPENAI_BASE_URL", "https://sub2api.eaislab.com/v1"),
#         "api_key": os.getenv(
#             "OPENAI_API_KEY",
#             "",
#         ),
#         "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-5.4"),
#     },
#     "claude-opus-4-6": {
#         "provider": "anthropic",
#         "display_name": "Claude-Opus-4.6",
#         "base_url": os.getenv(
#             "CLAUDE_BASE_URL",
#             os.getenv("GEMINI_BASE_URL", "https://sub2api.eaislab.com/antigravity"),
#         ),
#         "api_key": os.getenv(
#             "CLAUDE_API_KEY",
#             os.getenv(
#                 "ANTHROPIC_AUTH_TOKEN",
#                 os.getenv(
#                     "GEMINI_API_KEY",
#                     os.getenv(
#                         "GEMINI_ANTHROPIC_AUTH_TOKEN",
#                         os.getenv(
#                             "ANTHROPIC_AUTH_TOKEN_GEMINI",
#                             "",
#                         ),
#                     ),
#                 ),
#             ),
#         ),
#         "model_name": os.getenv("CLAUDE_MODEL_NAME", "claude-opus-4-6"),
#     },
#     "gemini-3.1-pro-preview": {
#         "provider": "anthropic",
#         "display_name": "Gemini-3.1-Pro-Preview",
#         "base_url": os.getenv("GEMINI_BASE_URL", "https://sub2api.eaislab.com/antigravity"),
#         "api_key": os.getenv(
#             "GEMINI_API_KEY",
#             os.getenv(
#                 "GEMINI_ANTHROPIC_AUTH_TOKEN",
#                 os.getenv(
#                     "ANTHROPIC_AUTH_TOKEN_GEMINI",
#                     "",
#                 ),
#             ),
#         ),
#         "model_name": os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-pro-preview"),
#     },
# }



# def mask_secret(secret: str) -> str:
#     if not secret:
#         return "<empty>"
#     if len(secret) <= 8:
#         return "*" * len(secret)
#     return f"{secret[:6]}...{secret[-4:]}"


# def format_debug_context(prefix: str, **kwargs) -> str:
#     parts = [prefix]
#     for key, value in kwargs.items():
#         parts.append(f"{key}={value}")
#     return " | ".join(parts)


# class ModelRequestError(RuntimeError):
#     pass


# TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


# def is_transient_exception(exc) -> bool:
#     status_code = getattr(exc, "status_code", None)
#     if status_code in TRANSIENT_HTTP_STATUS_CODES:
#         return True

#     if isinstance(exc, urllib.error.URLError):
#         return True

#     text = repr(exc).lower()
#     return "timeout" in text or "timed out" in text or "connection" in text


# def retry_sleep(attempt: int):
#     time.sleep(CLOUD_RETRY_SLEEP_SECONDS * attempt)


# CLOUD_SYSTEM_PROMPT = (
#     "You choose moves in a Sichuan Mahjong rules simulation. "
#     "Return exactly one legal action, with no explanation."
# )

# REQUEST_HEADERS = {
#     "User-Agent": "Claude Code",
#     "accept": "application/json",
#     "Claude-Code-Disable-Nonessential-Traffic": "1",
#     "Claude-Code-Attribution-Header": "0",
# }


# def get_http_response_text(response) -> str:
#     if response is None:
#         return "<none>"
#     text = getattr(response, "text", None)
#     if isinstance(text, str) and text:
#         return text[:2000]
#     content = getattr(response, "content", None)
#     if isinstance(content, bytes):
#         return content[:2000].decode("utf-8", errors="ignore")
#     return repr(response)


# def normalize_openai_base_url(base_url: str) -> str:
#     clean = base_url.rstrip("/")
#     if clean.endswith("/v1"):
#         return clean
#     return f"{clean}/v1"


# def tile_to_ascii(tile) -> str:
#     if tile is None:
#         return "NA"
#     suit_map = {
#         "WAN": "W",
#         "TIAO": "T",
#         "TONG": "D",
#     }
#     return f"{suit_map.get(tile.suit.name, tile.suit.name[:1])}{tile.number}"


# def action_to_ascii(action: str) -> str:
#     if action in ("h", "g", "n"):
#         return action
#     if action.startswith("d "):
#         tile = parse_console_tile(action[2:])
#         if tile:
#             return f"d {tile_to_ascii(tile)}"
#     return action


# def suit_to_ascii(suit) -> str:
#     if suit is None:
#         return "unknown"
#     suit_map = {
#         "WAN": "W",
#         "TIAO": "T",
#         "TONG": "D",
#     }
#     return suit_map.get(suit.name, suit.name)


# def build_ascii_history(game: MahjongGame, k: int = 15) -> str:
#     if not getattr(game, "history", None):
#         return "none"

#     lines = []
#     for item in game.history[-k:]:
#         pid = item.get("pid", "?")
#         act = item.get("act", "")
#         tile_text = item.get("tile", "")
#         tile = parse_console_tile(tile_text) if tile_text else None
#         tile_ascii = tile_to_ascii(tile) if tile else "NA"
#         desc = item.get("desc", "")
#         if act == "discard":
#             lines.append(f"P{pid} discard {tile_ascii}")
#         elif act == "peng":
#             lines.append(f"P{pid} peng {tile_ascii}")
#         elif act == "gang":
#             lines.append(f"P{pid} gang {tile_ascii} {desc}".strip())
#         elif act == "hu":
#             lines.append(f"P{pid} hu {tile_ascii} {desc}".strip())
#     return "\n".join(lines) if lines else "none"


# def build_ascii_risk_context(game: MahjongGame, my_pid: int) -> str:
#     risks = []
#     for p in game.players:
#         if p.player_id == my_pid:
#             continue
#         if len(p.open_melds) >= 3:
#             risk = "very_high"
#         elif len(p.open_melds) == 2:
#             risk = "medium"
#         else:
#             risk = "low"
#         last_discard_missing = (
#             bool(p.discarded_tiles)
#             and p.missing_suit is not None
#             and p.discarded_tiles[-1].suit == p.missing_suit
#         )
#         note = "clearing_missing" if last_discard_missing else "observe"
#         risks.append(f"P{p.player_id}:{risk}:{note}")
#     return " | ".join(risks)


# class OpenAICompatibleAgent:
#     def __init__(self, api_key: str, base_url: str, model_name: str):
#         if not HAS_OPENAI_SDK:
#             raise ImportError("openai package is not installed.")
#         if not api_key:
#             raise ValueError("Missing API key for OpenAI-compatible model.")
#         self.api_key = api_key
#         self.base_url = normalize_openai_base_url(base_url)
#         self.client = OpenAI(
#             api_key=api_key,
#             base_url=self.base_url,
#             default_headers=REQUEST_HEADERS,
#         )
#         self.model_name = model_name

#     def decide(self, prompt: str) -> str:
#         last_exc = None
#         for attempt in range(1, CLOUD_REQUEST_RETRIES + 1):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": CLOUD_SYSTEM_PROMPT,
#                         },
#                         {"role": "user", "content": prompt},
#                     ],
#                     temperature=0.1,
#                     max_tokens=20,
#                     timeout=float(CLOUD_REQUEST_TIMEOUT),
#                 )
#                 return response.choices[0].message.content.strip()
#             except Exception as exc:
#                 last_exc = exc
#                 if attempt < CLOUD_REQUEST_RETRIES and is_transient_exception(exc):
#                     print(f"[Retry] {self.model_name} request failed on attempt {attempt}, retrying -> {exc}")
#                     retry_sleep(attempt)
#                     continue

#                 status_code = getattr(exc, "status_code", None)
#                 response_body = getattr(exc, "response", None)
#                 body_text = response_body if isinstance(response_body, str) else get_http_response_text(response_body)
#                 debug_info = format_debug_context(
#                     "OpenAI-compatible request failed",
#                     model=self.model_name,
#                     base_url=self.base_url,
#                     api_key=mask_secret(self.api_key),
#                     prompt_chars=len(prompt),
#                     status_code=status_code,
#                     attempts=attempt,
#                     error=repr(exc),
#                     response=body_text,
#                 )
#                 raise ModelRequestError(debug_info) from exc

#         raise ModelRequestError(f"OpenAI-compatible request failed after retries: {last_exc!r}")


# class AnthropicCompatibleAgent:
#     def __init__(self, api_key: str, base_url: str, model_name: str):
#         if not api_key:
#             raise ValueError("Missing API key for Anthropic-compatible model.")
#         self.api_key = api_key
#         self.base_url = base_url.rstrip("/")
#         self.model_name = model_name

#     def decide(self, prompt: str) -> str:
#         payload = {
#             "model": self.model_name,
#             "max_tokens": 20,
#             "temperature": 0.1,
#             "system": CLOUD_SYSTEM_PROMPT,
#             "messages": [{"role": "user", "content": prompt}],
#         }
#         request_url = f"{self.base_url}/v1/messages"
#         headers = {
#             **REQUEST_HEADERS,
#             "content-type": "application/json",
#             "x-api-key": self.api_key,
#             "authorization": f"Bearer {self.api_key}",
#             "anthropic-version": "2023-06-01",
#         }
#         request = urllib.request.Request(
#             url=request_url,
#             data=json.dumps(payload).encode("utf-8"),
#             headers=headers,
#             method="POST",
#         )
#         data = None
#         for attempt in range(1, CLOUD_REQUEST_RETRIES + 1):
#             try:
#                 with urllib.request.urlopen(request, timeout=CLOUD_REQUEST_TIMEOUT) as response:
#                     data = json.loads(response.read().decode("utf-8"))
#                 break
#             except urllib.error.HTTPError as exc:
#                 detail = exc.read().decode("utf-8", errors="ignore")
#                 if attempt < CLOUD_REQUEST_RETRIES and exc.code in TRANSIENT_HTTP_STATUS_CODES:
#                     print(
#                         f"[Retry] {self.model_name} HTTP {exc.code} on attempt {attempt}, retrying -> {detail[:200]}"
#                     )
#                     retry_sleep(attempt)
#                     continue
#                 debug_info = format_debug_context(
#                     "Anthropic-compatible HTTPError",
#                     model=self.model_name,
#                     url=request_url,
#                     api_key=mask_secret(self.api_key),
#                     prompt_chars=len(prompt),
#                     status_code=exc.code,
#                     attempts=attempt,
#                     reason=exc.reason,
#                     response_body=detail,
#                 )
#                 raise ModelRequestError(debug_info) from exc
#             except urllib.error.URLError as exc:
#                 if attempt < CLOUD_REQUEST_RETRIES:
#                     print(f"[Retry] {self.model_name} URL error on attempt {attempt}, retrying -> {exc.reason!r}")
#                     retry_sleep(attempt)
#                     continue
#                 debug_info = format_debug_context(
#                     "Anthropic-compatible URLError",
#                     model=self.model_name,
#                     url=request_url,
#                     api_key=mask_secret(self.api_key),
#                     prompt_chars=len(prompt),
#                     attempts=attempt,
#                     reason=repr(exc.reason),
#                 )
#                 raise ModelRequestError(debug_info) from exc
#             except Exception as exc:
#                 if attempt < CLOUD_REQUEST_RETRIES and is_transient_exception(exc):
#                     print(f"[Retry] {self.model_name} unexpected error on attempt {attempt}, retrying -> {exc!r}")
#                     retry_sleep(attempt)
#                     continue
#                 debug_info = format_debug_context(
#                     "Anthropic-compatible unexpected error",
#                     model=self.model_name,
#                     url=request_url,
#                     api_key=mask_secret(self.api_key),
#                     prompt_chars=len(prompt),
#                     attempts=attempt,
#                     error=repr(exc),
#                 )
#                 raise ModelRequestError(debug_info) from exc

#         parts = data.get("content", [])
#         texts = [item.get("text", "") for item in parts if item.get("type") == "text"]
#         return "".join(texts).strip()


# def get_agent_debug_name(agent):
#     if agent is None:
#         return "BotFallback"
#     if hasattr(agent, "model_name"):
#         return getattr(agent, "model_name")
#     return agent.__class__.__name__


# def get_model_display_name(model_tag=None):
#     if model_tag is None:
#         model_tag = MODEL_TAG
#     return MODEL_DISPLAY_NAMES.get(model_tag, model_tag)


# def get_model_config(model_tag=None):
#     if model_tag is None:
#         model_tag = MODEL_TAG
#     if model_tag in MODEL_CONFIGS:
#         return MODEL_CONFIGS[model_tag]

#     if model_tag.startswith("claude-"):
#         return {
#             "provider": "anthropic",
#             "display_name": get_model_display_name(model_tag),
#             "base_url": os.getenv(
#                 "CLAUDE_BASE_URL",
#                 os.getenv("GEMINI_BASE_URL", "https://sub2api.eaislab.com/antigravity"),
#             ),
#             "api_key": os.getenv(
#                 "CLAUDE_API_KEY",
#                 os.getenv(
#                     "ANTHROPIC_AUTH_TOKEN",
#                     os.getenv(
#                         "GEMINI_API_KEY",
#                         os.getenv(
#                             "GEMINI_ANTHROPIC_AUTH_TOKEN",
#                             os.getenv("ANTHROPIC_AUTH_TOKEN_GEMINI", ""),
#                         ),
#                     ),
#                 ),
#             ),
#             "model_name": os.getenv("CLAUDE_MODEL_NAME", model_tag),
#         }

#     if model_tag.startswith("gemini-"):
#         return {
#             "provider": "anthropic",
#             "display_name": get_model_display_name(model_tag),
#             "base_url": os.getenv("GEMINI_BASE_URL", "https://sub2api.eaislab.com/antigravity"),
#             "api_key": os.getenv(
#                 "GEMINI_API_KEY",
#                 os.getenv("GEMINI_ANTHROPIC_AUTH_TOKEN", os.getenv("ANTHROPIC_AUTH_TOKEN_GEMINI", "")),
#             ),
#             "model_name": os.getenv("GEMINI_MODEL_NAME", model_tag),
#         }

#     raise KeyError(f"Unsupported MODEL_TAG: {model_tag}")


# def get_risk_analysis(game, my_pid):
#     risks = []
#     for p in game.players:
#         if p.player_id == my_pid:
#             continue

#         risk_level = "安全"
#         note = "观察"

#         if len(p.open_melds) >= 3:
#             risk_level = "极高"
#             note = "可能单吊/清一色"
#         elif len(p.open_melds) == 2:
#             risk_level = "中等"

#         if p.discarded_tiles and p.discarded_tiles[-1].suit == p.missing_suit:
#             note += ", 正在清缺"

#         risks.append(f"P{p.player_id}({risk_level}): {note}")

#     return " | ".join(risks)


# def build_observation_prompt(game: MahjongGame, player_id: int, valid_actions: list = None) -> str:
#     player = game.players[player_id]
#     history_raw = game.get_history_text(k=15)
#     risk_context = get_risk_analysis(game, player_id)
#     hand_str = " ".join([str(t) for t in player.hand_tiles])
#     melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
#     missing = player.missing_suit.value if player.missing_suit else "未定"
#     tiles_left = game.deck.remaining_count()
#     valid_str = ", ".join(valid_actions) if valid_actions else "无限制"

#     prompt = f"""
# 【战局记忆】
# {history_raw}

# 【局势分析】
# 剩余牌数: {tiles_left}
# 对手状态: {risk_context}

# 【当前视角】
# 我是 P{player_id}
# 我的定缺: {missing}
# 我的副露: {melds_str}
# 我的手牌: {hand_str}

# 【决策空间】
# 合法动作: {valid_str}

# 基于以上信息，为了最快胡牌，请给出最佳决策（只输出动作指令）：
# """
#     return prompt.strip()


# def build_cloud_observation_prompt(game: MahjongGame, player_id: int, valid_actions: list = None) -> str:
#     player = game.players[player_id]
#     history_raw = build_ascii_history(game, k=15)
#     risk_context = build_ascii_risk_context(game, player_id)
#     hand_str = " ".join(tile_to_ascii(t) for t in player.hand_tiles)
#     melds_str = " ".join([f"[{tile_to_ascii(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "none"
#     missing = suit_to_ascii(player.missing_suit)
#     tiles_left = game.deck.remaining_count()
#     valid_str = ", ".join(action_to_ascii(action) for action in valid_actions) if valid_actions else "none"

#     prompt = f"""
# Game: Sichuan Mahjong rules simulation.
# History:
# {history_raw}

# State:
# player=P{player_id}
# tiles_left={tiles_left}
# opponents={risk_context}
# missing_suit={missing}
# melds={melds_str}
# hand={hand_str}

# Legal actions:
# {valid_str}

# Choose one legal action only. Use exactly one of the legal action strings above.
# """
#     return prompt.strip()


# @contextlib.contextmanager
# def suppress_stdout():
#     if VERBOSE:
#         yield
#     else:
#         with open(os.devnull, "w") as devnull:
#             old_stdout = sys.stdout
#             sys.stdout = devnull
#             try:
#                 yield
#             finally:
#                 sys.stdout = old_stdout


# def print_game_snapshot(game, current_pid, drawn_tile=None):
#     if not VERBOSE:
#         return

#     print("\n" + "-" * 30 + f" 剩余牌墙: {game.deck.remaining_count()} " + "-" * 30)
#     for p in game.players:
#         marker = ">>" if p.player_id == current_pid else "  "
#         hu_mark = "[已胡]" if p.is_hu else ""
#         hand_str = " ".join([str(t) for t in p.hand_tiles])
#         if p.player_id == current_pid and drawn_tile and not p.is_hu:
#             hand_str += f" + 摸[{drawn_tile}]"

#         meld_str = ""
#         if p.open_melds:
#             meld_str = " | 副露: " + " ".join([f"[{str(m[0])}x{len(m)}]" for m in p.open_melds])

#         missing = p.missing_suit.value if p.missing_suit else "None"
#         print(f"{marker} P{p.player_id} [{p.name}]: {hand_str}{meld_str} | 缺: {missing} {hu_mark}")
#     print("-" * 75)


# def print_detailed_settlement(game: MahjongGame, start_balances: list):
#     if not VERBOSE:
#         return

#     print("\n" + "=" * 28 + " 本局详细结算 " + "=" * 28)
#     for i, p in enumerate(game.players):
#         net_score = p.balance - start_balances[i]
#         score_str = f"+{net_score}" if net_score >= 0 else f"{net_score}"
#         role = "[BOT]" if p.is_bot else "[AI]"
#         status = "胡牌" if p.is_hu else "未胡"
#         print(f"P{p.player_id} {role} {status} {score_str} 番数={p.hu_fan}")
#     print("-" * 76)
#     print("当前余额: ", end="")
#     for p in game.players:
#         print(f"P{p.player_id}:{p.balance}  ", end="")
#     print("\n" + "=" * 76 + "\n")


# def initialize_model_agent(model_tag=None):
#     model_config = get_model_config(model_tag)
#     provider = model_config["provider"]

#     if provider == "local":
#         if not HAS_LOCAL_LLM:
#             raise RuntimeError("LocalLLMAgent is unavailable.")
#         print("Initializing local LLM agent...")
#         # return LocalLLMAgent(enable_xai=ENABLE_XAI and VERBOSE)
#         return LocalLLMAgent()

#     if provider == "openai":
#         print(f"Initializing cloud model: {model_config['display_name']}...")
#         return OpenAICompatibleAgent(
#             api_key=model_config["api_key"],
#             base_url=model_config["base_url"],
#             model_name=model_config["model_name"],
#         )

#     if provider == "anthropic":
#         print(f"Initializing cloud model: {model_config['display_name']}...")
#         return AnthropicCompatibleAgent(
#             api_key=model_config["api_key"],
#             base_url=model_config["base_url"],
#             model_name=model_config["model_name"],
#         )

#     raise ValueError(f"Unsupported provider: {provider}")


# def initialize_table_agents(model_tags=None):
#     if model_tags is None:
#         model_tags = TABLE_MODEL_TAGS

#     agents = {}
#     player_names = []
#     for pid, tag in enumerate(model_tags):
#         print(f"Initializing seat P{pid}: {get_model_display_name(tag)}")
#         agents[pid] = initialize_model_agent(tag)
#         player_names.append(get_model_display_name(tag))
#         print(f"Seat P{pid} ready -> {get_agent_debug_name(agents[pid])}")
#     return agents, player_names


# def build_preflight_prompt() -> str:
#     return """
# Health check for a game agent.
# Legal actions:
# n

# Return exactly: n
# """.strip()


# def preflight_table_agents(agents, model_tags):
#     if not PREFLIGHT_CLOUD_MODELS:
#         return

#     failures = []
#     print("\nPreflight checking cloud model availability...")
#     for pid, tag in enumerate(model_tags):
#         model_config = get_model_config(tag)
#         if model_config["provider"] == "local":
#             print(f"   - P{pid} {get_model_display_name(tag)}: local model, skipped")
#             continue

#         agent = agents[pid]
#         try:
#             raw = decide_with_llm(agent, build_preflight_prompt())
#             choice = normalize_response(raw, ["n"])
#             if choice != "n":
#                 raise ModelRequestError(f"Unexpected preflight response: {raw!r}")
#             print(f"   - P{pid} {get_model_display_name(tag)}: OK")
#         except Exception as exc:
#             failures.append(f"P{pid} {get_model_display_name(tag)} -> {exc}")
#             print(f"   - P{pid} {get_model_display_name(tag)}: FAILED -> {exc}")

#     if failures:
#         message = "\n".join(failures)
#         raise RuntimeError(
#             "Cloud model preflight failed. Evaluation stopped to avoid bot-fallback-contaminated results.\n"
#             f"{message}\n\n"
#             "Fix the corresponding key/base_url/account group, or set PREFLIGHT_CLOUD_MODELS=False "
#             "and FAIL_ON_MODEL_ERROR=False if you explicitly want bot fallback."
#         )


# def build_valid_actions(player, game, player_id):
#     valid_actions = []
#     if player.can_hu():
#         valid_actions.append("h")

#     gang_info = game.can_self_gang(player_id)
#     if gang_info["can_gang"]:
#         valid_actions.append("g")

#     has_missing = any(t.suit == player.missing_suit for t in player.hand_tiles)
#     seen_discard = set()
#     for t in player.hand_tiles:
#         if has_missing and t.suit != player.missing_suit:
#             continue
#         t_str = str(t)
#         if t_str not in seen_discard:
#             valid_actions.append(f"d {t_str}")
#             seen_discard.add(t_str)

#     return valid_actions


# def decide_with_llm(llm_agent, prompt):
#     if VERBOSE:
#         return llm_agent.decide(prompt)
#     with suppress_stdout():
#         return llm_agent.decide(prompt)


# def is_cloud_agent(agent) -> bool:
#     return isinstance(agent, (OpenAICompatibleAgent, AnthropicCompatibleAgent))


# def build_prompt_for_agent(agent, game, player_id, valid_actions):
#     if is_cloud_agent(agent):
#         return build_cloud_observation_prompt(game, player_id, valid_actions)
#     return build_observation_prompt(game, player_id, valid_actions)


# def normalize_action(action, valid_actions):
#     action = (action or "").strip()
#     if action in valid_actions:
#         return action

#     normalized = re.sub(r"\s+", " ", action.lower())
#     for valid_action in valid_actions:
#         ascii_action = action_to_ascii(valid_action).lower()
#         if normalized == ascii_action:
#             return valid_action
#         if re.search(rf"(^|[^a-z0-9]){re.escape(ascii_action)}([^a-z0-9]|$)", normalized):
#             return valid_action

#     for valid_action in valid_actions:
#         if valid_action in action:
#             return valid_action

#     if valid_actions:
#         return valid_actions[0]
#     return "n"


# def normalize_response(choice, valid_resps):
#     choice = (choice or "").strip()
#     if choice in valid_resps:
#         return choice

#     if "h" in choice and "h" in valid_resps:
#         return "h"
#     if "p" in choice and "p" in valid_resps:
#         return "p"
#     if "g" in choice and "g" in valid_resps:
#         return "g"
#     return "n"


# def create_stats():
#     return {
#         "hu_count": [0] * 4,
#         "hu_fan_ge_1": [0] * 4,
#         "dianpao_count": [0] * 4,
#         "total_fan": [0] * 4,
#     }


# def run_single_evaluation(llm_agent, run_idx: int, eval_mode=None, llm_agents=None, player_names=None):
#     if eval_mode is None:
#         eval_mode = EVAL_MODE

#     if eval_mode == "four_model_table":
#         if llm_agents is None:
#             raise ValueError("llm_agents is required in four_model_table mode.")
#         if player_names is None:
#             player_names = [get_model_display_name(tag) for tag in TABLE_MODEL_TAGS]
#         bots_config = [False, False, False, False]
#     else:
#         player_names = [get_model_display_name(), "Bot-1", "Bot-2", "Bot-3"]
#         bots_config = [False, True, True, True]

#     global_balances = [INITIAL_BALANCE] * 4
#     balance_history = [[INITIAL_BALANCE] * 4]
#     stats = create_stats()

#     iterator = range(NUM_EPISODES) if VERBOSE else tqdm(
#         range(NUM_EPISODES),
#         desc=f"对战进度 Run {run_idx}/{NUM_RUNS}",
#     )

#     for ep in iterator:
#         if VERBOSE:
#             print(f"\n>>>>>> 第 {ep + 1} 局开始 <<<<<<")

#         game = MahjongGame(f"EVAL_{MODEL_TAG}_{run_idx}_{ep}", player_names, bots=bots_config)
#         for i, p in enumerate(game.players):
#             p.balance = global_balances[i]

#         round_start_balances = [p.balance for p in game.players]
#         game.start_game()

#         game.phase = game.phase.EXCHANGE
#         for p in game.players:
#             game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))

#         game.phase = game.phase.CHOOSE_MISSING
#         for p in game.players:
#             game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))

#         game.phase = game.phase.PLAYING
#         skip_draw = True
#         game_step_count = 0

#         while not game.is_game_over:
#             game_step_count += 1
#             if game_step_count > 300 or sum(1 for p in game.players if p.is_hu) >= 3:
#                 game.is_game_over = True
#                 break

#             pid = game.current_player_id
#             player = game.players[pid]

#             if player.is_hu:
#                 game.next_player()
#                 skip_draw = False
#                 continue

#             drawn = None
#             if not skip_draw:
#                 drawn = game.draw_tile(pid)
#                 if not drawn:
#                     game.check_game_over()
#                     break
#             else:
#                 skip_draw = False

#             print_game_snapshot(game, pid, drawn)

#             turn_end = False
#             loop_attempts = 0

#             while not turn_end:
#                 loop_attempts += 1
#                 action = ""
#                 force_bot = loop_attempts > 3

#                 active_agent = None
#                 if eval_mode == "four_model_table":
#                     active_agent = llm_agents.get(pid) if llm_agents is not None else None
#                 elif pid == LLM_PLAYER_ID:
#                     active_agent = llm_agent

#                 if VERBOSE:
#                     route_name = get_agent_debug_name(active_agent) if not force_bot else "BotFallback(force_bot)"
#                     print(f"[TurnRoute] P{pid} -> {route_name}")

#                 if active_agent is not None and not force_bot:
#                     try:
#                         valid_actions = build_valid_actions(player, game, pid)
#                         prompt = build_prompt_for_agent(active_agent, game, pid, valid_actions)
#                         action = decide_with_llm(active_agent, prompt)
#                         action = normalize_action(action, valid_actions)
#                         if VERBOSE:
#                             print(f"[TurnAction] P{pid} -> {action}")
#                     except Exception as e:
#                         print(f"[TurnFallback] P{pid} model failed -> {e}")
#                         if FAIL_ON_MODEL_ERROR:
#                             raise RuntimeError(
#                                 f"Model call failed at turn stage for P{pid}; "
#                                 "evaluation stopped to avoid using bot fallback as model output."
#                             ) from e
#                         action = bot_decide_turn_action(player, game)
#                         if VERBOSE:
#                             print(f"[TurnAction] P{pid} -> {action} (bot fallback)")
#                 else:
#                     action = bot_decide_turn_action(player, game)
#                     if VERBOSE:
#                         print(f"[TurnAction] P{pid} -> {action} (bot)")

#                 if action == "h":
#                     if player.can_hu():
#                         win_card = drawn if drawn else player.hand_tiles[-1]
#                         game.hu(pid, win_card, True)
#                         game.check_game_over()
#                         if game.is_game_over:
#                             turn_end = True
#                             break
#                         turn_end = True
#                         game.next_player()
#                         skip_draw = False
#                 elif action == "g":
#                     g_info = game.can_self_gang(pid)
#                     if g_info["can_gang"]:
#                         game.gang(pid, g_info["gang_tiles"][0])
#                         continue
#                 elif action.startswith("d "):
#                     t = parse_console_tile(action[2:])
#                     if t and game.discard_tile(pid, t):
#                         responses = game.check_responses(t, pid)
#                         someone_responded = False

#                         if responses:
#                             for r_id, acts in responses.items():
#                                 if someone_responded:
#                                     break

#                                 responder = game.players[r_id]
#                                 choice = "n"

#                                 response_agent = None
#                                 if eval_mode == "four_model_table":
#                                     response_agent = llm_agents.get(r_id) if llm_agents is not None else None
#                                 elif r_id == LLM_PLAYER_ID:
#                                     response_agent = llm_agent

#                                 if response_agent is not None:
#                                     if VERBOSE:
#                                         print(f"[RespRoute] P{r_id} -> {get_agent_debug_name(response_agent)}")
#                                     try:
#                                         valid_resps = ["n"]
#                                         if "hu" in acts:
#                                             valid_resps.append("h")
#                                         if "gang" in acts:
#                                             valid_resps.append("g")
#                                         if "peng" in acts:
#                                             valid_resps.append("p")

#                                         prompt = build_prompt_for_agent(response_agent, game, r_id, valid_resps)
#                                         if is_cloud_agent(response_agent):
#                                             prompt += f"\nEvent: P{pid} discarded {tile_to_ascii(t)}. Choose a response action."
#                                         else:
#                                             prompt += f"\n【突发事件】\n对手 P{pid} 打出了 {t}，触发响应机会。"
#                                         choice = decide_with_llm(response_agent, prompt)
#                                         choice = normalize_response(choice, valid_resps)
#                                         if VERBOSE:
#                                             print(f"[RespAction] P{r_id} -> {choice}")
#                                     except Exception as e:
#                                         print(f"[RespFallback] P{r_id} model failed -> {e}")
#                                         if FAIL_ON_MODEL_ERROR:
#                                             raise RuntimeError(
#                                                 f"Model call failed at response stage for P{r_id}; "
#                                                 "evaluation stopped to avoid using bot fallback as model output."
#                                             ) from e
#                                         choice = "n"
#                                 else:
#                                     choice = bot_decide_response(responder, acts)
#                                     if VERBOSE:
#                                         print(f"[RespAction] P{r_id} -> {choice} (bot)")

#                                 if choice == "h" and "hu" in acts:
#                                     game.hu(r_id, t, False, pid)
#                                     stats["dianpao_count"][pid] += 1
#                                     game.check_game_over()
#                                     someone_responded = True
#                                     if game.is_game_over:
#                                         turn_end = True
#                                         break
#                                 elif choice == "g" and "gang" in acts:
#                                     game.gang(r_id, t, pid)
#                                     game.current_player_id = r_id
#                                     turn_end = True
#                                     someone_responded = True
#                                     skip_draw = True
#                                 elif choice == "p" and "peng" in acts:
#                                     game.peng(r_id, t, pid)
#                                     game.current_player_id = r_id
#                                     turn_end = True
#                                     someone_responded = True
#                                     skip_draw = True

#                         if game.is_game_over:
#                             break

#                         if not someone_responded:
#                             game.next_player()
#                             skip_draw = False
#                         turn_end = True

#         game.check_game_over()
#         print_detailed_settlement(game, round_start_balances)

#         for i, p in enumerate(game.players):
#             global_balances[i] = p.balance
#             if p.is_hu:
#                 stats["hu_count"][i] += 1
#                 stats["total_fan"][i] += p.hu_fan
#                 if p.hu_fan >= 2:
#                     stats["hu_fan_ge_1"][i] += 1

#         balance_history.append(global_balances.copy())

#     print("\n" + "=" * 104)
#     print(f"Run {run_idx} single evaluation result ({NUM_EPISODES} games)")
#     print("=" * 104)
#     print(f"{'ID':<4} {'Role':<18} {'Wins':<8} {'HuRate':<10} {'Win(>=2F)':<12} {'DianPao':<10} {'TotalFan':<10} {'Balance':<12} {'Net':<10}")
#     print("-" * 104)

#     for i in range(4):
#         role = player_names[i]
#         wins = stats["hu_count"][i]
#         hu_rate = (wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
#         win_ge_2f = stats["hu_fan_ge_1"][i]
#         dianpao = stats["dianpao_count"][i]
#         total_fan = stats["total_fan"][i]
#         balance = global_balances[i]
#         net = balance - INITIAL_BALANCE
#         net_str = f"+{net}" if net > 0 else str(net)

#         print(
#             f"{i:<4} {role:<18} {wins:<8} {hu_rate:<9.2f}% {win_ge_2f:<12} "
#             f"{dianpao:<10} {total_fan:<10} {balance:<12} {net_str:<10}"
#         )

#     print("-" * 104)

#     return {
#         "player_names": player_names,
#         "balance_history": balance_history,
#         "stats": stats,
#         "final_balances": global_balances,
#     }


# def summarize_average_results(results):
#     history_stack = np.array([r["balance_history"] for r in results], dtype=float)
#     avg_history = history_stack.mean(axis=0)

#     avg_stats = {}
#     for key in results[0]["stats"].keys():
#         metric_stack = np.array([r["stats"][key] for r in results], dtype=float)
#         avg_stats[key] = metric_stack.mean(axis=0)

#     final_balance_stack = np.array([r["final_balances"] for r in results], dtype=float)
#     avg_final_balances = final_balance_stack.mean(axis=0)

#     return {
#         "player_names": results[0]["player_names"],
#         "avg_history": avg_history,
#         "avg_stats": avg_stats,
#         "avg_final_balances": avg_final_balances,
#     }


# def build_ai_summary(summary):
#     avg_wins = float(summary["avg_stats"]["hu_count"][LLM_PLAYER_ID])
#     avg_hu_rate = (avg_wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
#     avg_win_ge_2f = float(summary["avg_stats"]["hu_fan_ge_1"][LLM_PLAYER_ID])
#     avg_dianpao = float(summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID])
#     avg_total_fan = float(summary["avg_stats"]["total_fan"][LLM_PLAYER_ID])
#     avg_balance = float(summary["avg_final_balances"][LLM_PLAYER_ID])
#     avg_net = avg_balance - INITIAL_BALANCE

#     return {
#         "model_tag": MODEL_TAG,
#         "num_episodes": NUM_EPISODES,
#         "num_runs": NUM_RUNS,
#         "initial_balance": INITIAL_BALANCE,
#         "avg_wins": avg_wins,
#         "avg_hu_rate": avg_hu_rate,
#         "avg_win_ge_2f": avg_win_ge_2f,
#         "avg_dianpao": avg_dianpao,
#         "avg_total_fan": avg_total_fan,
#         "avg_balance": avg_balance,
#         "avg_net": avg_net,
#     }


# def save_summary_results(summary):
#     os.makedirs(RESULTS_DIR, exist_ok=True)

#     npz_path = os.path.join(RESULTS_DIR, f"{MODEL_TAG}_summary.npz")
#     json_path = os.path.join(RESULTS_DIR, f"{MODEL_TAG}_summary.json")

#     np.savez(
#         npz_path,
#         model_tag=MODEL_TAG,
#         player_names=np.array(summary["player_names"], dtype=object),
#         avg_history=np.array(summary["avg_history"], dtype=float),
#         hu_count=np.array(summary["avg_stats"]["hu_count"], dtype=float),
#         hu_fan_ge_1=np.array(summary["avg_stats"]["hu_fan_ge_1"], dtype=float),
#         dianpao_count=np.array(summary["avg_stats"]["dianpao_count"], dtype=float),
#         total_fan=np.array(summary["avg_stats"]["total_fan"], dtype=float),
#         avg_final_balances=np.array(summary["avg_final_balances"], dtype=float),
#         num_episodes=np.array(NUM_EPISODES),
#         num_runs=np.array(NUM_RUNS),
#         initial_balance=np.array(INITIAL_BALANCE),
#     )

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(build_ai_summary(summary), f, ensure_ascii=False, indent=2)

#     print(f"Saved evaluation summary: {npz_path}")
#     print(f"Saved metric summary: {json_path}")


# def load_saved_summary(model_tag):
#     path = os.path.join(RESULTS_DIR, f"{model_tag}_summary.npz")
#     if not os.path.exists(path):
#         return None

#     data = np.load(path, allow_pickle=True)
#     return {
#         "model_tag": model_tag,
#         "raw_model_tag": str(data["model_tag"].item()) if np.ndim(data["model_tag"]) == 0 else str(data["model_tag"]),
#         "player_names": data["player_names"].tolist(),
#         "avg_history": np.array(data["avg_history"], dtype=float),
#         "avg_final_balances": np.array(data["avg_final_balances"], dtype=float),
#         "avg_stats": {
#             "hu_count": np.array(data["hu_count"], dtype=float),
#             "hu_fan_ge_1": np.array(data["hu_fan_ge_1"], dtype=float),
#             "dianpao_count": np.array(data["dianpao_count"], dtype=float),
#             "total_fan": np.array(data["total_fan"], dtype=float),
#         },
#         "num_episodes": int(data["num_episodes"]),
#         "num_runs": int(data["num_runs"]),
#         "initial_balance": float(data["initial_balance"]),
#     }


# def annotate_last_point(x, y, text, color, x_offset=8, y_offset=0):
#     plt.scatter([x], [y], color=color, s=28, zorder=5)
#     plt.annotate(
#         text,
#         xy=(x, y),
#         xytext=(x_offset, y_offset),
#         textcoords="offset points",
#         color=color,
#         fontsize=10,
#         fontweight="bold",
#         va="center",
#     )


# def plot_model_comparison_from_saved(
#     model_tags=None,
#     include_bot_avg=INCLUDE_BOT_AVG_IN_COMPARISON,
#     annotate_last_values=True,
# ):
#     if model_tags is None:
#         model_tags = COMPARISON_MODEL_TAGS

#     summaries = []
#     missing_tags = []

#     for tag in model_tags:
#         loaded = load_saved_summary(tag)
#         if loaded is None:
#             missing_tags.append(tag)
#         else:
#             summaries.append(loaded)

#     if missing_tags:
#         print(f"Comparison chart skipped, missing summary files: {', '.join(missing_tags)}")
#         return

#     base_episodes = summaries[0]["avg_history"].shape[0]
#     if any(s["avg_history"].shape[0] != base_episodes for s in summaries):
#         print("Comparison chart skipped, inconsistent episode count across models.")
#         return

#     if any(s["initial_balance"] != summaries[0]["initial_balance"] for s in summaries):
#         print("Comparison chart skipped, inconsistent initial balance across models.")
#         return

#     episodes = range(base_episodes)
#     plt.figure(figsize=(12, 7))

#     for idx, summary in enumerate(summaries):
#         model_tag = summary["model_tag"]
#         y_values = summary["avg_history"][:, 0]
#         color = MODEL_COLORS.get(model_tag, "#333333")
#         plt.plot(
#             episodes,
#             y_values,
#             label=get_model_display_name(model_tag),
#             linewidth=2.5,
#             color=color,
#         )
#         if annotate_last_values:
#             annotate_last_point(
#                 base_episodes - 1,
#                 y_values[-1],
#                 f"{y_values[-1]:.0f}",
#                 color,
#                 y_offset=(idx - len(summaries) / 2) * 10,
#             )

#     if include_bot_avg:
#         bot_curves = [summary["avg_history"][:, 1:4].mean(axis=1) for summary in summaries]
#         combined_bot_avg = np.mean(np.stack(bot_curves, axis=0), axis=0)
#         plt.plot(
#             episodes,
#             combined_bot_avg,
#             label="Bot Avg",
#             linewidth=2.0,
#             linestyle="--",
#             color="#2A7FFF",
#         )
#         if annotate_last_values:
#             annotate_last_point(
#                 base_episodes - 1,
#                 combined_bot_avg[-1],
#                 f"{combined_bot_avg[-1]:.0f}",
#                 "#2A7FFF",
#                 y_offset=-14,
#             )

#     plt.axhline(
#         y=summaries[0]["initial_balance"],
#         color="gray",
#         linestyle=":",
#         alpha=0.7,
#         label="Initial Balance",
#     )
#     plt.title("Model Comparison on Average Balance")
#     plt.xlabel("Episodes")
#     plt.ylabel("Average Balance")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     filename = os.path.join(RESULTS_DIR, "model_comparison.png")
#     plt.savefig(filename, dpi=150)
#     plt.close()
#     print(f"Saved comparison chart: {filename}")


# def redraw_comparison_from_existing_results():
#     plot_model_comparison_from_saved(
#         model_tags=COMPARISON_MODEL_TAGS,
#         include_bot_avg=True,
#         annotate_last_values=True,
#     )


# def plot_metric_bar_chart_from_saved(model_tags=None):
#     if model_tags is None:
#         model_tags = COMPARISON_MODEL_TAGS

#     summaries = []
#     missing_tags = []

#     for tag in model_tags:
#         loaded = load_saved_summary(tag)
#         if loaded is None:
#             missing_tags.append(tag)
#         else:
#             summaries.append(loaded)

#     if missing_tags:
#         print(f"Metric bar chart skipped, missing summary files: {', '.join(missing_tags)}")
#         return

#     labels = [get_model_display_name(summary["model_tag"]) for summary in summaries]
#     hu_rates = []
#     dianpao_rates = []
#     total_fans = []

#     for summary in summaries:
#         hu_count = float(summary["avg_stats"]["hu_count"][LLM_PLAYER_ID])
#         dianpao_count = float(summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID])
#         total_fan = float(summary["avg_stats"]["total_fan"][LLM_PLAYER_ID])
#         num_episodes = int(summary["num_episodes"])

#         hu_rate = (hu_count / num_episodes) * 100 if num_episodes > 0 else 0.0
#         dianpao_rate = (dianpao_count / num_episodes) * 100 if num_episodes > 0 else 0.0

#         hu_rates.append(hu_rate)
#         dianpao_rates.append(dianpao_rate)
#         total_fans.append(total_fan)

#     colors = [MODEL_COLORS.get(summary["model_tag"], "#333333") for summary in summaries]
#     x = np.arange(len(labels))

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     metric_specs = [
#         ("Hu Rate (%)", hu_rates, "胡牌率对比", "{:.2f}%"),
#         ("DianPao Rate (%)", dianpao_rates, "点炮率对比", "{:.2f}%"),
#         ("Total Fan", total_fans, "累计总番数对比", "{:.2f}"),
#     ]

#     for ax, (ylabel, values, title, fmt) in zip(axes, metric_specs):
#         bars = ax.bar(x, values, color=colors, width=0.62)
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=12)
#         ax.set_ylabel(ylabel)
#         ax.set_title(title)
#         ax.grid(axis="y", alpha=0.3, linestyle="--")

#         upper = max(values) if values else 0.0
#         ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)

#         for bar, value in zip(bars, values):
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height() + (upper * 0.03 if upper > 0 else 0.03),
#                 fmt.format(value),
#                 ha="center",
#                 va="bottom",
#                 fontsize=10,
#                 fontweight="bold",
#             )

#     fig.suptitle("Model Comparison on Core Metrics", fontsize=16, fontweight="bold")
#     plt.tight_layout()

#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     filename = os.path.join(RESULTS_DIR, "model_metrics_bar.png")
#     plt.savefig(filename, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"Saved metric bar chart: {filename}")


# def redraw_metric_bar_chart_from_existing_results():
#     plot_metric_bar_chart_from_saved(model_tags=COMPARISON_MODEL_TAGS)


# def plot_balance_curve(history, player_names, output_name=None):
#     history = np.array(history, dtype=float)
#     episodes = range(len(history))
#     plt.figure(figsize=(12, 7))

#     colors = ["#FF4444", "#2A7FFF", "#32A852", "#F5A623"]
#     styles = ["-", "-", "-", "-"]
#     widths = [2.5, 2.5, 2.5, 2.5]
#     markers = ["o", "s", "^", "D"]

#     for i, name in enumerate(player_names):
#         plt.plot(
#             episodes,
#             history[:, i],
#             label=name,
#             color=colors[i],
#             linewidth=widths[i],
#             linestyle=styles[i],
#             marker=markers[i],
#             markersize=4 if markers[i] else None,
#         )
#         annotate_last_point(
#             len(history) - 1,
#             history[-1, i],
#             f"{history[-1, i]:.0f}",
#             colors[i],
#             y_offset=(i - 1.5) * 10,
#         )

#     plt.axhline(y=INITIAL_BALANCE, color="gray", linestyle=":", alpha=0.5)
#     plt.title(f"Average Evaluation Result ({NUM_RUNS} Runs)")
#     plt.xlabel("Episodes")
#     plt.ylabel("Average Balance")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     if output_name is None:
#         output_name = f"{MODEL_TAG}_avg_balance_curve.png"
#     filename = os.path.join(RESULTS_DIR, output_name)
#     plt.savefig(filename, dpi=150)
#     plt.close()
#     print(f"Saved single-model average balance curve: {filename}")


# def plot_four_model_metric_bar_chart(summary, output_name="four_model_table_core_metrics.png"):
#     player_names = summary["player_names"]
#     colors = ["#FF4444", "#2A7FFF", "#32A852", "#F5A623"]
#     x = np.arange(len(player_names))

#     hu_counts = np.array(summary["avg_stats"]["hu_count"], dtype=float)
#     hu_rates = (hu_counts / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else np.zeros(4)
#     dianpao_counts = np.array(summary["avg_stats"]["dianpao_count"], dtype=float)
#     dianpao_rates = (dianpao_counts / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else np.zeros(4)
#     total_fans = np.array(summary["avg_stats"]["total_fan"], dtype=float)

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     metric_specs = [
#         (axes[0], "Win Rate", "Win Rate (%)", hu_rates, "{:.2f}%"),
#         (axes[1], "Discard Loss Rate", "Rate (%)", dianpao_rates, "{:.2f}%"),
#         (axes[2], "Total Fan", "Fan Count", total_fans, "{:.2f}"),
#     ]

#     for ax, title, ylabel, values, fmt in metric_specs:
#         bars = ax.bar(x, values, color=colors, width=0.62)
#         ax.set_xticks(x)
#         ax.set_xticklabels(player_names, rotation=12)
#         ax.set_ylabel(ylabel)
#         ax.set_title(title, fontsize=13, fontweight="bold")
#         ax.grid(axis="y", alpha=0.3, linestyle="--")

#         upper = max(float(np.max(values)), 0.0)
#         ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)

#         for bar, value in zip(bars, values):
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height() + (upper * 0.025 if upper > 0 else 0.03),
#                 fmt.format(value),
#                 ha="center",
#                 va="bottom",
#                 fontsize=10,
#                 fontweight="bold",
#             )

#     fig.suptitle("Core Metrics Comparison of Four Models", fontsize=16, fontweight="bold")
#     plt.tight_layout()

#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     filename = os.path.join(RESULTS_DIR, output_name)
#     plt.savefig(filename, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"Saved four-model metric chart: {filename}")


# def run_evaluation():
#     print("\nStarting evaluation")
#     print(f"   - Evaluation mode: {EVAL_MODE}")
#     print(f"   - Single-run episodes: {NUM_EPISODES}")
#     print(f"   - Number of runs: {NUM_RUNS}")
#     print(f"   - Verbose: {VERBOSE}")

#     if EVAL_MODE == "four_model_table":
#         print(f"   - Table models: {', '.join(TABLE_MODEL_TAGS)}")
#         llm_agents, player_names = initialize_table_agents(TABLE_MODEL_TAGS)
#         preflight_table_agents(llm_agents, TABLE_MODEL_TAGS)
#         for pid, name in enumerate(player_names):
#             print(f"   - Seat P{pid}: {name}")
#         llm_agent = None
#     else:
#         print(f"   - Current model tag: {MODEL_TAG}")
#         llm_agents = None
#         player_names = None
#         llm_agent = initialize_model_agent(MODEL_TAG)

#     results = []

#     for run_idx in range(1, NUM_RUNS + 1):
#         results.append(
#             run_single_evaluation(
#                 llm_agent,
#                 run_idx,
#                 eval_mode=EVAL_MODE,
#                 llm_agents=llm_agents,
#                 player_names=player_names,
#             )
#         )

#     summary = summarize_average_results(results)

#     print("\n" + "=" * 124)
#     print(f"{NUM_RUNS} independent runs average result ({NUM_RUNS} x {NUM_EPISODES} games)")
#     print("=" * 124)
#     print(f"{'ID':<4} {'Role':<18} {'AvgWins':<10} {'AvgHuRate':<12} {'AvgWin(>=2F)':<14} {'AvgDianPao':<12} {'AvgTotalFan':<14} {'AvgBalance':<14} {'AvgNet':<12}")
#     print("-" * 124)

#     for i in range(4):
#         role = summary["player_names"][i]
#         avg_wins = summary["avg_stats"]["hu_count"][i]
#         avg_hu_rate = (avg_wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
#         avg_win_ge_2f = summary["avg_stats"]["hu_fan_ge_1"][i]
#         avg_dianpao = summary["avg_stats"]["dianpao_count"][i]
#         avg_total_fan = summary["avg_stats"]["total_fan"][i]
#         avg_balance = summary["avg_final_balances"][i]
#         avg_net = avg_balance - INITIAL_BALANCE
#         avg_net_str = f"+{avg_net:.2f}" if avg_net > 0 else f"{avg_net:.2f}"

#         print(
#             f"{i:<4} {role:<18} {avg_wins:<10.2f} {avg_hu_rate:<11.2f}% {avg_win_ge_2f:<14.2f} "
#             f"{avg_dianpao:<12.2f} {avg_total_fan:<14.2f} {avg_balance:<14.2f} {avg_net_str:<12}"
#         )

#     print("-" * 124)

#     if EVAL_MODE == "four_model_table":
#         print("\nFour-model table battle finished.")
#         final_balances = np.array(summary["avg_final_balances"], dtype=float)
#         net_values = final_balances - INITIAL_BALANCE
#         print("\nFour-model final balance summary")
#         for name, balance, net in zip(summary["player_names"], final_balances, net_values):
#             print(f"   - {name}: balance={balance:.2f}, net={net:+.2f}")

#         plot_balance_curve(
#             summary["avg_history"],
#             summary["player_names"],
#             output_name="four_model_table_avg_balance_curve.png",
#         )
#         plot_four_model_metric_bar_chart(summary)
#     else:
#         ai_avg_wins = summary["avg_stats"]["hu_count"][LLM_PLAYER_ID]
#         ai_avg_hu_rate = (ai_avg_wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
#         ai_avg_win_ge_2f = summary["avg_stats"]["hu_fan_ge_1"][LLM_PLAYER_ID]
#         ai_avg_dianpao = summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID]
#         ai_avg_total_fan = summary["avg_stats"]["total_fan"][LLM_PLAYER_ID]
#         ai_avg_balance = summary["avg_final_balances"][LLM_PLAYER_ID]
#         ai_avg_net = ai_avg_balance - INITIAL_BALANCE

#         print("\nAI core average metrics")
#         print(f"   - Average wins: {ai_avg_wins:.2f}")
#         print(f"   - Average hu rate: {ai_avg_hu_rate:.2f}%")
#         print(f"   - Average win count (>=2 fan): {ai_avg_win_ge_2f:.2f}")
#         print(f"   - Average dianpao count: {ai_avg_dianpao:.2f}")
#         print(f"   - Average total fan: {ai_avg_total_fan:.2f}")
#         print(f"   - Average net income: {ai_avg_net:.2f}")

#         plot_balance_curve(summary["avg_history"], summary["player_names"])
#         save_summary_results(summary)
#         plot_model_comparison_from_saved()
#         plot_metric_bar_chart_from_saved()


# if __name__ == "__main__":
#     run_evaluation()
import contextlib
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from game import (
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    bot_decide_turn_action,
    parse_console_tile,
)

try:
    from local_llm_agent import LocalLLMAgent

    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️ Warning: local_llm_agent.py 未找到，将使用 Bot 替代 LLM 进行评估。")


# ================= 🔧 评估配置 =================
NUM_EPISODES = 1000
NUM_RUNS = 3
INITIAL_BALANCE = 100000
LLM_PLAYER_ID = 0
VERBOSE = False
ENABLE_XAI = False
MODEL_TAG = "qwen-grpo"
COMPARISON_MODEL_TAGS = ["qwen", "qwen-sft-bot", "qwen-sft-selfplay", "qwen-grpo"]
RESULTS_DIR = "eval_results"
INCLUDE_BOT_AVG_IN_COMPARISON = True
# ==============================================


MODEL_DISPLAY_NAMES = {
    "qwen": "Qwen",
    "qwen-sft-bot": "Qwen-SFT-Bot",
    "qwen-sft-selfplay": "Qwen-SFT-SelfPlay",
    "qwen-grpo": "Qwen-GRPO",
}


def get_model_display_name(model_tag=None):
    if model_tag is None:
        model_tag = MODEL_TAG
    return MODEL_DISPLAY_NAMES.get(model_tag, model_tag)


def get_risk_analysis(game, my_pid):
    risks = []
    for p in game.players:
        if p.player_id == my_pid:
            continue

        risk_level = "安全"
        note = "观察"

        if len(p.open_melds) >= 3:
            risk_level = "极高"
            note = "可能单钓/清一色"
        elif len(p.open_melds) == 2:
            risk_level = "中等"

        if p.discarded_tiles and p.discarded_tiles[-1].suit == p.missing_suit:
            note += ", 正在清缺"

        risks.append(f"P{p.player_id}({risk_level}): {note}")

    return " | ".join(risks)


def build_observation_prompt(game: MahjongGame, player_id: int, valid_actions: list = None) -> str:
    player = game.players[player_id]
    history_raw = game.get_history_text(k=15)
    risk_context = get_risk_analysis(game, player_id)
    hand_str = " ".join([str(t) for t in player.hand_tiles])
    melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
    missing = player.missing_suit.value if player.missing_suit else "未定"
    tiles_left = game.deck.remaining_count()
    valid_str = ", ".join(valid_actions) if valid_actions else "无限制"

    prompt = f"""
【战局记忆】
{history_raw}

【局势分析】
剩余牌数: {tiles_left}
对手状态: {risk_context}

【当前视角】
我是 P{player_id}
我的定缺: {missing}
我的副露: {melds_str}
我的手牌: {hand_str}

【决策空间】
合法动作: {valid_str}

基于以上信息，为了最快胡牌，请给出最佳决策（只输出动作指令）：
"""
    return prompt.strip()


@contextlib.contextmanager
def suppress_stdout():
    if VERBOSE:
        yield
    else:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def print_game_snapshot(game, current_pid, drawn_tile=None):
    if not VERBOSE:
        return

    print("\n" + "-" * 30 + f" 🀄 剩余牌墙: {game.deck.remaining_count()} " + "-" * 30)
    for p in game.players:
        marker = "👉" if p.player_id == current_pid else "  "
        hu_mark = "🎉[已胡]" if p.is_hu else ""

        hand_str = " ".join([str(t) for t in p.hand_tiles])
        if p.player_id == current_pid and drawn_tile and not p.is_hu:
            hand_str += f" + 摸[{drawn_tile}]"

        meld_str = ""
        if p.open_melds:
            meld_str = " | 副露: " + " ".join([f"[{str(m[0])}x{len(m)}]" for m in p.open_melds])

        missing = p.missing_suit.value if p.missing_suit else "None"
        print(f"{marker} P{p.player_id} [{p.name}]: {hand_str}{meld_str} | 缺: {missing} {hu_mark}")
    print("-" * 75)


def print_detailed_settlement(game: MahjongGame, start_balances: list):
    if not VERBOSE:
        return

    print("\n" + "█" * 30 + " 本局详细结算 " + "█" * 30)

    dian_pao_players = {}
    for p in game.players:
        if p.is_hu and not p.hu_is_self_drawn and p.hu_discard_player_id is not None:
            dian_pao_players[p.player_id] = p.hu_discard_player_id

    for i, p in enumerate(game.players):
        net_score = p.balance - start_balances[i]
        score_str = f"+{net_score}" if net_score >= 0 else f"{net_score}"

        role = "[BOT]" if p.is_bot else "[YOU]"
        status_tags = []
        details = []

        if p.is_hu:
            status_tags.append("【胡牌】")
            if p.hu_is_self_drawn:
                method = "自摸"
            else:
                loser_id = p.hu_discard_player_id
                method = f"捉 玩家{loser_id} 炮"
            fan_str = ",".join(p.hu_fan_types)
            details.append(f"{method} | {fan_str} | 共{p.hu_fan}番")
        else:
            has_missing = any(t.suit == p.missing_suit for t in p.hand_tiles)
            if has_missing:
                status_tags.append("【花猪】")
                details.append("定缺牌未打完，赔付所有非花猪玩家满番")
            else:
                max_fan, _ = p.calculate_potential_fan()
                if max_fan == 0:
                    status_tags.append("【无叫】")
                    if game.deck.remaining_count() == 0:
                        details.append("流局未听牌，赔付听牌玩家")
                else:
                    status_tags.append("【听牌】")
                    if game.deck.remaining_count() == 0:
                        details.append(f"手握{max_fan}番，理论最大番")

            pao_targets = [str(wid) for wid, lid in dian_pao_players.items() if lid == p.player_id]
            if pao_targets:
                status_tags.append("【点炮】")
                details.append(f"点炮给 -> 玩家 {','.join(pao_targets)}")

        print(f"P{p.player_id} {role} {score_str} {' '.join(status_tags)}")
        for d in details:
            print(f"   └─ {d}")

    print("-" * 76)
    print("当前余额: ", end="")
    for p in game.players:
        print(f"P{p.player_id}:{p.balance}  ", end="")
    print("\n" + "█" * 76 + "\n")


def initialize_llm_agent():
    global HAS_LLM
    if not HAS_LLM:
        return None

    try:
        print("⏳ 正在初始化 LLM Agent...")
        llm_agent = LocalLLMAgent()
        # llm_agent = LocalLLMAgent(enable_xai=ENABLE_XAI and VERBOSE)
        print("✅ 模型加载成功。")
        return llm_agent
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        HAS_LLM = False
        return None


def build_valid_actions(player, game, player_id):
    valid_actions = []
    if player.can_hu():
        valid_actions.append("h")

    gang_info = game.can_self_gang(player_id)
    if gang_info["can_gang"]:
        valid_actions.append("g")

    has_missing = any(t.suit == player.missing_suit for t in player.hand_tiles)
    seen_discard = set()
    for t in player.hand_tiles:
        if has_missing and t.suit != player.missing_suit:
            continue
        t_str = str(t)
        if t_str not in seen_discard:
            valid_actions.append(f"d {t_str}")
            seen_discard.add(t_str)

    return valid_actions


def decide_with_llm(llm_agent, prompt):
    if VERBOSE:
        return llm_agent.decide(prompt)
    with suppress_stdout():
        return llm_agent.decide(prompt)


def normalize_action(action, valid_actions):
    if action in valid_actions:
        return action

    for valid_action in valid_actions:
        if valid_action in action:
            return valid_action

    if valid_actions:
        return valid_actions[0]
    return "n"


def normalize_response(choice, valid_resps):
    if choice in valid_resps:
        return choice

    if "h" in choice and "h" in valid_resps:
        return "h"
    if "p" in choice and "p" in valid_resps:
        return "p"
    if "g" in choice and "g" in valid_resps:
        return "g"
    return "n"


def create_stats():
    return {
        "hu_count": [0] * 4,
        "hu_fan_ge_1": [0] * 4,
        "dianpao_count": [0] * 4,
        "total_fan": [0] * 4,
    }


def run_single_evaluation(llm_agent, run_idx: int):
    player_names = [get_model_display_name(), "Bot-1", "Bot-2", "Bot-3"]
    bots_config = [False, True, True, True]

    global_balances = [INITIAL_BALANCE] * 4
    balance_history = [[INITIAL_BALANCE] * 4]
    stats = create_stats()

    iterator = range(NUM_EPISODES) if VERBOSE else tqdm(
        range(NUM_EPISODES),
        desc=f"对战进度 Run {run_idx}/{NUM_RUNS}",
    )

    for ep in iterator:
        if VERBOSE:
            print(f"\n📢 >>>>>> 第 {ep + 1} 局开始 <<<<<<")

        game = MahjongGame(f"EVAL_{run_idx}_{ep}", player_names, bots=bots_config)
        for i, p in enumerate(game.players):
            p.balance = global_balances[i]

        round_start_balances = [p.balance for p in game.players]
        game.start_game()

        game.phase = game.phase.EXCHANGE
        for p in game.players:
            game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))

        game.phase = game.phase.CHOOSE_MISSING
        for p in game.players:
            game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))

        game.phase = game.phase.PLAYING
        skip_draw = True
        game_step_count = 0

        while not game.is_game_over:
            game_step_count += 1
            if game_step_count > 300 or sum(1 for p in game.players if p.is_hu) >= 3:
                game.is_game_over = True
                break

            pid = game.current_player_id
            player = game.players[pid]

            if player.is_hu:
                game.next_player()
                skip_draw = False
                continue

            drawn = None
            if not skip_draw:
                drawn = game.draw_tile(pid)
                if not drawn:
                    game.check_game_over()
                    break
            else:
                skip_draw = False

            print_game_snapshot(game, pid, drawn)

            turn_end = False
            loop_attempts = 0

            while not turn_end:
                loop_attempts += 1
                action = ""
                force_bot = loop_attempts > 3

                if HAS_LLM and llm_agent is not None and pid == LLM_PLAYER_ID and not force_bot:
                    try:
                        valid_actions = build_valid_actions(player, game, pid)
                        prompt = build_observation_prompt(game, pid, valid_actions)

                        if VERBOSE:
                            print(f"🤖 [AI 思考] (可选: {len(valid_actions)} 个动作)...")
                        action = decide_with_llm(llm_agent, prompt)
                        if VERBOSE:
                            print(f"🤖 [AI 决策]: {action}")
                        action = normalize_action(action, valid_actions)
                    except Exception as e:
                        if VERBOSE:
                            print(f"❌ AI 出错: {e}")
                        action = bot_decide_turn_action(player, game)
                else:
                    action = bot_decide_turn_action(player, game)
                    if VERBOSE and pid != LLM_PLAYER_ID:
                        print(f"🤖 [Bot P{pid}]: {action}")

                if action == "h":
                    if player.can_hu():
                        win_card = drawn if drawn else player.hand_tiles[-1]
                        if VERBOSE:
                            print(f"🎉 P{pid} 自摸胡牌！")
                        game.hu(pid, win_card, True)
                        game.check_game_over()
                        if game.is_game_over:
                            turn_end = True
                            break
                        turn_end = True
                        game.next_player()
                        skip_draw = False
                elif action == "g":
                    g_info = game.can_self_gang(pid)
                    if g_info["can_gang"]:
                        if VERBOSE:
                            print(f"💥 P{pid} 杠牌！")
                        game.gang(pid, g_info["gang_tiles"][0])
                        continue
                elif action.startswith("d "):
                    t = parse_console_tile(action[2:])
                    if t and game.discard_tile(pid, t):
                        if VERBOSE:
                            print(f"👉 P{pid} 打出: {t}")

                        responses = game.check_responses(t, pid)
                        someone_responded = False

                        if responses:
                            for r_id, acts in responses.items():
                                if someone_responded:
                                    break

                                responder = game.players[r_id]
                                choice = "n"

                                if HAS_LLM and llm_agent is not None and r_id == LLM_PLAYER_ID:
                                    try:
                                        valid_resps = ["n"]
                                        if "hu" in acts:
                                            valid_resps.append("h")
                                        if "gang" in acts:
                                            valid_resps.append("g")
                                        if "peng" in acts:
                                            valid_resps.append("p")

                                        prompt = build_observation_prompt(game, r_id, valid_resps)
                                        prompt += f"\n【突发事件】\n对手 P{pid} 打出了 {t}，触发响应机会。"

                                        if VERBOSE:
                                            print(f"⚡ [AI 响应思考] 对手打出 {t} (可选: {valid_resps})...")
                                        choice = decide_with_llm(llm_agent, prompt)
                                        if VERBOSE:
                                            print(f"⚡ [AI 响应]: {choice}")
                                        choice = normalize_response(choice, valid_resps)
                                    except Exception:
                                        choice = "n"
                                else:
                                    choice = bot_decide_response(responder, acts)

                                if choice == "h" and "hu" in acts:
                                    if VERBOSE:
                                        print(f"🎉 P{r_id} 食胡！点炮者: P{pid}")
                                    game.hu(r_id, t, False, pid)
                                    stats["dianpao_count"][pid] += 1
                                    game.check_game_over()
                                    someone_responded = True
                                    if game.is_game_over:
                                        turn_end = True
                                        break
                                elif choice == "g" and "gang" in acts:
                                    if VERBOSE:
                                        print(f"💥 P{r_id} 明杠: {t}")
                                    game.gang(r_id, t, pid)
                                    game.current_player_id = r_id
                                    turn_end = True
                                    someone_responded = True
                                    skip_draw = True
                                elif choice == "p" and "peng" in acts:
                                    if VERBOSE:
                                        print(f"🤜 P{r_id} 碰牌: {t}")
                                    game.peng(r_id, t, pid)
                                    game.current_player_id = r_id
                                    turn_end = True
                                    someone_responded = True
                                    skip_draw = True

                        if game.is_game_over:
                            break

                        if not someone_responded:
                            game.next_player()
                            skip_draw = False
                        turn_end = True

        game.check_game_over()
        print_detailed_settlement(game, round_start_balances)

        for i, p in enumerate(game.players):
            global_balances[i] = p.balance
            if p.is_hu:
                stats["hu_count"][i] += 1
                stats["total_fan"][i] += p.hu_fan
                if p.hu_fan >= 2:
                    stats["hu_fan_ge_1"][i] += 1

        balance_history.append(global_balances.copy())

    print("\n" + "=" * 104)
    print(f"📊 Run {run_idx} 单次评测结果 (共 {NUM_EPISODES} 局)")
    print("=" * 104)
    print(f"{'ID':<4} {'Role':<15} {'Wins':<8} {'HuRate':<10} {'Win(>=2F)':<12} {'DianPao':<10} {'TotalFan':<10} {'Balance':<12} {'Net':<10}")
    print("-" * 104)

    for i in range(4):
        role = player_names[i]
        wins = stats["hu_count"][i]
        hu_rate = (wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
        win_ge_2f = stats["hu_fan_ge_1"][i]
        dianpao = stats["dianpao_count"][i]
        total_fan = stats["total_fan"][i]
        balance = global_balances[i]
        net = balance - INITIAL_BALANCE
        net_str = f"+{net}" if net > 0 else str(net)

        print(
            f"{i:<4} {role:<15} {wins:<8} {hu_rate:<9.2f}% {win_ge_2f:<12} "
            f"{dianpao:<10} {total_fan:<10} {balance:<12} {net_str:<10}"
        )

    print("-" * 104)

    return {
        "player_names": player_names,
        "balance_history": balance_history,
        "stats": stats,
        "final_balances": global_balances,
    }


def summarize_average_results(results):
    history_stack = np.array([r["balance_history"] for r in results], dtype=float)
    avg_history = history_stack.mean(axis=0)

    avg_stats = {}
    for key in results[0]["stats"].keys():
        metric_stack = np.array([r["stats"][key] for r in results], dtype=float)
        avg_stats[key] = metric_stack.mean(axis=0)

    final_balance_stack = np.array([r["final_balances"] for r in results], dtype=float)
    avg_final_balances = final_balance_stack.mean(axis=0)

    return {
        "player_names": results[0]["player_names"],
        "avg_history": avg_history,
        "avg_stats": avg_stats,
        "avg_final_balances": avg_final_balances,
    }


def build_ai_summary(summary):
    avg_wins = float(summary["avg_stats"]["hu_count"][LLM_PLAYER_ID])
    avg_hu_rate = (avg_wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
    avg_win_ge_2f = float(summary["avg_stats"]["hu_fan_ge_1"][LLM_PLAYER_ID])
    avg_dianpao = float(summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID])
    avg_total_fan = float(summary["avg_stats"]["total_fan"][LLM_PLAYER_ID])
    avg_balance = float(summary["avg_final_balances"][LLM_PLAYER_ID])
    avg_net = avg_balance - INITIAL_BALANCE

    return {
        "model_tag": MODEL_TAG,
        "num_episodes": NUM_EPISODES,
        "num_runs": NUM_RUNS,
        "initial_balance": INITIAL_BALANCE,
        "avg_wins": avg_wins,
        "avg_hu_rate": avg_hu_rate,
        "avg_win_ge_2f": avg_win_ge_2f,
        "avg_dianpao": avg_dianpao,
        "avg_total_fan": avg_total_fan,
        "avg_balance": avg_balance,
        "avg_net": avg_net,
    }


def save_summary_results(summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    npz_path = os.path.join(RESULTS_DIR, f"{MODEL_TAG}_summary.npz")
    json_path = os.path.join(RESULTS_DIR, f"{MODEL_TAG}_summary.json")

    np.savez(
        npz_path,
        model_tag=MODEL_TAG,
        player_names=np.array(summary["player_names"], dtype=object),
        avg_history=np.array(summary["avg_history"], dtype=float),
        hu_count=np.array(summary["avg_stats"]["hu_count"], dtype=float),
        hu_fan_ge_1=np.array(summary["avg_stats"]["hu_fan_ge_1"], dtype=float),
        dianpao_count=np.array(summary["avg_stats"]["dianpao_count"], dtype=float),
        total_fan=np.array(summary["avg_stats"]["total_fan"], dtype=float),
        avg_final_balances=np.array(summary["avg_final_balances"], dtype=float),
        num_episodes=np.array(NUM_EPISODES),
        num_runs=np.array(NUM_RUNS),
        initial_balance=np.array(INITIAL_BALANCE),
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(build_ai_summary(summary), f, ensure_ascii=False, indent=2)

    print(f"💾 已保存评估摘要: {npz_path}")
    print(f"💾 已保存指标摘要: {json_path}")


def load_saved_summary(model_tag):
    path = os.path.join(RESULTS_DIR, f"{model_tag}_summary.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return {
        "model_tag": model_tag,
        "raw_model_tag": str(data["model_tag"].item()) if np.ndim(data["model_tag"]) == 0 else str(data["model_tag"]),
        "player_names": data["player_names"].tolist(),
        "avg_history": np.array(data["avg_history"], dtype=float),
        "avg_final_balances": np.array(data["avg_final_balances"], dtype=float),
        "avg_stats": {
            "hu_count": np.array(data["hu_count"], dtype=float),
            "hu_fan_ge_1": np.array(data["hu_fan_ge_1"], dtype=float),
            "dianpao_count": np.array(data["dianpao_count"], dtype=float),
            "total_fan": np.array(data["total_fan"], dtype=float),
        },
        "num_episodes": int(data["num_episodes"]),
        "num_runs": int(data["num_runs"]),
        "initial_balance": float(data["initial_balance"]),
    }


def annotate_last_point(x, y, text, color, x_offset=8, y_offset=0):
    plt.scatter([x], [y], color=color, s=28, zorder=5)
    plt.annotate(
        text,
        xy=(x, y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        color=color,
        fontsize=10,
        fontweight="bold",
        va="center",
    )


def plot_model_comparison_from_saved(
    model_tags=None,
    include_bot_avg=INCLUDE_BOT_AVG_IN_COMPARISON,
    annotate_last_values=True,
):
    if model_tags is None:
        model_tags = COMPARISON_MODEL_TAGS

    summaries = []
    missing_tags = []

    for tag in model_tags:
        loaded = load_saved_summary(tag)
        if loaded is None:
            missing_tags.append(tag)
        else:
            summaries.append(loaded)

    if missing_tags:
        print(f"ℹ️ 暂未生成三模型对比图，缺少结果文件: {', '.join(missing_tags)}")
        return

    base_episodes = summaries[0]["avg_history"].shape[0]
    if any(s["avg_history"].shape[0] != base_episodes for s in summaries):
        print("⚠️ 模型结果的局数不一致，已跳过三模型对比图。")
        return

    if any(s["initial_balance"] != summaries[0]["initial_balance"] for s in summaries):
        print("⚠️ 模型结果的初始资金不一致，已跳过三模型对比图。")
        return

    colors = {
        "qwen": "#666666",
        "qwen-sft-bot": "#FFB347",
        "qwen-sft-selfplay": "#FF8844",
        "qwen-grpo": "#FF4444",
    }

    episodes = range(base_episodes)
    plt.figure(figsize=(12, 7))

    for idx, summary in enumerate(summaries):
        model_tag = summary["model_tag"]
        y_values = summary["avg_history"][:, 0]
        color = colors.get(model_tag, "#333333")
        plt.plot(
            episodes,
            y_values,
            label=get_model_display_name(model_tag),
            linewidth=2.5,
            color=color,
        )
        if annotate_last_values:
            annotate_last_point(
                base_episodes - 1,
                y_values[-1],
                f"{y_values[-1]:.0f}",
                color,
                y_offset=(idx - len(summaries) / 2) * 10,
            )

    if include_bot_avg:
        bot_curves = [summary["avg_history"][:, 1:4].mean(axis=1) for summary in summaries]
        combined_bot_avg = np.mean(np.stack(bot_curves, axis=0), axis=0)
        plt.plot(
            episodes,
            combined_bot_avg,
            label="Bot Avg",
            linewidth=2.0,
            linestyle="--",
            color="#2A7FFF",
        )
        if annotate_last_values:
            annotate_last_point(
                base_episodes - 1,
                combined_bot_avg[-1],
                f"{combined_bot_avg[-1]:.0f}",
                "#2A7FFF",
                y_offset=-14,
            )

    plt.axhline(
        y=summaries[0]["initial_balance"],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="Initial Balance",
    )
    plt.title("Qwen vs Qwen-SFT-Bot vs Qwen-SFT-SelfPlay vs Qwen-GRPO")
    plt.xlabel("Episodes")
    plt.ylabel("Average Balance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"📊 三模型对比图已保存: {filename}")


def redraw_comparison_from_existing_results():
    plot_model_comparison_from_saved(
        model_tags=["qwen", "qwen-sft-bot", "qwen-sft-selfplay", "qwen-grpo"],
        include_bot_avg=True,
        annotate_last_values=True,
    )


def plot_balance_curve(history, player_names):
    history = np.array(history, dtype=float)
    episodes = range(len(history))
    plt.figure(figsize=(12, 7))

    colors = ["#FF4444", "#FFDD44", "#44AAFF", "#44FF44"]
    styles = ["-", "--", "--", "--"]
    widths = [3, 1.5, 1.5, 1.5]
    markers = ["o", None, None, None]

    for i, name in enumerate(player_names):
        plt.plot(
            episodes,
            history[:, i],
            label=name,
            color=colors[i],
            linewidth=widths[i],
            linestyle=styles[i],
            marker=markers[i],
            markersize=4 if markers[i] else None,
        )

    plt.axhline(y=INITIAL_BALANCE, color="gray", linestyle=":", alpha=0.5)
    plt.title(f"Average Evaluation Result ({NUM_RUNS} Runs)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Balance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = "local_llm_result_avg.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n📈 平均资金曲线已保存: {filename}")


def run_evaluation():
    print(f"\n🚀 开始评估: 微调大模型 (Qwen) vs 规则Bot")
    print(f"   - 当前模型标签: {MODEL_TAG}")
    print(f"   - 单次局数: {NUM_EPISODES}")
    print(f"   - 独立重复次数: {NUM_RUNS}")
    print(f"   - 详细模式: {'✅ 开启' if VERBOSE else '❌ 关闭'}")

    llm_agent = initialize_llm_agent()
    results = []

    for run_idx in range(1, NUM_RUNS + 1):
        results.append(run_single_evaluation(llm_agent, run_idx))

    summary = summarize_average_results(results)

    print("\n" + "=" * 124)
    print(f"📊 {NUM_RUNS} 次独立实验平均结果 (共 {NUM_RUNS} x {NUM_EPISODES} 局)")
    print("=" * 124)
    print(f"{'ID':<4} {'Role':<15} {'AvgWins':<10} {'AvgHuRate':<12} {'AvgWin(>=2F)':<14} {'AvgDianPao':<12} {'AvgTotalFan':<14} {'AvgBalance':<14} {'AvgNet':<12}")
    print("-" * 124)

    for i in range(4):
        role = summary["player_names"][i]
        avg_wins = summary["avg_stats"]["hu_count"][i]
        avg_hu_rate = (avg_wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
        avg_win_ge_2f = summary["avg_stats"]["hu_fan_ge_1"][i]
        avg_dianpao = summary["avg_stats"]["dianpao_count"][i]
        avg_total_fan = summary["avg_stats"]["total_fan"][i]
        avg_balance = summary["avg_final_balances"][i]
        avg_net = avg_balance - INITIAL_BALANCE
        avg_net_str = f"+{avg_net:.2f}" if avg_net > 0 else f"{avg_net:.2f}"

        print(
            f"{i:<4} {role:<15} {avg_wins:<10.2f} {avg_hu_rate:<11.2f}% {avg_win_ge_2f:<14.2f} "
            f"{avg_dianpao:<12.2f} {avg_total_fan:<14.2f} {avg_balance:<14.2f} {avg_net_str:<12}"
        )

    print("-" * 124)

    ai_avg_wins = summary["avg_stats"]["hu_count"][LLM_PLAYER_ID]
    ai_avg_hu_rate = (ai_avg_wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0.0
    ai_avg_win_ge_2f = summary["avg_stats"]["hu_fan_ge_1"][LLM_PLAYER_ID]
    ai_avg_dianpao = summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID]
    ai_avg_total_fan = summary["avg_stats"]["total_fan"][LLM_PLAYER_ID]
    ai_avg_balance = summary["avg_final_balances"][LLM_PLAYER_ID]
    ai_avg_net = ai_avg_balance - INITIAL_BALANCE

    print("\n🤖 AI 核心指标平均值:")
    print(f"   - 平均胡牌次数: {ai_avg_wins:.2f}")
    print(f"   - 平均胡牌率: {ai_avg_hu_rate:.2f}%")
    print(f"   - 平均 2 番及以上胡牌次数: {ai_avg_win_ge_2f:.2f}")
    print(f"   - 平均点炮次数: {ai_avg_dianpao:.2f}")
    print(f"   - 平均总番数: {ai_avg_total_fan:.2f}")
    print(f"   - 平均总收益: {ai_avg_net:.2f}")

    plot_balance_curve(summary["avg_history"], summary["player_names"])
    save_summary_results(summary)
    plot_model_comparison_from_saved()


if __name__ == "__main__":
    run_evaluation()

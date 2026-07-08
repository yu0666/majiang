# import json
# from typing import List, Dict
# from openai import OpenAI
# from game import MahjongGame, PlayerState, Tile
# from rule_engine import ShantenCalculator # 引入规则引擎计算向听数

# class LLMAgent:
#     def __init__(self):
#         # 配置腾讯混元大模型
#         # self.client = OpenAI(
#         #     api_key=os.getenv("HUNYUAN_API_KEY", ""), 
#         #     base_url="https://api.hunyuan.cloud.tencent.com/v1"
#         # )
#         # self.model_name = "hunyuan-pro"
#         # self.client = OpenAI(
#         #     api_key=os.getenv("DASHSCOPE_API_KEY", ""), 
#         #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#         # )
#         # self.model_name = "qwen-plus-latest"
#         self.client = OpenAI(
#             api_key=os.getenv("DASHSCOPE_API_KEY", ""), 
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#         )
#         self.model_name = "qwen-plus"

#     def get_action(self, player: PlayerState, game: MahjongGame, valid_actions: List[str]) -> str:
#         # 1. 预计算牌理解析（这是让大模型变聪明的关键）
#         analysis_info = self._analyze_shanten(player)
        
#         # 2. 构建提示词 (注入分析结果)
#         prompt = self._construct_prompt(player, game, valid_actions, analysis_info)
        
#         # 3. 调用大模型
#         response_text = self._call_llm_api(prompt)
        
#         # 4. 解析动作
#         action = self._parse_response(response_text, valid_actions)
#         return action

#     def _analyze_shanten(self, player: PlayerState) -> str:
#         """
#         帮大模型算牌：计算打出每一张牌后的向听数
#         """
#         # 如果有定缺，必须打定缺，不需要算向听
#         if player.missing_suit:
#             missing_tiles = [t for t in player.hand_tiles if t.suit == player.missing_suit]
#             if missing_tiles:
#                 return f"【强制规则】你定缺花色是 {player.missing_suit.value}，手牌中还有此花色，必须优先打出！"

#         # 正常计算
#         hand = player.hand_tiles.copy()
#         unique_tiles = sorted(list(set(hand)), key=lambda t: (t.suit.value, t.number))
        
#         analysis_lines = []
#         best_shanten = 100
        
#         for tile in unique_tiles:
#             # 模拟打出这张牌
#             temp_hand = hand.copy()
#             temp_hand.remove(tile)
            
#             # 计算剩余手牌的向听数
#             s = ShantenCalculator.calculate_shanten(temp_hand, player.missing_suit)
            
#             if s < best_shanten:
#                 best_shanten = s
            
#             # 标记
#             shanten_str = "听牌!" if s == 0 else f"{s}向听"
#             analysis_lines.append(f"- 打 {str(tile)} -> 剩余{shanten_str} (Shanten={s})")

#         # 总结建议
#         result = "【牌理分析 (Shanten Analysis)】:\n" + "\n".join(analysis_lines)
#         result += f"\n\n>>> 最佳策略：请选择打出后 Shanten={best_shanten} 的牌，这样最快胡牌。"
#         return result

#     def _call_llm_api(self, prompt: str) -> str:
#         # 修改 🤖 -> [LLM]
#         print(f"\n[LLM] 混元大模型 ({self.model_name}) 正在思考...")
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": """你是四川麻将高手。你的核心目标是【极速胡牌】。
# 请严格根据用户的【牌理分析】做决策：
# 1. 必须优先打出【向听数 (Shanten)】最小的牌。
# 2. 如果有定缺牌，必须打定缺。
# 3. 只输出动作指令，不要废话。"""
#                     },
#                     {
#                         "role": "user", 
#                         "content": prompt
#                     }
#                 ],
#                 temperature=0.1, 
#                 max_tokens=20 
#             )
#             content = response.choices[0].message.content.strip()
#             # 修改 🤖 -> [LLM]
#             print(f"[LLM] 返回: {content}")
#             return content
#         except Exception as e:
#             # 修改 ❌ -> [Error]
#             print(f"[Error] API 调用失败: {e}")
#             return "pass"

#     def _construct_prompt(self, player: PlayerState, game: MahjongGame, valid_actions: List[str], analysis_info: str) -> str:
        
#         is_discard_phase = any(act.startswith("d ") for act in valid_actions)

#         # 基础信息
#         hand_str = " ".join([str(t) for t in player.hand_tiles])
#         melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
#         missing = player.missing_suit.value if player.missing_suit else "未定"
        
#         # 场面
#         discards_info = []
#         for p in game.players:
#             last = p.discarded_tiles[-1] if p.discarded_tiles else "无"
#             discards_info.append(f"P{p.player_id}: {last}")
        
#         if is_discard_phase:
#             phase_instruction = f"""
#             【当前阶段：出牌】
#             {analysis_info}
            
#             请根据上面的【牌理分析】，选择 Shanten 值最小的牌打出。
#             """
#         else:
#             phase_instruction = f"""
#             【当前阶段：响应 (碰/杠/胡)】
#             有人打出了一张牌，触发了你的动作。
#             - 如果能胡 (h)，直接胡！
#             - 如果能杠 (g)，且杠后向听数不退步，建议杠。
#             - 如果能碰 (p)，请判断碰后是否有利于进张。
#             - 严禁打牌 (不要输出 d ...)。
#             """

#         prompt = f"""
#         【局势】
#         手牌: {hand_str}
#         定缺: {missing}
#         副露: {melds_str}
#         场面: {', '.join(discards_info)}

#         【合法动作】
#         {json.dumps(valid_actions, ensure_ascii=False)}

#         {phase_instruction}

#         指令:
#         """
#         return prompt

#     def _parse_response(self, response: str, valid_actions: List[str]) -> str:
#         clean_act = response.replace("```", "").replace("'", "").replace('"', "").strip().lower()
        
#         if clean_act in valid_actions:
#             return clean_act
            
#         map_dict = {"碰": "p", "杠": "g", "胡": "h", "过": "n", "pass": "n"}
#         if clean_act in map_dict:
#             mapped = map_dict[clean_act]
#             if mapped in valid_actions:
#                 return mapped

#         for act in valid_actions:
#             if act in clean_act:
#                 return act
        
#         print(f"[修正] 模型输出 '{response}' 无效，执行兜底策略")
        
#         # 智能兜底：如果是出牌阶段，帮它选向听数最小的（既然模型犯傻了，我们代码接管）
#         if any(a.startswith('d ') for a in valid_actions):
#             # 这里简单返回第一个，因为我们在Prompt里已经尽力了
#             # 也可以在这里再次调用 analyze_shanten 选最优，但为了简化直接返回
#             return valid_actions[0]
            
#         if 'n' in valid_actions: return 'n'
#         return valid_actions[0]
import json
from typing import List, Dict
from openai import OpenAI
from game import MahjongGame, PlayerState, Tile
from rule_engine import ShantenCalculator 

class LLMAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        初始化 LLM Agent，支持动态传入配置
        """
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        self.model_name = model_name

    def get_action(self, player: PlayerState, game: MahjongGame, valid_actions: List[str]) -> str:
        # 1. 预计算牌理解析
        analysis_info = self._analyze_shanten(player)
        
        # 2. 构建提示词
        prompt = self._construct_prompt(player, game, valid_actions, analysis_info)
        
        # 3. 调用大模型
        response_text = self._call_llm_api(prompt)
        
        # 4. 解析动作
        action = self._parse_response(response_text, valid_actions)
        return action

    def _analyze_shanten(self, player: PlayerState) -> str:
        # 如果有定缺，必须打定缺
        if player.missing_suit:
            missing_tiles = [t for t in player.hand_tiles if t.suit == player.missing_suit]
            if missing_tiles:
                return f"【强制规则】你定缺花色是 {player.missing_suit.value}，必须优先打出！"

        hand = player.hand_tiles.copy()
        unique_tiles = sorted(list(set(hand)), key=lambda t: (t.suit.value, t.number))
        
        analysis_lines = []
        best_shanten = 100
        
        for tile in unique_tiles:
            temp_hand = hand.copy()
            temp_hand.remove(tile)
            s = ShantenCalculator.calculate_shanten(temp_hand, player.missing_suit)
            if s < best_shanten: best_shanten = s
            shanten_str = "听牌!" if s == 0 else f"{s}向听"
            analysis_lines.append(f"- 打 {str(tile)} -> {shanten_str} (Shanten={s})")

        return "【牌理分析】:\n" + "\n".join(analysis_lines) + f"\n建议：选择打出后 Shanten={best_shanten} 的牌。"

    def _call_llm_api(self, prompt: str) -> str:
        # 打印当前是哪个模型在思考
        print(f"\n[{self.model_name}] 正在思考...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是四川麻将高手。目标：极速胡牌。规则：1.优先打向听数最小的牌。2.有定缺必打定缺。3.只输出动作指令(如 'd 1万')。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1, 
                max_tokens=20,
                timeout=8.0 # 稍微延长超时时间，防止并发卡顿
            )
            content = response.choices[0].message.content.strip()
            print(f"[{self.model_name}] 返回: {content}")
            return content
        except Exception as e:
            print(f"[Error] {self.model_name} 调用失败: {e}")
            return "pass"

    def _construct_prompt(self, player: PlayerState, game: MahjongGame, valid_actions: List[str], analysis_info: str) -> str:
        is_discard = any(act.startswith("d ") for act in valid_actions)
        
        hand_str = " ".join([str(t) for t in player.hand_tiles])
        melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
        missing = player.missing_suit.value if player.missing_suit else "未定"
        
        discards = []
        for p in game.players:
            last = p.discarded_tiles[-1] if p.discarded_tiles else "无"
            discards.append(f"P{p.player_id}: {last}")
        
        if is_discard:
            instruction = f"【出牌阶段】\n{analysis_info}\n请选择 Shanten 值最小的牌打出。"
        else:
            instruction = "【响应阶段】\n有人打牌触发动作。能胡则胡(h)，能杠则杠(g)。严禁打牌(d ...)。"

        return f"""
        【局势】手牌:{hand_str} | 定缺:{missing} | 副露:{melds_str} | 场面:{','.join(discards)}
        【动作】{json.dumps(valid_actions, ensure_ascii=False)}
        {instruction}
        指令:
        """

    def _parse_response(self, response: str, valid_actions: List[str]) -> str:
        clean = response.replace("```", "").replace("'", "").replace('"', "").strip().lower()
        if clean in valid_actions: return clean
        
        # 模糊匹配
        for act in valid_actions:
            if act in clean: return act
            
        # 兜底
        if any(a.startswith('d ') for a in valid_actions): return valid_actions[0]
        if 'n' in valid_actions: return 'n'
        return valid_actions[0]

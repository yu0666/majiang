"""
四川麻将游戏主逻辑：封装游戏全流程 (含人机模块 & 循环对局 & 查叫查花猪)
"""
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import random
import time
from tiles import Tile, TileDeck, Suit, sort_tiles
from rule_engine import HandPattern, FanCalculator, check_missing_suit, detect_flower_pig
from rule_engine import ShantenCalculator
# from llm_agent import LLMAgent  # <--- 新增这行

class GamePhase(Enum):
    DEALING = "dealing"
    EXCHANGE = "exchange"
    CHOOSE_MISSING = "choose_missing"
    PLAYING = "playing"
    GAME_OVER = "game_over"


class ExchangeDirection(Enum):
    CLOCKWISE = "clockwise"
    COUNTER_CLOCKWISE = "counter_clockwise"
    OPPOSITE = "opposite"


class ActionType(Enum):
    DISCARD = "discard"
    PENG = "peng"
    GANG = "gang"
    HU = "hu"
    PASS = "pass"


class GangType(Enum):
    MING = "ming"
    AN = "an"
    JIE = "jie"


class PlayerState:
    def __init__(self, player_id: int, name: str, initial_balance: int = 10000, is_bot: bool = False):
        self.player_id = player_id
        self.name = name
        self.is_bot = is_bot
        self.hand_tiles: List[Tile] = []
        self.discarded_tiles: List[Tile] = []
        self.open_melds: List[List[Tile]] = []
        self.missing_suit: Optional[Suit] = None
        self.balance: int = initial_balance
        self.is_hu: bool = False
        self.gang_count: int = 0
        self.concealed_kong_count: int = 0
        self.gang_types: List[GangType] = []
        self.exchange_tiles: Optional[List[Tile]] = None
        self.received_tiles: Optional[List[Tile]] = None
        self.exchange_ready: bool = False
        self.missing_suit_ready: bool = False
        self.hu_fan: int = 0
        self.hu_fan_types: List[str] = []
        self.hu_win_tile: Optional[Tile] = None  # 保存胡的牌
        self.hu_is_self_drawn: bool = False  # 保存是否自摸
        self.hu_discard_player_id: Optional[int] = None  # 保存点炮玩家ID
        self.last_drawn_tile: Optional[Tile] = None  # 保存最后摸到的牌
        self.tian_hu_waiting: bool = False

    def add_tiles(self, tiles: List[Tile]):
        self.hand_tiles.extend(tiles)
        self.hand_tiles = sort_tiles(self.hand_tiles)

    def remove_tile(self, tile: Tile) -> bool:
        for i, t in enumerate(self.hand_tiles):
            if t == tile:
                self.hand_tiles.pop(i)
                return True
        return False

    def discard(self, tile: Tile):
        if self.remove_tile(tile):
            self.discarded_tiles.append(tile)
            return True
        return False

    def can_peng(self, tile: Tile) -> bool:
        if self.missing_suit and tile.suit == self.missing_suit:
            return False
        count = sum(1 for t in self.hand_tiles if t == tile)
        return count >= 2

    def can_gang(self, tile: Tile) -> bool:
        if self.missing_suit and tile.suit == self.missing_suit:
            return False
        count = sum(1 for t in self.hand_tiles if t == tile)
        return count >= 3

    def can_hu(self, new_tile: Optional[Tile] = None) -> bool:
        test_tiles = self.hand_tiles.copy()
        if new_tile:
            test_tiles.append(new_tile)

        # 检查牌数：必须是14张才能胡（或11, 8, 5, 2 - 有明牌的情况）
        # 正常情况：14张（手牌）
        # 有明牌情况：手牌 + 明牌 = 14张，手牌可能是11, 8, 5, 2张
        if len(test_tiles) % 3 != 2:
            return False

        if not check_missing_suit(test_tiles, self.missing_suit):
            return False
        return HandPattern(test_tiles).is_winning_hand()

    def is_ready_with_missing_suit(self) -> Tuple[bool, List[Tile]]:
        if len(self.hand_tiles) % 3 != 1:
            return False, []

        valid_waiting_tiles = []
        # 只检查非缺门花色
        valid_suits = [s for s in Suit if s != self.missing_suit]

        for suit in valid_suits:
            for number in range(1, 10):
                test_tile = Tile(suit, number)
                if self.can_hu(test_tile):
                    valid_waiting_tiles.append(test_tile)
        return len(valid_waiting_tiles) > 0, valid_waiting_tiles

    def calculate_potential_fan(self) -> Tuple[int, List[str]]:
        is_ready, waiting_tiles = self.is_ready_with_missing_suit()
        if not is_ready: return 0, []

        max_fan = 0
        max_fan_types = []
        for waiting_tile in waiting_tiles:
            test_hand = self.hand_tiles + [waiting_tile]
            fan, fan_types = FanCalculator.calculate_fan(
                test_hand,
                waiting_tile,
                open_melds=self.open_melds,
                is_self_drawn=False, # 查叫默认按自摸算最大番
                gang_count=self.gang_count,
                concealed_kong_count=self.concealed_kong_count
            )
            if fan > max_fan:
                max_fan = fan
                max_fan_types = fan_types
        return max_fan, max_fan_types

    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "is_bot": self.is_bot,
            "hand_tiles": [t.to_dict() for t in self.hand_tiles],
            "discarded_tiles": [t.to_dict() for t in self.discarded_tiles],
            "open_melds": [[{"suit": t.suit.value, "number": t.number} for t in meld] for meld in self.open_melds],
            "missing_suit": self.missing_suit.value if self.missing_suit else None,
            "balance": self.balance,
            "is_hu": self.is_hu
        }


class MahjongGame:
    def __init__(self, game_id: str, player_names: List[str], bots: List[bool] = None, game_config: Optional[Dict] = None):
        if len(player_names) != 4: raise ValueError("需要4个玩家")
        if game_config is None: game_config = {}
        if bots is None: bots = [False] * 4

        self.base_score = game_config.get("baseScore", 10)
        self.max_fan = game_config.get("maxFan", 8)
        initial_balance = game_config.get("initialBalance", 10000)

        self.game_id = game_id
        self.deck = TileDeck()
        self.players = [
            PlayerState(i, name, initial_balance, is_bot=bots[i])
            for i, name in enumerate(player_names)
        ]
        self.dealer_id = 0
        self.current_player_id = 0
        self.round_number = 0
        self.last_discarded_tile: Optional[Tile] = None
        self.last_discard_player_id: Optional[int] = None
        self.is_game_over = False
        self.winners: List[int] = []
        self.phase: GamePhase = GamePhase.DEALING
        self.exchange_direction: Optional[ExchangeDirection] = None
        self.first_discard: bool = True
        
        # 【新增】动作历史记录，用于 Transformer 记忆模块
        # 格式: [{"pid": 0, "act": "d", "tile": "1万", "ts": 1}, ...]
        self.history: List[Dict] = []

    # 【新增】记录日志的辅助函数
    def _log_action(self, pid: int, action_type: str, tile: Optional[Tile] = None, details: str = ""):
        """记录动作到历史列表"""
        record = {
            "pid": pid,
            "act": action_type,
            "tile": str(tile) if tile else "",
            "desc": details,
            "ts": time.time()
        }
        self.history.append(record)

    # 【新增】获取格式化的历史文本 (记忆模块核心)
    def get_history_text(self, k: int = 20) -> str:
        """获取最近 k 步的历史记录文本"""
        if not self.history: return "无历史记录"
        
        recent = self.history[-k:]
        lines = []
        for h in recent:
            role = f"P{h['pid']}"
            act = h['act']
            tile = h['tile']
            desc = h.get('desc', '')
            
            if act == 'discard':
                lines.append(f"{role} 打出 {tile}")
            elif act == 'peng':
                lines.append(f"{role} 碰 {tile}")
            elif act == 'gang':
                lines.append(f"{role} {desc} {tile}")
            elif act == 'hu':
                lines.append(f"{role} 胡牌: {tile} ({desc})")
        
        return "\n".join(lines)

    def start_game(self):
        self.deck.shuffle()
        for player in self.players:
            player.add_tiles(self.deck.draw(13))
        self.phase = GamePhase.EXCHANGE
        self.exchange_direction = random.choice(list(ExchangeDirection))
        self.current_player_id = self.dealer_id

    def select_exchange_tiles(self, player_id: int, tiles: List[Tile]) -> bool:
        if self.phase != GamePhase.EXCHANGE: return False
        if len(tiles) != 3: return False

        suits = set(t.suit for t in tiles)
        if len(suits) != 1:
            return False

        player = self.players[player_id]
        temp_hand = list(player.hand_tiles)
        for tile in tiles:
            if tile in temp_hand:
                temp_hand.remove(tile)
            else:
                return False

        player.exchange_tiles = tiles
        player.exchange_ready = True
        if all(p.exchange_ready for p in self.players):
            self._execute_exchange()
        return True

    def _execute_exchange(self):
        if not self.exchange_direction: return
        exchange_map = {}
        for player in self.players:
            if player.exchange_tiles:
                exchange_map[player.player_id] = player.exchange_tiles
                for tile in player.exchange_tiles:
                    player.remove_tile(tile)

        for player in self.players:
            source_id = self._get_exchange_source(player.player_id)
            if source_id in exchange_map:
                received = exchange_map[source_id]
                player.received_tiles = received
                player.add_tiles(received)

        dealer = self.players[self.dealer_id]
        dealer.add_tiles(self.deck.draw(1))
        self.phase = GamePhase.CHOOSE_MISSING

    def _get_exchange_source(self, player_id: int) -> int:
        if self.exchange_direction == ExchangeDirection.CLOCKWISE:
            return (player_id - 1) % 4
        elif self.exchange_direction == ExchangeDirection.COUNTER_CLOCKWISE:
            return (player_id + 1) % 4
        else:
            return (player_id + 2) % 4

    def set_missing_suit(self, player_id: int, suit: Suit) -> bool:
        if self.phase != GamePhase.CHOOSE_MISSING: return False
        player = self.players[player_id]
        player.missing_suit = suit
        player.missing_suit_ready = True
        if all(p.missing_suit_ready for p in self.players):
            self.phase = GamePhase.PLAYING
            self._check_tian_hu()
        return True

    def _auto_select_missing_suit(self, player: PlayerState):
        suit_counts = {Suit.WAN: 0, Suit.TIAO: 0, Suit.TONG: 0}
        for tile in player.hand_tiles:
            suit_counts[tile.suit] += 1
        min_suit = min(suit_counts, key=suit_counts.get)
        player.missing_suit = min_suit
        player.missing_suit_ready = True
        print(f"玩家{player.player_id} 自动定缺: {min_suit.value}")

    def _check_tian_hu(self) -> bool:
        dealer = self.players[self.dealer_id]
        if len(dealer.hand_tiles) == 14 and dealer.can_hu():
            # print(f"\n[天胡！] 庄家{self.dealer_id} 起手胡牌！")
            dealer.is_hu = True
            dealer.tian_hu_waiting = True
            # 【新增】记录天胡
            self._log_action(self.dealer_id, "hu", dealer.hand_tiles[-1], "天胡")
            return True
        return False

    def draw_tile(self, player_id: int) -> Optional[Tile]:
        if self.deck.remaining_count() == 0: return None
        tile = self.deck.draw(1)[0]
        player = self.players[player_id]
        player.add_tiles([tile])
        player.last_drawn_tile = tile  # 保存最后摸到的牌
        return tile

    def discard_tile(self, player_id: int, tile: Tile) -> bool:
        player = self.players[player_id]

        if player.missing_suit:
            if tile.suit != player.missing_suit:
                has_missing_suit_tiles = any(t.suit == player.missing_suit for t in player.hand_tiles)
                if has_missing_suit_tiles:
                    return False

        if player.discard(tile):
            # 【新增】记录出牌
            self._log_action(player_id, "discard", tile, "打出") 
            self.last_discarded_tile = tile
            self.last_discard_player_id = player_id
            if self.first_discard: self.first_discard = False
            return True
        return False

    def peng(self, player_id: int, tile: Tile, discard_player_id: Optional[int] = None) -> bool:
        player = self.players[player_id]
        if not player.can_peng(tile): return False
        if discard_player_id is not None:
            discard_player = self.players[discard_player_id]
            if discard_player.discarded_tiles and discard_player.discarded_tiles[-1] == tile:
                discard_player.discarded_tiles.pop()

        removed_count = 0
        for _ in range(2):
            for t in player.hand_tiles:
                if t == tile:
                    player.hand_tiles.remove(t)
                    removed_count += 1
                    break
        
        if removed_count != 2: return False
        player.open_melds.append([tile, tile, tile])
        # 【新增】记录碰牌
        self._log_action(player_id, "peng", tile, f"碰 P{discard_player_id}")
        return True

    def gang(self, player_id: int, tile: Tile, discard_player_id: Optional[int] = None) -> Tuple[bool, Optional[Tile]]:
        player = self.players[player_id]

        tiles_to_remove = 0
        is_bu_gang = False

        if discard_player_id is not None:
            if not player.can_gang(tile): return False, None
            tiles_to_remove = 3
            discard_player = self.players[discard_player_id]
            if discard_player.discarded_tiles and discard_player.discarded_tiles[-1] == tile:
                discard_player.discarded_tiles.pop()
        else:
            count_in_hand = sum(1 for t in player.hand_tiles if t == tile)
            if count_in_hand == 4:
                tiles_to_remove = 4
            elif count_in_hand == 1:
                has_triplet = any(len(m)==3 and m[0]==tile for m in player.open_melds)
                if not has_triplet: return False, None
                tiles_to_remove = 1
                is_bu_gang = True
            else:
                return False, None

        removed_count = 0
        for _ in range(tiles_to_remove):
            for t in player.hand_tiles:
                if t == tile:
                    player.hand_tiles.remove(t)
                    removed_count += 1
                    break
        
        if removed_count != tiles_to_remove:
            return False, None

        if is_bu_gang:
            for meld in player.open_melds:
                if len(meld) == 3 and meld[0] == tile:
                    meld.append(tile)
                    break
        else:
            player.open_melds.append([tile, tile, tile, tile])

        player.gang_count += 1
        
        # 【新增】记录杠动作
        g_type = "直杠" if discard_player_id is not None else ("补杠" if is_bu_gang else "暗杠")
        self._log_action(player_id, "gang", tile, g_type) 

        drawn_tile = None
        if self.deck.remaining_count() > 0:
            drawn = self.deck.draw(1)
            player.add_tiles(drawn)
            drawn_tile = drawn[0]

        return True, drawn_tile

    def check_responses(self, tile: Tile, discard_player_id: int) -> Dict[int, List[str]]:
        responses = {}
        for player in self.players:
            if player.player_id == discard_player_id or player.is_hu: continue
            available = []
            if player.can_hu(tile): available.append("hu")
            if player.can_gang(tile): available.append("gang")
            if player.can_peng(tile): available.append("peng")
            if available: responses[player.player_id] = available
        return responses

    def can_self_gang(self, player_id: int, drawn_tile: Optional[Tile] = None) -> Dict:
        player = self.players[player_id]
        result = {"can_gang": False, "gang_tiles": [], "gang_types": {}}
        tile_counts = {}
        for tile in player.hand_tiles:
            ts = f"{tile.number}{tile.suit.value}"
            tile_counts[ts] = tile_counts.get(ts, 0) + 1

        for ts, count in tile_counts.items():
            if count == 4:
                target_tile = None
                for tile in player.hand_tiles:
                    if f"{tile.number}{tile.suit.value}" == ts:
                        target_tile = tile
                        break
                
                if target_tile and player.missing_suit and target_tile.suit == player.missing_suit:
                    continue

                if target_tile:
                    result["gang_tiles"].append(target_tile)
                    result["gang_types"][ts] = "an"
                    result["can_gang"] = True

        for meld in player.open_melds:
            if len(meld) == 3 and meld[0] == meld[1]:
                mt = meld[0]
                mts = f"{mt.number}{mt.suit.value}"
                
                if player.missing_suit and mt.suit == player.missing_suit:
                    continue

                if mts in tile_counts and tile_counts[mts] >= 1:
                    if mt not in result["gang_tiles"]:
                        result["gang_tiles"].append(mt)
                        result["gang_types"][mts] = "bu"
                        result["can_gang"] = True
        return result

    def hu(self, player_id: int, win_tile: Tile, is_self_drawn: bool = True, discard_player_id: Optional[int] = None) -> Tuple[int, List[str]]:
        player = self.players[player_id]

        # ========== 调试：验证win_tile是否正确 ==========
        if not is_self_drawn and discard_player_id is not None:
            # 点炮胡牌：验证win_tile是否是打牌玩家打出的牌
            discard_player = self.players[discard_player_id]
            if discard_player.discarded_tiles and discard_player.discarded_tiles[-1] != win_tile:
                print(f"\n[警告] 点炮胡牌的win_tile不匹配！")
                print(f"  传入的win_tile: {win_tile}")
                print(f"  打牌玩家{discard_player_id}最后打出的牌: {discard_player.discarded_tiles[-1]}")
                # 修正win_tile
                win_tile = discard_player.discarded_tiles[-1]
                print(f"  已修正为: {win_tile}")
        # ========== 调试结束 ==========

        check_tile = None if is_self_drawn else win_tile
        if not player.can_hu(check_tile): return 0, []

        if not is_self_drawn and discard_player_id is not None:
            discard_player = self.players[discard_player_id]
            if discard_player.discarded_tiles and discard_player.discarded_tiles[-1] == win_tile:
                discard_player.discarded_tiles.pop()

        all_tiles = player.hand_tiles.copy()
        if not is_self_drawn: all_tiles.append(win_tile)

        is_tian_hu = False
        is_di_hu = False
        if self.first_discard:
            if is_self_drawn and player_id == self.dealer_id: is_tian_hu = True
            elif not is_self_drawn and player_id != self.dealer_id: is_di_hu = True

        fan, fan_types = FanCalculator.calculate_fan(
            all_tiles, win_tile,
            open_melds=player.open_melds,
            is_self_drawn=is_self_drawn,
            gang_count=player.gang_count,
            is_last_tile=(self.deck.remaining_count()==0),
            is_tian_hu=is_tian_hu, is_di_hu=is_di_hu
        )

        player.is_hu = True
        player.hu_fan = fan
        player.hu_fan_types = fan_types
        player.hu_win_tile = win_tile  # 保存胡的牌
        player.hu_is_self_drawn = is_self_drawn  # 保存是否自摸
        player.hu_discard_player_id = discard_player_id  # 保存点炮玩家ID
        self.winners.append(player_id)
        self._settle_balance(player_id, fan, is_self_drawn, discard_player_id)
        
        # 【新增】记录胡牌
        h_type = "自摸" if is_self_drawn else f"捉炮(P{discard_player_id})"
        self._log_action(player_id, "hu", win_tile, f"{h_type} {','.join(fan_types)}")
        
        return fan, fan_types

    def _settle_balance(self, winner_id: int, fan: int, is_self_drawn: bool, discard_player_id: Optional[int] = None):
        multiplier = 2 ** min(fan, self.max_fan)
        score = self.base_score * multiplier
        winner = self.players[winner_id]
        if is_self_drawn:
            for p in self.players:
                if p.player_id != winner_id and not p.is_hu:
                    p.balance -= score
                    winner.balance += score
        else:
            if discard_player_id is not None:
                payer = self.players[discard_player_id]
                payer.balance -= score
                winner.balance += score

    def check_game_over(self, check_round_only: bool = False) -> bool:
        hu_count = sum(1 for p in self.players if p.is_hu)
        if hu_count >= 3:
            if not check_round_only: self.is_game_over = True
            return True
        if self.deck.remaining_count() == 0:
            self._handle_liuju()
            if not check_round_only: self.is_game_over = True
            return True
        return False

    def _handle_liuju(self):
        self._settle_hua_zhu()
        self._settle_da_jiao()

    def _settle_hua_zhu(self):
        hua_zhu_players = []
        non_hu_players = [p for p in self.players if not p.is_hu]

        for p in non_hu_players:
            has_missing = any(t.suit == p.missing_suit for t in p.hand_tiles)
            if has_missing:
                hua_zhu_players.append(p)

        for pig in hua_zhu_players:
            penalty_score = self.base_score * (2 ** self.max_fan)
            for target in non_hu_players:
                if target not in hua_zhu_players:
                    pig.balance -= penalty_score
                    target.balance += penalty_score

    def _settle_da_jiao(self):
        non_hu_players = [p for p in self.players if not p.is_hu]
        valid_players = []
        for p in non_hu_players:
            has_missing = any(t.suit == p.missing_suit for t in p.hand_tiles)
            if not has_missing:
                valid_players.append(p)

        ready_players = []
        not_ready_players = []

        for p in valid_players:
            max_fan, _ = p.calculate_potential_fan()
            if max_fan > 0:
                ready_players.append((p, max_fan))
            else:
                not_ready_players.append(p)

        for loser in not_ready_players:
            for winner, fan in ready_players:
                score = self.base_score * (2 ** min(fan, self.max_fan))
                loser.balance -= score
                winner.balance += score

    def next_player(self):
        attempts = 0
        while attempts < 4:
            self.current_player_id = (self.current_player_id + 1) % 4
            if not self.players[self.current_player_id].is_hu:
                break
            attempts += 1

    def get_game_state(self) -> Dict:
        return {
            "game_id": self.game_id,
            "phase": self.phase.value,
            "current_player_id": self.current_player_id,
            "players": [p.to_dict() for p in self.players],
            "is_game_over": self.is_game_over,
            "winners": self.winners
        }

# =============================================================================
# 真人控制台启动代码 & AI逻辑
# =============================================================================

def parse_console_tile(input_str: str) -> Optional[Tile]:
    try:
        suit_map = {'w': Suit.WAN, '万': Suit.WAN, 't': Suit.TIAO, '条': Suit.TIAO, 'd': Suit.TONG, '筒': Suit.TONG}
        import re
        match = re.match(r'(\d+)(.*)', input_str.strip())
        if not match: return None
        number = int(match.group(1))
        suit_str = match.group(2).lower()
        suit = suit_map.get(suit_str)
        if suit and 1 <= number <= 9: return Tile(suit, number)
    except: pass
    return None

def print_player_hand(player: PlayerState, prefix: str = ""):
    tiles_str = " ".join([str(t) for t in player.hand_tiles])
    melds_str = ""
    if player.open_melds:
        melds_list = []
        for meld in player.open_melds:
            m_str = "[" + " ".join(str(t) for t in meld) + "]"
            melds_list.append(m_str)
        melds_str = "  明牌: " + " ".join(melds_list)

    role = "[BOT]" if player.is_bot else "[YOU]"
    print(f"{prefix}玩家{player.player_id}{role}[{player.name}] 手牌: {tiles_str}{melds_str}")

    parts = []
    if player.missing_suit: parts.append(f"定缺: {player.missing_suit.value}")
    parts.append(f"余额: {player.balance}")
    if player.is_hu: parts.append("【已胡牌】")
    print(f"   {' | '.join(parts)}")

# --- Bot 辅助函数 ---
def bot_decide_exchange(player: PlayerState) -> List[Tile]:
    suit_map = {Suit.WAN: [], Suit.TIAO: [], Suit.TONG: []}
    for t in player.hand_tiles:
        suit_map[t.suit].append(t)
    
    # 按花色牌数从小到大排序
    sorted_suits = sorted(suit_map.items(), key=lambda item: len(item[1]))
    for suit, tiles in sorted_suits:
        if len(tiles) >= 3:
            return tiles[:3]  # 换出该花色的前3张
    
    # 若所有花色均不足3张，则直接换手牌前三张
    return player.hand_tiles[:3]
    
# def bot_decide_exchange(player: PlayerState) -> List[Tile]:
#     suit_map = {Suit.WAN: [], Suit.TIAO: [], Suit.TONG: []}
#     for t in player.hand_tiles:
#         suit_map[t.suit].append(t)
#     for suit in suit_map:
#         if len(suit_map[suit]) >= 3:
#             return suit_map[suit][:3]
#     return player.hand_tiles[:3]

def bot_decide_missing_suit(player: PlayerState) -> Suit:
    suit_counts = {Suit.WAN: 0, Suit.TIAO: 0, Suit.TONG: 0}
    for tile in player.hand_tiles:
        suit_counts[tile.suit] += 1
    return min(suit_counts, key=suit_counts.get)
#
# def bot_decide_turn_action(player: PlayerState, game: MahjongGame) -> str:
#     if player.can_hu():
#         return "h"
#     gang_info = game.can_self_gang(player.player_id)
#     if gang_info['can_gang']:
#         return "g"
#     if player.missing_suit:
#         missing_tiles = [t for t in player.hand_tiles if t.suit == player.missing_suit]
#         if missing_tiles:
#             return f"d {missing_tiles[0]}"
#     random_tile = random.choice(player.hand_tiles)
#     return f"d {random_tile}"

def print_detailed_settlement(game: MahjongGame, start_balances: List[int]):
    print("\n" + "█" * 30 + " 本局详细结算 " + "█" * 30)
    
    # 1. 找出点炮的玩家 (反向查找：如果赢家的胡牌方式不是自摸，那个 discard_player_id 就是点炮者)
    dian_pao_players = {} # {winner_id: loser_id}
    for p in game.players:
        if p.is_hu and not p.hu_is_self_drawn and p.hu_discard_player_id is not None:
            dian_pao_players[p.player_id] = p.hu_discard_player_id

    for i, p in enumerate(game.players):
        # 计算本局净胜负
        net_score = p.balance - start_balances[i]
        score_str = f"+{net_score}" if net_score >= 0 else f"{net_score}"
        
        # 基础信息
        role = "[BOT]" if p.is_bot else "[YOU]"
        status_tags = []
        details = []

        # --- A. 赢家分析 ---
        if p.is_hu:
            status_tags.append("【胡牌】")
            # 胡牌方式
            if p.hu_is_self_drawn:
                method = "自摸"
            else:
                loser_id = p.hu_discard_player_id
                method = f"捉 玩家{loser_id} 炮"
            
            # 番型详情
            fan_str = ",".join(p.hu_fan_types)
            details.append(f"{method} | {fan_str} | 共{p.hu_fan}番")
        
        # --- B. 输家分析 (未胡牌) ---
        else:
            # 1. 查花猪 (手里还有定缺牌)
            has_missing = any(t.suit == p.missing_suit for t in p.hand_tiles)
            if has_missing:
                status_tags.append("【花猪】")
                details.append("定缺牌未打完，赔付所有非花猪玩家满番")
            
            # 2. 查大叫 (没听牌，且不是花猪)
            else:
                max_fan, _ = p.calculate_potential_fan()
                if max_fan == 0:
                    status_tags.append("【无叫】")
                    if game.deck.remaining_count() == 0: # 只有流局才查大叫
                        details.append("流局未听牌，赔付听牌玩家")
                else:
                    status_tags.append("【听牌】") # 输了钱但听牌了（可能是被自摸或者没等到牌）
                    if game.deck.remaining_count() == 0:
                        details.append(f"手握{max_fan}番，理论最大番")

            # 3. 检查是否点炮
            # 遍历所有赢家，看有没有人胡的是我的牌
            pao_targets = []
            for winner_id, loser_id in dian_pao_players.items():
                if loser_id == p.player_id:
                    pao_targets.append(str(winner_id))
            
            if pao_targets:
                status_tags.append("【点炮】")
                details.append(f"点炮给 -> 玩家 {','.join(pao_targets)}")

        # --- 打印单人条目 ---
        print(f"玩家{p.player_id} {role} {score_str} {' '.join(status_tags)}")
        for d in details:
            print(f"   └─ {d}")
        
    print("-" * 76)
    print(f"当前余额: ", end="")
    for p in game.players:
        print(f"P{p.player_id}:{p.balance}  ", end="")
    print("\n" + "█" * 76 + "\n")

def bot_decide_turn_action(player: PlayerState, game: MahjongGame) -> str:
    """Bot决策：优先胡 > 优先杠 > 贪婪打出向听数最小的牌"""
    # 1. 优先胡牌
    if player.can_hu():
        return "h"

    # 2. 优先杠牌
    gang_info = game.can_self_gang(player.player_id)
    if gang_info['can_gang']:
        return "g"

    # 3. 出牌逻辑优化：贪婪选择
    # 遍历手里的每一张牌，试着打出去，看剩下的牌向听数是多少
    best_tile = None
    min_shanten = 100

    # 先筛选出合法牌（如果有缺门，只看缺门）
    candidates = player.hand_tiles.copy()
    if player.missing_suit:
        missing_tiles = [t for t in player.hand_tiles if t.suit == player.missing_suit]
        if missing_tiles:
            candidates = missing_tiles

    # 如果candidates为空（理论上不应该发生，但做个保护）
    if not candidates:
        candidates = player.hand_tiles.copy()

    # 如果手牌为空（异常情况），返回pass
    if not candidates:
        return "pass"

    # 去重测试，减少计算量
    unique_candidates = list(set(candidates))

    for tile in unique_candidates:
        # 模拟打出这张牌后的手牌
        temp_tiles = player.hand_tiles.copy()
        temp_tiles.remove(tile)

        # 计算向听数
        # 注意：Bot不知道别的，只看手牌结构
        s = ShantenCalculator.calculate_shanten(temp_tiles, player.missing_suit)

        if s < min_shanten:
            min_shanten = s
            best_tile = tile
        # 如果向听数一样，优先打花色数量少的，或者随机（这里保持简单，不改best_tile）

    # 如果没找到（逻辑兜底），随机打
    if best_tile is None:
        if candidates:
            best_tile = random.choice(candidates)
        else:
            # 极端情况：从所有手牌中随机选一张
            best_tile = random.choice(player.hand_tiles)

    return f"d {best_tile}"

def bot_decide_response(player: PlayerState, actions: List[str]) -> str:
    if 'hu' in actions: return 'h'
    if 'gang' in actions: return 'g'
    if 'peng' in actions: return 'p'
    return 'n'


# =============================================================================
# 三种风格 Bot（用于预实验：激进型 / 保守型 / 随机型）
# =============================================================================

def _get_discard_candidates(player: PlayerState):
    """返回合法出牌候选（先打缺门，否则全部手牌）"""
    if player.missing_suit:
        missing = [t for t in player.hand_tiles if t.suit == player.missing_suit]
        if missing:
            return missing
    return player.hand_tiles.copy() if player.hand_tiles else []


def bot_decide_turn_action_aggressive(player: PlayerState, game: MahjongGame) -> str:
    """激进型：胡 > 杠 > 优先打幺九孤张，快速压缩手牌求胡"""
    if player.can_hu():
        return "h"
    gang_info = game.can_self_gang(player.player_id)
    if gang_info['can_gang']:
        return "g"

    candidates = _get_discard_candidates(player)
    if not candidates:
        return "pass"

    # 优先打孤张幺九（1/9），其次打向听数最小的牌
    def score(tile):
        temp = player.hand_tiles.copy()
        temp.remove(tile)
        s = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
        # 幺九孤张额外加分（更倾向打出）
        is_terminal = tile.number in (1, 9)
        cnt = sum(1 for t in player.hand_tiles if t == tile)
        is_isolated = cnt == 1
        bonus = 0.5 if (is_terminal and is_isolated) else 0
        return s - bonus

    best = min(set(candidates), key=score)
    return f"d {best}"


def bot_decide_turn_action_conservative(player: PlayerState, game: MahjongGame) -> str:
    """保守型：胡 > 优先打对手已出的'安全牌'，减少点炮风险"""
    if player.can_hu():
        return "h"
    # 保守型不主动杠（暴露手牌结构）

    candidates = _get_discard_candidates(player)
    if not candidates:
        return "pass"

    # 收集场上所有人的弃牌（安全牌集合）
    safe_tiles = set()
    for p in game.players:
        for t in p.discarded_tiles:
            safe_tiles.add((t.suit, t.number))

    unique_candidates = list(set(candidates))

    # 先尝试在安全牌中找向听数最小的打出
    best_tile = None
    min_shanten = 100

    for tile in unique_candidates:
        temp = player.hand_tiles.copy()
        temp.remove(tile)
        s = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
        is_safe = (tile.suit, tile.number) in safe_tiles
        # 安全牌给向听数一个惩罚加成（让安全牌在同等向听时优先）
        effective_s = s - (1.0 if is_safe else 0)
        if effective_s < min_shanten:
            min_shanten = effective_s
            best_tile = tile

    if best_tile is None:
        best_tile = random.choice(unique_candidates)
    return f"d {best_tile}"


def bot_decide_turn_action_random(player: PlayerState, game: MahjongGame) -> str:
    """随机型：合法范围内随机出牌"""
    if player.can_hu():
        return "h"
    candidates = _get_discard_candidates(player)
    if not candidates:
        return "pass"
    return f"d {random.choice(candidates)}"


def bot_decide_response_aggressive(player: PlayerState, actions: List[str]) -> str:
    """激进型响应：见胡就胡，见杠就杠，见碰就碰"""
    if 'hu' in actions: return 'h'
    if 'gang' in actions: return 'g'
    if 'peng' in actions: return 'p'
    return 'n'


def bot_decide_response_conservative(player: PlayerState, actions: List[str]) -> str:
    """保守型响应：只胡不碰杠（减少暴露）"""
    if 'hu' in actions: return 'h'
    return 'n'


def bot_decide_response_random(player: PlayerState, actions: List[str]) -> str:
    """随机型响应：随机选择"""
    if 'hu' in actions: return 'h'
    return random.choice(['n'] + [a[0] for a in [('p', 'peng'), ('g', 'gang')]
                                  if a[1] in actions])


# 风格名 → (出牌函数, 响应函数) 映射表，方便外部按名字调用
STYLE_BOTS = {
    "greedy":       (bot_decide_turn_action,            bot_decide_response),
    "aggressive":   (bot_decide_turn_action_aggressive, bot_decide_response_aggressive),
    "conservative": (bot_decide_turn_action_conservative, bot_decide_response_conservative),
    "random":       (bot_decide_turn_action_random,     bot_decide_response_random),
}

# --- 主程序 (支持循环对局) ---

# =============================================================================
# 展示模块：打印全场信息 (新增)
# =============================================================================
def print_table_info(game: MahjongGame, current_player_id: int, drawn_tile: Optional[Tile]):
    """展示全场信息：所有玩家（含Bot）的手牌、明牌、弃牌"""
    print("\n" + "-"*35 + f" 剩余牌数: {game.deck.remaining_count()} " + "-"*35)
    
    for p in game.players:
        # 标记当前行动玩家
        prefix = ">>> " if p.player_id == current_player_id else "    "
        role = "[BOT]" if p.is_bot else "[YOU]"
        
        # 状态标记
        status = []
        if p.is_hu: status.append("已胡")
        if p.missing_suit: status.append(f"缺{p.missing_suit.value}")
        status_str = f"({', '.join(status)})" if status else ""
        
        print(f"{prefix}P{p.player_id} {role} {status_str} 余额:{p.balance}")
        
        # 手牌 (上帝视角：全部显示，不再隐藏Bot手牌)
        hand_str = " ".join([str(t) for t in p.hand_tiles])
        print(f"      手牌: {hand_str}")
        
        # 明牌
        if p.open_melds:
            melds_list = ["[" + " ".join(str(t) for t in m) + "]" for m in p.open_melds]
            print(f"      明牌: {' '.join(melds_list)}")
            
        # 弃牌 (显示最近出的几张，避免刷屏)
        if p.discarded_tiles:
            discards = " ".join([str(t) for t in p.discarded_tiles])
            print(f"      弃牌: {discards}")
            
    if drawn_tile:
        print(f"\n>>> 玩家 {current_player_id} 摸牌: 【{drawn_tile}】")
    print("-" * 85)

def run_console_game():
    # 1. 尝试导入本地 Agent
    try:
        from local_llm_agent import LocalLLMAgent
        has_llm = True
    except ImportError:
        has_llm = False
        print("⚠️ 未找到 local_llm_agent.py，将使用全 Bot 模式")

    print("\n" + "="*60 + "\n四川麻将：本地大模型 (Qwen2.5-1.5B) 1v3 规则Bot\n" + "="*60)

    # 2. 配置本地模型路径 (请确保此前下载成功)
    model_path = r"C:\Users\31801\.cache\modelscope\hub\models\qwen\Qwen2___5-1___5B-Instruct"
    
    # 3. 加载模型 (只加载一次！)
    llm_agent = None
    if has_llm:
        try:
            print(f"⏳ 正在加载本地模型: {model_path} ...")
            llm_agent = LocalLLMAgent(model_path)
            print("✅ 本地模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            has_llm = False

    # 4. 配置玩家
    # 假设我们让 P0, P1, P2 都由这个本地模型控制 (共享同一个模型实例，节省显存)
    # P3 是规则 Bot
    player_names = ["Local-Qwen-P0", "Local-Qwen-P1", "Local-Qwen-P2", "Rule-Bot"]
    
    # 将模型实例映射给玩家 ID
    llm_agents = {}
    if has_llm and llm_agent:
        llm_agents[0] = llm_agent
        llm_agents[1] = llm_agent
        llm_agents[2] = llm_agent

    # 这里的 bots 配置为 False 表示不由游戏内部的简单逻辑接管，而是我们在循环里手动调用 LLM
    bots_config = [False, False, False, True] 
    
    global_balances = [10000] * 4
    round_count = 1

    while True:
        print(f"\n>>> 第 {round_count} 局初始化...")
        game = MahjongGame("G1", player_names, bots_config)
        for i, p in enumerate(game.players): p.balance = global_balances[i]
        start_balances = global_balances.copy()
        
        game.start_game()

        # --- 快速开局 (换三张/定缺) ---
        # 为节省时间，开局阶段暂时都用规则处理，核心的出牌阶段再交给 LLM
        game.phase = GamePhase.EXCHANGE
        for p in game.players:
            game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))
        
        game.phase = GamePhase.CHOOSE_MISSING
        for p in game.players:
            if p.player_id in llm_agents:
                # LLM 玩家也打印一下定缺
                s = bot_decide_missing_suit(p)
                print(f"[{player_names[p.player_id]}] 规则定缺: {s.value}")
                game.set_missing_suit(p.player_id, s)
            else:
                game._auto_select_missing_suit(p)
        
        game.phase = GamePhase.PLAYING
        print("\n>>> 对局开始")
        
        skip_draw = True # 庄家第一轮不摸牌
        step_count = 0

        while not game.is_game_over:
            # 防死锁熔断
            step_count += 1
            if step_count > 300:
                print("⚠️ 强制熔断：步数过多")
                break
            if sum(1 for p in game.players if p.is_hu) >= 3:
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
            
            # 打印场面
            if pid in llm_agents:
                print_table_info(game, pid, drawn)

            turn_end = False
            loop_retry = 0

            while not turn_end:
                loop_retry += 1
                force_bot = loop_retry > 3
                action = ""

                # === 决策 ===
                if pid in llm_agents and not force_bot:
                    # 构造合法动作列表
                    valid_acts = []
                    if player.can_hu(): valid_acts.append("h")
                    gang = game.can_self_gang(pid)
                    if gang['can_gang']: valid_acts.append("g")
                    
                    # 出牌列表
                    miss = any(t.suit == player.missing_suit for t in player.hand_tiles)
                    seen = set()
                    for t in player.hand_tiles:
                        if miss and t.suit != player.missing_suit: continue
                        if str(t) not in seen:
                            valid_acts.append(f"d {str(t)}")
                            seen.add(str(t))
                    
                    try:
                        # 调用本地模型推理
                        action = llm_agents[pid].get_action(player, game, valid_acts)
                        print(f"[{player_names[pid]}] 思考后决定: {action}")
                    except Exception as e:
                        print(f"推理出错: {e}")
                        action = bot_decide_turn_action(player, game)
                else:
                    action = bot_decide_turn_action(player, game)
                    if pid in llm_agents: print(f"[{player_names[pid]}] (规则接管): {action}")

                # === 执行 ===
                if action == 'h':
                    if player.can_hu():
                        win = drawn if drawn else player.hand_tiles[-1]
                        f, t = game.hu(pid, win, True)
                        print(f"🎉 玩家{pid} 自摸! {','.join(t)} {f}番")
                        if game.check_game_over(True): 
                            turn_end=True; break
                        turn_end=True; game.next_player(); skip_draw=False
                    else: pass

                elif action == 'g':
                    g_info = game.can_self_gang(pid)
                    if g_info['can_gang']:
                        succ, new_d = game.gang(pid, g_info['gang_tiles'][0])
                        if succ:
                            print(f"玩家{pid} 杠牌")
                            if new_d: drawn=new_d
                        continue
                    else: pass

                elif action.startswith('d '):
                    t = parse_console_tile(action[2:])
                    if t and game.discard_tile(pid, t):
                        print(f"玩家{pid} 打出 {t}")
                        
                        # 响应
                        res = game.check_responses(t, pid)
                        responded = False
                        if res:
                            for rid, acts in res.items():
                                if responded: break
                                r_player = game.players[rid]
                                r_choice = 'n'
                                
                                # LLM 响应
                                if rid in llm_agents:
                                    v_r = ['n']
                                    if 'hu' in acts: v_r.append('h')
                                    if 'gang' in acts: v_r.append('g')
                                    if 'peng' in acts: v_r.append('p')
                                    try:
                                        print(f" -> 询问 P{rid} 是否响应 {v_r}...")
                                        r_choice = llm_agents[rid].get_action(r_player, game, v_r)
                                        print(f" -> P{rid} 回复: {r_choice}")
                                    except: r_choice='n'
                                else:
                                    r_choice = bot_decide_response(r_player, acts)
                                
                                # 执行响应
                                if r_choice=='h' and 'hu' in acts:
                                    f, types = game.hu(rid, t, False, pid)
                                    print(f"⚡ 玩家{rid} 胡牌! (点炮:P{pid}) {','.join(types)}")
                                    game.check_game_over()
                                    responded = True
                                    if game.check_game_over(True): turn_end=True; break
                                elif r_choice=='g' and 'gang' in acts:
                                    game.gang(rid, t, pid)
                                    print(f"玩家{rid} 直杠")
                                    game.current_player_id = rid
                                    turn_end=True; responded=True; skip_draw=True
                                elif r_choice=='p' and 'peng' in acts:
                                    game.peng(rid, t, pid)
                                    print(f"玩家{rid} 碰")
                                    game.current_player_id = rid
                                    turn_end=True; responded=True; skip_draw=True
                        
                        if game.is_game_over: break
                        if not responded:
                            game.next_player(); skip_draw=False
                        turn_end = True
                    else: pass

        game.check_game_over()
        print_detailed_settlement(game, start_balances)
        for i, p in enumerate(game.players): global_balances[i] = p.balance
        
        c = input(">>> 继续? (Enter/n): ").strip().lower()
        if c == 'n': break
        round_count += 1

if __name__ == '__main__':
    run_console_game()

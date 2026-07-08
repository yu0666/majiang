"""
四川麻将规则引擎：胡牌判断、番数计算、向听数计算
"""
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from tiles import Tile, Suit

# ==================== 向听数计算器 (修复版) ====================
class ShantenCalculator:
    """
    四川麻将向听数计算器
    向听数定义:
      -1 = 胡牌
       0 = 听牌 (差1张胡)
       1 = 一向听 (差2张听牌)
       ...
       8 = 最大向听数
    
    计算公式: 向听数 = 8 - 2*面子数 - 搭子数 - (有将?1:0)
    """
    
    @staticmethod
    def calculate_shanten(tiles: List[Tile], missing_suit: Optional[Suit] = None) -> int:
        """
        计算向听数
        改进：如果手牌中有定缺花色的牌，向听数 += 定缺牌数量
        这样可以引导模型优先打掉定缺的牌
        """
        # 1. 分离定缺牌和有效牌
        missing_count = 0
        valid_tiles = []
        
        if missing_suit:
            for t in tiles:
                if t.suit == missing_suit:
                    missing_count += 1
                else:
                    valid_tiles.append(t)
        else:
            valid_tiles = list(tiles)
        
        # 如果所有牌都是定缺花色，返回极大值
        if not valid_tiles:
            return 8 + missing_count
        
        # 2. 计算有效牌的基础向听数
        base_shanten = ShantenCalculator._calculate_base_shanten(valid_tiles)
        
        # 3. 加上定缺牌惩罚 (每张定缺牌+1向听)
        return base_shanten + missing_count
    
    @staticmethod
    def _calculate_base_shanten(tiles: List[Tile]) -> int:
        """计算基础向听数（不考虑定缺）"""
        n = len(tiles)
        
        # 特殊情况：检查是否已经胡牌
        if n == 14 or n == 11 or n == 8 or n == 5 or n == 2:
            if HandPattern(tiles).is_winning_hand():
                return -1
        
        # 检查七对向听
        if n >= 13:
            seven_pairs_shanten = ShantenCalculator._seven_pairs_shanten(tiles)
        else:
            seven_pairs_shanten = 8
        
        # 计算标准向听 (面子+将)
        standard_shanten = ShantenCalculator._standard_shanten(tiles)
        
        return min(seven_pairs_shanten, standard_shanten)
    
    @staticmethod
    def _seven_pairs_shanten(tiles: List[Tile]) -> int:
        """计算七对向听数"""
        if len(tiles) < 13:
            return 8
        
        counter = Counter((t.suit, t.number) for t in tiles)
        pairs = 0
        singles = 0
        
        for count in counter.values():
            pairs += count // 2
            singles += count % 2
        
        # 七对需要7个对子
        # 向听数 = 6 - 对子数 + 单张修正
        if pairs > 7:
            pairs = 7
        
        shanten = 6 - pairs + (singles if singles > 0 else 0)
        
        # 修正：如果牌数不对，向听数要调整
        if len(tiles) == 13:
            return max(-1, 6 - pairs)
        elif len(tiles) == 14:
            if pairs >= 7:
                return -1
            return 6 - pairs
        
        return min(8, shanten)
    
    @staticmethod  
    def _standard_shanten(tiles: List[Tile]) -> int:
        """计算标准胡牌向听数 (4面子+1将)"""
        # 按花色分组
        suits = {Suit.WAN: [], Suit.TIAO: [], Suit.TONG: []}
        for t in tiles:
            suits[t.suit].append(t.number)
        
        for s in suits:
            suits[s].sort()
        
        # 需要的面子数 (根据牌数)
        n = len(tiles)
        need_melds = (n - 2) // 3  # 去掉将牌后需要的面子数
        
        # 搜索最优解
        best_shanten = 8
        
        # 遍历所有可能的将牌
        counter = Counter((t.suit, t.number) for t in tiles)
        tried_pairs = set()
        
        for tile_key, count in counter.items():
            if count >= 2 and tile_key not in tried_pairs:
                tried_pairs.add(tile_key)
                # 选这个对子做将
                temp_suits = {s: list(suits[s]) for s in suits}
                suit, num = tile_key
                temp_suits[suit].remove(num)
                temp_suits[suit].remove(num)
                
                # 计算剩余牌的面子和搭子
                total_melds = 0
                total_partials = 0
                
                for s in Suit:
                    if s in temp_suits:
                        melds, partials = ShantenCalculator._count_melds_partials(temp_suits[s])
                        total_melds += melds
                        total_partials += partials
                
                # 搭子不能超过 (需要面子数 - 已有面子数)
                useful_partials = min(total_partials, need_melds - total_melds)
                
                # 向听数 = 需要面子数 - 2*已有面子 - 搭子 - 1(有将)
                shanten = (need_melds - total_melds) * 2 - total_melds - useful_partials - 1
                # 简化公式: shanten = need_melds - 2*melds - partials - 1
                shanten = need_melds - 2 * total_melds - useful_partials - 1
                
                # 修正计算
                shanten = 8 - 2 * total_melds - useful_partials - 1
                shanten = max(-1, min(8, shanten))
                
                if shanten < best_shanten:
                    best_shanten = shanten
        
        # 也考虑没有将的情况（单钓）
        total_melds = 0
        total_partials = 0
        for s in Suit:
            if s in suits:
                melds, partials = ShantenCalculator._count_melds_partials(suits[s])
                total_melds += melds
                total_partials += partials
        
        useful_partials = min(total_partials, 4 - total_melds)
        shanten = 8 - 2 * total_melds - useful_partials
        shanten = max(-1, min(8, shanten))
        
        if shanten < best_shanten:
            best_shanten = shanten
        
        return best_shanten
    
    @staticmethod
    def _count_melds_partials(cards: List[int]) -> Tuple[int, int]:
        """
        计算一个花色中的面子数和搭子数
        使用回溯搜索找最优解
        """
        if not cards:
            return 0, 0
        
        cards = sorted(cards)
        best_result = [0, 0]  # [melds, partials]
        
        ShantenCalculator._search_melds(cards, 0, 0, best_result)
        
        return best_result[0], best_result[1]
    
    @staticmethod
    def _search_melds(cards: List[int], melds: int, partials: int, best: List[int]):
        """回溯搜索面子和搭子"""
        if not cards:
            # 更新最优解 (优先最大化面子，其次最大化搭子)
            if melds > best[0] or (melds == best[0] and partials > best[1]):
                best[0] = melds
                best[1] = partials
            return
        
        # 当前牌
        c = cards[0]
        remaining = cards[1:]
        
        # 选项1: 尝试刻子 (c, c, c)
        if cards.count(c) >= 3:
            new_cards = list(cards)
            for _ in range(3):
                new_cards.remove(c)
            ShantenCalculator._search_melds(new_cards, melds + 1, partials, best)
        
        # 选项2: 尝试顺子 (c, c+1, c+2)
        if (c + 1) in cards and (c + 2) in cards:
            new_cards = list(cards)
            new_cards.remove(c)
            new_cards.remove(c + 1)
            new_cards.remove(c + 2)
            ShantenCalculator._search_melds(new_cards, melds + 1, partials, best)
        
        # 选项3: 尝试对子搭子 (c, c)
        if cards.count(c) >= 2:
            new_cards = list(cards)
            new_cards.remove(c)
            new_cards.remove(c)
            ShantenCalculator._search_melds(new_cards, melds, partials + 1, best)
        
        # 选项4: 尝试连张搭子 (c, c+1)
        if (c + 1) in cards:
            new_cards = list(cards)
            new_cards.remove(c)
            new_cards.remove(c + 1)
            ShantenCalculator._search_melds(new_cards, melds, partials + 1, best)
        
        # 选项5: 尝试坎张搭子 (c, c+2)
        if (c + 2) in cards:
            new_cards = list(cards)
            new_cards.remove(c)
            new_cards.remove(c + 2)
            ShantenCalculator._search_melds(new_cards, melds, partials + 1, best)
        
        # 选项6: 跳过这张牌（作为废牌）
        ShantenCalculator._search_melds(remaining, melds, partials, best)

# ==================== 原有类保持不变 ====================

class HandPattern:
    """手牌牌型分析类"""
    def __init__(self, tiles: List[Tile]):
        self.tiles = tiles
        self.tile_count = len(tiles)

    def is_winning_hand(self) -> bool:
        if self.tile_count % 3 != 2:
            return False
        if self.tile_count == 14 and self._is_seven_pairs():
            return True
        return self._is_standard_win()

    def _is_seven_pairs(self) -> bool:
        if self.tile_count != 14:
            return False
        counter = Counter((t.suit, t.number) for t in self.tiles)
        pair_count = 0
        for count in counter.values():
            if count == 2:
                pair_count += 1
            elif count == 4:
                pair_count += 2
            else:
                return False
        return pair_count == 7

    def _is_standard_win(self) -> bool:
        counter = Counter((t.suit, t.number) for t in self.tiles)
        for tile_key, count in counter.items():
            if count >= 2:
                temp_counter = counter.copy()
                temp_counter[tile_key] -= 2
                if self._can_form_melds(temp_counter):
                    return True
        return False

    def _can_form_melds(self, counter: Counter) -> bool:
        if not counter or all(v == 0 for v in counter.values()):
            return True
        remaining_keys = sorted([k for k, v in counter.items() if v > 0], key=lambda x: (x[0].value, x[1]))
        if not remaining_keys: return True
        tile_key = remaining_keys[0]
        suit, number = tile_key
        if counter[tile_key] >= 3:
            temp = counter.copy()
            temp[tile_key] -= 3
            if self._can_form_melds(temp): return True
        if number <= 7:
            key2 = (suit, number + 1)
            key3 = (suit, number + 2)
            if counter.get(key2, 0) >= 1 and counter.get(key3, 0) >= 1:
                temp = counter.copy()
                temp[tile_key] -= 1
                temp[key2] -= 1
                temp[key3] -= 1
                if self._can_form_melds(temp): return True
        return False

    def is_ready_hand(self) -> Tuple[bool, List[Tile]]:
        if self.tile_count % 3 != 1:
            return False, []
        waiting_tiles = []
        for suit in Suit:
            for number in range(1, 10):
                test_tile = Tile(suit, number)
                test_hand = self.tiles + [test_tile]
                if HandPattern(test_hand).is_winning_hand():
                    waiting_tiles.append(test_tile)
        return len(waiting_tiles) > 0, waiting_tiles


class FanCalculator:
    """番数计算器"""
    @staticmethod
    def calculate_fan(tiles: List[Tile],
                      win_tile: Tile,
                      open_melds: List[List[Tile]] = None,
                      is_self_drawn: bool = True,
                      gang_count: int = 0,
                      concealed_kong_count: int = 0,
                      is_last_tile: bool = False,
                      is_kong_draw: bool = False,
                      is_robbing_kong: bool = False,
                      is_tian_hu: bool = False,
                      is_di_hu: bool = False) -> Tuple[int, List[str]]:

        fan = 0
        fan_types = []

        full_tiles = tiles.copy()
        if open_melds:
            for meld in open_melds:
                full_tiles.extend(meld)

        is_pure = FanCalculator._is_pure_suit(full_tiles)
        is_seven = FanCalculator._is_seven_pairs(tiles)

        if is_seven:
            luxury_count = FanCalculator._count_luxury_pairs(tiles)
            if luxury_count >= 2:
                base = 6 if is_pure else 4
                name = "清双豪七对" if is_pure else "豪华七对"
                fan += base
                fan_types.append(f"{name}({base}番)")
            elif luxury_count == 1:
                base = 5 if is_pure else 3
                name = "清龙七对" if is_pure else "龙七对"
                fan += base
                fan_types.append(f"{name}({base}番)")
            else:
                base = 4 if is_pure else 2
                name = "清七对" if is_pure else "七对"
                fan += base
                fan_types.append(f"{name}({base}番)")
        else:
            fan += 0
            fan_types.append("平胡(0番)")
            if is_pure:
                fan += 2
                fan_types.append("清一色(2番)")
            if FanCalculator._is_all_pungs(full_tiles):
                fan += 1
                fan_types.append("碰碰胡(1番)")
            if FanCalculator._is_dai_yao(full_tiles):
                fan += 2
                fan_types.append("带幺九(2番)")
            root_count = FanCalculator._count_roots(full_tiles)
            if root_count > 0:
                fan += root_count
                fan_types.append(f"根×{root_count}({root_count}番)")

        if is_self_drawn:
            fan += 1
            fan_types.append("自摸(1番)")
        if is_tian_hu:
            fan += 8
            fan_types.append("天胡(8番)")
        if is_di_hu:
            fan += 6
            fan_types.append("地胡(6番)")
        if is_kong_draw:
            fan += 1
            fan_types.append("杠上花(1番)")
        if is_last_tile:
            fan += 1
            fan_types.append("海底捞月(1番)")
        if is_robbing_kong:
            fan += 1
            fan_types.append("抢杠胡(1番)")
        if FanCalculator._is_single_wait(tiles, win_tile):
            fan += 2
            fan_types.append("金钩钓(2番)")
        return fan, fan_types

    @staticmethod
    def _is_dai_yao(tiles: List[Tile]) -> bool:
        for t in tiles:
            if t.number in [4, 5, 6]: return False
        counter = Counter((t.suit, t.number) for t in tiles)
        for tile_key, count in counter.items():
            suit, number = tile_key
            if count >= 2 and number in [1, 9]:
                temp_counter = counter.copy()
                temp_counter[tile_key] -= 2
                if FanCalculator._can_form_dai_yao_melds(temp_counter):
                    return True
        return False

    @staticmethod
    def _can_form_dai_yao_melds(counter: Counter) -> bool:
        if not counter or all(v == 0 for v in counter.values()): return True
        remaining_keys = sorted([k for k, v in counter.items() if v > 0], key=lambda x: (x[0].value, x[1]))
        if not remaining_keys: return True
        tile_key = remaining_keys[0]
        suit, number = tile_key
        if number in [1, 9] and counter[tile_key] >= 3:
            temp = counter.copy()
            temp[tile_key] -= 3
            if FanCalculator._can_form_dai_yao_melds(temp): return True
        if number == 1 or number == 7:
            key2 = (suit, number + 1)
            key3 = (suit, number + 2)
            if counter.get(key2, 0) >= 1 and counter.get(key3, 0) >= 1:
                temp = counter.copy()
                temp[tile_key] -= 1
                temp[key2] -= 1
                temp[key3] -= 1
                if FanCalculator._can_form_dai_yao_melds(temp): return True
        return False

    @staticmethod
    def _is_seven_pairs(tiles: List[Tile]) -> bool:
        if len(tiles) != 14: return False
        counter = Counter((t.suit, t.number) for t in tiles)
        return sum(1 for c in counter.values() if c == 2) + sum(1 for c in counter.values() if c == 4) * 2 == 7

    @staticmethod
    def _count_roots(tiles: List[Tile]) -> int:
        counter = Counter((t.suit, t.number) for t in tiles)
        return sum(1 for c in counter.values() if c == 4)

    @staticmethod
    def _count_luxury_pairs(tiles: List[Tile]) -> int:
        return FanCalculator._count_roots(tiles)

    @staticmethod
    def _is_pure_suit(tiles: List[Tile]) -> bool:
        suits = set(t.suit for t in tiles)
        return len(suits) == 1

    @staticmethod
    def _is_all_pungs(tiles: List[Tile]) -> bool:
        counter = Counter((t.suit, t.number) for t in tiles)
        pair_count = sum(1 for c in counter.values() if c == 2)
        return pair_count == 1 and all(c in [2,3,4] for c in counter.values())

    @staticmethod
    def _is_single_wait(tiles: List[Tile], win_tile: Tile) -> bool:
        return len(tiles) == 2

def check_missing_suit(tiles: List[Tile], missing_suit: Optional[Suit]) -> bool:
    if missing_suit is None: return True
    suits_in_hand = set(t.suit for t in tiles)
    return missing_suit not in suits_in_hand

def detect_flower_pig(tiles: List[Tile]) -> bool:
    suits = set(t.suit for t in tiles)
    return len(suits) == 3
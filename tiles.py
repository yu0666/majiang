"""
麻将牌定义和基础操作
"""
# 导入枚举类型，用于定义麻将花色的枚举类
from enum import Enum
# 导入类型提示模块，List用于标注列表类型，Tuple用于元组类型（此处备用）
from typing import List, Tuple
# 导入随机数模块，用于实现洗牌功能
import random


class Suit(Enum):
    """花色枚举：定义麻将的三种基础花色"""
    WAN = "万"  # 万子花色
    TIAO = "条"  # 条子花色
    TONG = "筒"  # 筒子花色


class Tile:
    """麻将牌核心类：描述单张麻将牌的属性和行为"""
    # 构造方法：初始化单张麻将牌，参数为花色（Suit枚举）和数字（1-9）
    def __init__(self, suit: Suit, number: int):
        # 校验牌号合法性：麻将牌数字只能是1-9
        if not 1 <= number <= 9:
            # 数字超出范围时抛出值错误异常
            raise ValueError("牌号必须在1-9之间")
        self.suit = suit  # 实例属性：麻将牌的花色（Suit枚举类型）
        self.number = number  # 实例属性：麻将牌的数字（1-9）

    # 重写等于运算符：用于判断两张麻将牌是否为同一张牌
    def __eq__(self, other):
        # 先判断另一个对象是否为Tile类的实例
        if not isinstance(other, Tile):
            return False  # 非Tile实例直接返回不相等
        # 花色和数字都相同，才判定为同一张牌
        return self.suit == other.suit and self.number == other.number

    # 重写哈希方法：使Tile实例可被哈希（如用作字典的键、放入集合）
    def __hash__(self):
        # 基于花色和数字生成唯一哈希值
        return hash((self.suit, self.number))

    # 重写repr方法：调试/交互式环境下的字符串表示（如控制台打印实例）
    def __repr__(self):
        # 返回"数字+花色"格式（如"3万"）
        return f"{self.number}{self.suit.value}"

    # 重写str方法：print()打印或字符串转换时的显示格式
    def __str__(self):
        # 返回"数字+花色"的人性化格式（如"5条"）
        return f"{self.number}{self.suit.value}"

    # 实例方法：将Tile对象转为字典，便于序列化/数据传输
    def to_dict(self):
        # 返回包含花色（字符）和数字的字典
        return {"suit": self.suit.value, "number": self.number}


class TileDeck:
    """牌堆管理类：负责麻将牌堆的初始化、洗牌、摸牌等操作"""
    # 构造方法：初始化牌堆对象
    def __init__(self):
        # 实例属性：存储牌堆中所有麻将牌的列表（类型标注为Tile列表）
        self.tiles: List[Tile] = []
        # 调用初始化方法，生成完整的108张麻将牌
        self.initialize()

    # 初始化牌堆方法：生成标准的108张麻将牌（万/条/筒各1-9，每种4张）
    def initialize(self):
        """初始化108张牌（每种牌4张）"""
        self.tiles = []  # 清空牌堆列表（避免重复初始化）
        for suit in Suit:  # 遍历三种花色（万、条、筒）
            for number in range(1, 10):  # 遍历每个花色的1-9数字
                for _ in range(4):  # 每种牌生成4张
                    # 创建Tile实例并添加到牌堆
                    self.tiles.append(Tile(suit, number))

    # 洗牌方法：打乱牌堆中麻将牌的顺序
    def shuffle(self):
        """洗牌"""
        # 使用random模块的shuffle方法原地打乱牌堆列表
        random.shuffle(self.tiles)

    # 摸牌方法：从牌堆顶部摸取指定数量的牌（默认摸1张）
    def draw(self, count: int = 1) -> List[Tile]:
        """摸牌"""
        # 校验摸牌数量是否超过剩余牌数
        if count > len(self.tiles):
            # 数量超出时抛出值错误，提示剩余牌数
            raise ValueError(f"牌堆不足，剩余{len(self.tiles)}张")
        # 从牌堆头部取出指定数量的牌
        drawn = self.tiles[:count]
        # 更新牌堆：移除已摸出的牌
        self.tiles = self.tiles[count:]
        # 返回摸出的牌列表
        return drawn

    # 获取剩余牌数方法：返回牌堆中剩余的麻将牌数量
    def remaining_count(self) -> int:
        """剩余牌数"""
        # 返回牌堆列表的长度（即剩余牌数）
        return len(self.tiles)


# 解析麻将牌字符串的函数：将"3万"这类字符串转为Tile实例
def parse_tile(tile_str: str) -> Tile:
    """解析牌字符串，如 '3万' -> Tile(Suit.WAN, 3)"""
    # 校验字符串长度：至少包含数字+花色（如"3万"是2位）
    if len(tile_str) < 2:
        raise ValueError(f"无效的牌字符串: {tile_str}")

    # 提取字符串第一位作为数字（转为整型）
    number = int(tile_str[0])
    # 提取字符串第二位作为花色字符
    suit_char = tile_str[1]

    # 花色字符到Suit枚举的映射字典（便于快速转换）
    suit_map = {"万": Suit.WAN, "条": Suit.TIAO, "筒": Suit.TONG}
    # 校验花色字符是否合法
    if suit_char not in suit_map:
        raise ValueError(f"未知花色: {suit_char}")

    # 根据映射创建并返回Tile实例
    return Tile(suit_map[suit_char], number)


# 牌列表转字符串函数：将Tile列表转为易读的字符串
def tiles_to_str(tiles: List[Tile]) -> str:
    """将牌列表转为字符串"""
    # 遍历牌列表，将每个Tile转为字符串后用逗号分隔拼接
    return ", ".join(str(t) for t in tiles)


# 手牌排序函数：按花色和数字对麻将牌列表排序
def sort_tiles(tiles: List[Tile]) -> List[Tile]:
    """排序手牌"""
    # 排序规则：先按花色字符（万/条/筒）排序，再按数字排序
    return sorted(tiles, key=lambda t: (t.suit.value, t.number))
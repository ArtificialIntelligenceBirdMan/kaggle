from sympy.combinatorics import Permutation
from typing import Union, List, Tuple
from abc import abstractmethod

import tensorflow as tf

class PuzzleSubType:
    STANDARD = "S"      # for puzzle [A;A;A;A;B;B;B;B;...]
    CROSS = "C"         # for puzzle [A;B;A;B;...]
    SEQUENTIAL = "N"    # for puzzle [N0;N1;N2;N3;...]


# Puzzle 基类
class Puzzle:
    @abstractmethod
    def __init__(self) -> None:
        pass
   
    # 将 字符状态 转换为 数值向量形式
    @staticmethod
    def to_state_vec(state : Union[List[str], Tuple[str]], sub_type : str=PuzzleSubType.STANDARD):
        if sub_type == PuzzleSubType.SEQUENTIAL:
            return [int(s[1:]) for s in state]
        else:
            return [ord(s) - ord("A") for s in state]
    
    # 将 数值向量形式的状态 转换为 字符形式
    @staticmethod
    def to_state_str(state : Union[List[int], Tuple[int]], sub_type : str=PuzzleSubType.STANDARD):
        if sub_type == PuzzleSubType.SEQUENTIAL:
            return [f"N{i}" for i in state]
        else:
            return [chr(s + ord("A")) for s in state]

class PuzzleInfo(Puzzle):
    def __init__(self, puzzle_type : str, goal_state : str, sub_type : str=PuzzleSubType.STANDARD) -> None:
        self.puzzle_type = puzzle_type
        self.sub_type = sub_type
        
        self.goal_state = goal_state
        self.goal_state_vec = self.to_state_vec(goal_state.split(";"), sub_type)
        
        # 保存 状态长度 state_length 和 每个位置的 状态深度 state_depth
        self.state_length = len(goal_state.split(";"))
        self.state_depth = max(self.goal_state_vec) + 1 # 从 0 开始计数

class PuzzleAction(Puzzle):
    def __init__(self, puzzle_type : str, moves : dict):
        self.puzzle_type = puzzle_type
        self.moves = moves
        # 将 moves 转换为 Permutation 对象
        self.actions = {}
        # 保存 tensorflow Permutation 算子
        self.tf_actions = {}

        # maps from action to index
        self.idx_to_act, self.act_to_idx = {}, {}

        for i, (k,p) in enumerate(moves.items()):
            self.actions[k] = Permutation(p)
            self.tf_actions[k] = tf.constant(p, dtype=tf.int32)
            # the inverse move
            self.actions[f"-{k}"] = ~Permutation(p)
            self.tf_actions[f"-{k}"] = tf.constant((~Permutation(p)).array_form, dtype=tf.int32)

            self.idx_to_act[2*i], self.idx_to_act[2*i + 1] = k, f"-{k}"
            self.act_to_idx[k], self.act_to_idx[f"-{k}"] = 2*i, 2*i + 1
    
    def take_action(self, state : Union[List[int], Tuple[int]], act : Union[str, int]):
        if isinstance(act, int):
            act = self.idx_to_act[act]
        
        return self.actions[act](state)

    def __len__(self):
        return len(self.actions)

class PuzzleNode:
    def __init__(self, state : List[int], parent=None, 
                 act_str=None, act=None, g=0, h=0, lambd : float = 0.8):
        self.state : List[int] = state
        self.parent : PuzzleNode = parent
        self.act_str : str = act_str
        self.act = act

        self.g = g
        self.h = h
        self.lambd = lambd

        # 创建 state_key 作为字典的 key
        self.state_key = ";".join([str(i) for i in self.state])
    
    @property
    def cost(self):
        # 计算 total cost = lambda * g + h
        return self.g * self.lambd + self.h

    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        return self.state == other.state

def get_puzzle_sub_type(puzzle_type : str, goal_state : list):
    if puzzle_type.startswith("cube"):
        if goal_state[0] and goal_state[1] == "A":
            sub_type = PuzzleSubType.STANDARD
        elif goal_state[0] == "A" and goal_state[1] == "B":
            sub_type = PuzzleSubType.CROSS
        elif goal_state[0] == "N0":
            sub_type = PuzzleSubType.SEQUENTIAL
        else:
            raise ValueError("Unknown goal state")
    else:
        if goal_state[0] == "N0":
            sub_type = PuzzleSubType.SEQUENTIAL
        else:
            sub_type = PuzzleSubType.STANDARD
    
    return sub_type
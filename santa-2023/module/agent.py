import tensorflow as tf
from tensorflow import keras
import numpy as np
from sympy.combinatorics import Permutation

from typing import Union, List, Tuple, Dict, Any, Optional
from abc import abstractmethod
from copy import deepcopy
import os
from datetime import datetime
from heapq import heappush, heappop

from .puzzle import PuzzleAction, PuzzleInfo, PuzzleNode
from .model import DeepCube, PuzzleDataset
from .utils import Animator, Logger

class Agent:
    def __init__(self, puzzle_info : PuzzleInfo=None, puzzle_act : PuzzleAction=None, cost_model : tf.keras.Model=None, name : str=None) -> None:
        # 保存 puzzle 信息和动作集合
        self.puzzle : PuzzleInfo = puzzle_info
        self.actions : PuzzleAction = puzzle_act
        # 度量到目标状态的代价函数 
        self.cost_model : tf.keras.Model = cost_model
        
        # Agent 的名称，如果为 None，则使用当前时间（年-月-日-分）作为名称
        self.name = name if name is not None else datetime.now().strftime("%Y-%m-%d-%H-%M")

        # 保存 tensorflow actions 算子，便于在其他函数中使用
        self.tf_acts = tf.stack(list(self.actions.tf_actions.values()))
        # 保存 Agent 的 action 集合
        self.act_str = list(self.actions.tf_actions.keys())

    def sample(self, n: int) -> List[int]:
        pass

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def bfs_search(self, init_state : tf.Tensor):
        """
        做深度为 1 的 BFS 搜索，利用 cost_model 找到每个当前状态的最优 action
        
        Parameters
        ----------
        init_state : tf.Tensor
            当前状态，形状为 (batch_size, state_len)
        """
        
        # 保存移动后的状态
        num_of_act = len(self.actions)
        batch_size = tf.shape(init_state)[0]

        # 扩展 init_state 和 tf_actions 的维度为 (batch_size, num_of_act, state_len)
        init_state = tf.tile(init_state[:, None, :], [1, num_of_act, 1])
        tf_actions = tf.tile(tf.expand_dims(self.tf_acts, axis=0), [batch_size, 1, 1])

        # 操作后的状态
        next_state = tf.gather(init_state, tf_actions, batch_dims=2)
        next_state = tf.reshape(next_state, [batch_size * num_of_act, -1]) # 合并前两个维度

        # 计算代价
        cost_pred = self.cost_model(next_state) # (batch_size * num_of_act, 1)

        # 如果 next_state 是 goal_state，cost = 0
        goal_state = tf.constant(self.puzzle.goal_state_vec, dtype=tf.int32)[None, :]
        is_goal = tf.reduce_all(tf.equal(next_state, goal_state), axis=1, keepdims=True)
        cost_pred = tf.where(is_goal, tf.zeros_like(cost_pred), cost_pred)

        # 将 cost_pred 的维度还原为 (batch_size, num_of_act)
        cost_pred = tf.reshape(cost_pred, [batch_size, num_of_act])

        # 求解最优 action
        best_action = tf.argmin(cost_pred, axis=1, output_type=tf.int32) # (batch_size, )
        best_cost = tf.reduce_min(cost_pred, axis=1, keepdims=True) # (batch_size, )

        return best_action, best_cost

    def Astar_search(self, init_state : List, goal_state : List, lamda : float=0.8, N : int=1) -> List[str]:
        """
        Batch Weighted A* search algorithm, 用于寻找从 init_state 到 goal_state 的最优路径.

        Parameters
        ----------
        init_state : List
            The initial state of the puzzle.
        goal_state : List
            The goal state of the puzzle.
        lamda : float, default = 0.8
            The weight of cost path used.
        N : int, default = 1
            The number of possible paths to be searched in parallel.
        """
        # 初始化优先队列
        open_set = []
        init_node = PuzzleNode(state=init_state, 
                            g=0, h=self.heuristic(init_state), lambd=lamda)
        heappush(open_set, init_node)
        
        # 初始化 close_set
        close_set = {}

        goal_state_key = ";".join([str(i) for i in goal_state])


        while open_set:
            # 取出 open_set 中的最小值
            node = heappop(open_set)

            # 如果 node 是目标状态
            if node.state_key == goal_state_key:
                act_path = []
                act_str_path = []
                while node.parent:
                    act_path.append(node.action)
                    act_str_path.append(node.act_str)
                    node = node.parent
                act_path.reverse()
                act_str_path.reverse()

                return act_path, act_str_path

            # 将节点的邻居扩展到 open_set 中
            next_states = tf.gather(node.state, self.tf_acts)
            # 神经网络的推理最消耗时间，尽可能一次性并行推理出所有状态
            next_h = self.heuristic(next_states)
            
            for act, next_state, h in zip(self.act_str, next_states, next_h):
                next_node = PuzzleNode(state=next_state.numpy().tolist(), parent=node, 
                                    act_str=act, action=self.actions.actions[act], 
                                    g=node.g+1, h=h, lambd=lamda)
                
                # 检查 next_node 是否在 close_set 中
                state_key = next_node.state_key
                if state_key in close_set:
                    # 比较 next_node 和 close_set 中的节点的 path cost g
                    # 如果 next_node 的 path cost 更小，那么将该状态从 close_set 中移除，并加入 open_set
                    if next_node.g < close_set[state_key].g:
                        heappush(open_set, next_node)
                        del close_set[state_key]
                else:
                    heappush(open_set, next_node)

            # 将 node 加入 close_set
            close_set[node.state_key] = node

    
    def heuristic(self, state : Union[List, tf.Tensor]) -> float:
        """
        The heuristic function used in A* search algorithm.

        Parameters
        ----------
        state : List
            The current state of the puzzle.
        """
        if isinstance(state, list):
            state = tf.constant(state, dtype=tf.int32)
        
        # 如果维度为 1，添加 batch 维度
        if len(state.shape) == 1:
            state = state[None, :]
        
        return self.cost_model(state).numpy().squeeze()


def train_deepcube_agent(agent : Agent, M : int=5000, eps : float=0.05, K : int=30,
                         batch_size : int=10000, Epochs : int=1000, lr : float=0.001, warm_up : int=5, 
                         verbose : int=100, save_epoch : int=1000, show : bool=False):
    """
    Parameters
    ----------
    agent : Agent
        The agent to be trained.
    M : int, default = 5000
        The number of the checking rounds.
    eps : float, default = 0.05
        The threshold of the checking rounds.
    K : int, default = 30
        The maximum scrambling steps.
    Epochs : int, default = 1000
        The number of total training epochs.
    lr : float, default = 0.001
        The learning rate of the optimizer.
    warm_up : int, default = 5
        The number of warm up rounds.\n
        In warm up stage, K is set to 1. K is set to maximum scrambling steps util agent `update_cnt` >= `warm_up`.
    verbose : int, default = 100
        The frequency to show training information.
    save_epoch : int, default = 1000
        The frequency to save model.
    show : bool, default = False
        Whether to use Animator to show training information.
    """

    # 定义损失和优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    # 拷贝模型
    cost_model_ = DeepCube(**agent.cost_model.get_config())
    cost_model_.build(input_shape=(None, agent.puzzle.state_length))
    agent.cost_model.build(input_shape=(None, agent.puzzle.state_length))
    agent.cost_model.set_weights(cost_model_.get_weights())

    # 记录 agent 更新次数和未更新次数
    update_cnt, not_update_cnt = 0, 0

    # 创建数据集
    current_K = 1
    dataset = PuzzleDataset(agent, K=current_K, batch_size=batch_size, M=M)

    # 记录 loss 和 cost
    loss_metric = tf.keras.metrics.Mean()
    cost_metric = tf.keras.metrics.Mean()

    # 创建保存模型的文件夹
    puzzle_type = agent.puzzle.puzzle_type.replace("/", "x")
    model_dir, log_dir = f"./model/{puzzle_type}", f"./logs/{puzzle_type}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 配置 logger
    if show:
        animator = Animator(xlabel="Epochs", ylabel="metrics", xlim=(1, Epochs),
                            ncols=2, figsize=(7, 3), legend=(("value mse",), ("mean cost", )),
                            fmts=(('-',), ('m--')))
    else:
        animator = None
    logger = Logger(path_dir=log_dir, name=agent.name)

    # define the training loop
    for epoch in range(Epochs):
        # 通过 scramble 抽样生成初始状态 init_states 和 标签 steps
        for init_states, labels, steps in dataset:
            # 计算损失
            with tf.GradientTape() as tape:
                cost = cost_model_(init_states, training=True)
                loss = loss_func(labels, cost) # (batch_size, 1)
                # 在聚合均值时，可以考虑通过 steps 的权重来调整损失
                loss = tf.reduce_mean(loss)
            
            # 更新参数
            grads = tape.gradient(loss, cost_model_.trainable_variables)
            optimizer.apply_gradients(zip(grads, cost_model_.trainable_variables))

            # 记录到 metric 中
            loss_metric(loss)
            cost_metric(tf.reduce_mean(cost))
        
        # 上述迭代更新完成后，Agent 已经学习了 M * batch_size 个状态
        # M 轮后检查损失是否小于 eps，从而更新 agent.cost_model
        loss_, cost_ = loss_metric.result().numpy(), cost_metric.result().numpy()
        if loss_ < eps:
            # 拷贝 cost_model_ 的参数到 agent.cost_model
            agent.cost_model.set_weights(cost_model_.get_weights())
            msg = f"loss = {loss_:.4f}, mean cost = {cost_:.4f}, agent.cost_model updated."
            logger.info(epoch + 1, msg)
            
            update_cnt += 1
            # warm up 结束后，修改 K 的值，增加 scrambling 的难度
            if update_cnt == warm_up and current_K < K:
                msg = f"K = {current_K} stage warm up stop, "
                current_K = K
                dataset.set_scrambling_steps(current_K)
                msg += f"scrambling steps K is set to {current_K}"
                logger.info(epoch + 1, msg)

                # 重置 update_cnt
                update_cnt = 0
        else:
            # 记录未更新的次数，如果连续 500 次未更新，则强制更新
            not_update_cnt += 1
            if not_update_cnt == 500:
                agent.cost_model.set_weights(cost_model_.get_weights())
                msg = f"loss = {loss_:.4f}, mean cost = {cost_:.4f}, agent.cost_model not updated for 500 epochs, force update."
                logger.info(epoch + 1, msg)
                not_update_cnt = 0
                update_cnt += 1

        # 学习率衰减 lr = lr * (decay_rate ** (epoch + 1))
        if (epoch + 1) % 100 == 0:
            decay_rate = 0.99995
            optimizer.learning_rate.assign(lr * (decay_rate ** (epoch + 1)))
            msg = f'loss = {loss_:.4f}, mean cost = {cost_:.4f}, lr decayed to {optimizer.learning_rate.numpy():.6f}'
            logger.info(epoch + 1, msg)

        # 打印训练信息
        if epoch == 0 or (epoch + 1) % verbose == 0:
            msg = f"loss = {loss_:.4f}, mean cost = {cost_:.4f}"
            logger.info(epoch + 1, msg)
            if show:
                animator.add(epoch+1, (loss_, ), ax=0)
                animator.add(epoch+1, (cost_, ), ax=1)
        
        # 保存模型
        if save_epoch > 0 and (epoch + 1) % save_epoch == 0:
            model_name = "cost_model" if agent.name is None else f"{agent.name}_cost_model"
            agent.cost_model.save_weights(f"{model_dir}/{model_name}_{epoch+1}.h5")

        # 重置 metric
        loss_metric.reset_states()
        cost_metric.reset_states()

    return agent
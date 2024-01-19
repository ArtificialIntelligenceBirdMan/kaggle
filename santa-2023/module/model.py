import tensorflow as tf
from tensorflow import keras

import numpy as np

from typing import Union, List, Tuple, Dict, Any, Optional
from abc import abstractmethod

from .puzzle import PuzzleInfo, PuzzleAction

# 只提供 Agent 类型注释，不提供具体实现
# Agent 类型的具体实现在 agent.py 中
class Agent:
    @abstractmethod
    def __init__(self) -> None:
        # 保存 puzzle 信息和动作集合
        self.puzzle : PuzzleInfo = None
        self.actions : PuzzleAction = None
        # 度量到目标状态的代价函数 
        self.cost_model : tf.keras.Model = None

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, hidden_size : int, state_len : int=1000, dropout : float=0,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个足够长的位置编码矩阵 P
        # P 的第一个维度为 batch_size，便于广播
        self.P = np.zeros((1, state_len, hidden_size))

        X = np.arange(state_len, dtype=np.float32).reshape(-1, 1)\
            / np.power(10000, np.arange(0, hidden_size, 2) / hidden_size)
        self.P[:,:,0::2] = np.sin(X) # 偶数列用 sin
        self.P[:,:,1::2] = np.cos(X) # 奇数列用 cos
    
    def call(self, X : tf.Tensor, **kwargs):
        # X 的形状为 (batch_size, state_len, hidden_size)
        X = X + self.P
        return self.dropout(X, **kwargs)

class LearnablePE(tf.keras.layers.Layer):
    def __init__(self, hidden_size : int, state_len : int=1000, dropout : float=0,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个可学习的位置编码矩阵 P
        # P 的第一个维度为 batch_size，便于广播
        self.P = self.add_weight(
            shape=(1, state_len, hidden_size), 
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True)

    def call(self, X : tf.Tensor, **kwargs):
        # X 的形状为 (batch_size, state_len, hidden_size)
        X = X + self.P
        return self.dropout(X, **kwargs)
    
class AddNorm(tf.keras.layers.Layer):
    def __init__(self, dropout_rate : float=0.0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.bn = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(dropout_rate)
    
    def call(self, X : tf.Tensor, Y : tf.Tensor, **kwargs) -> tf.Tensor:
        # X, Y 形状 (batch_size, state_len, hidden_size)
        return self.bn(self.dropout(Y, **kwargs) + X, **kwargs)

class ResidualAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size : int, num_heads : int=4, dropout_rate : float=0.0, 
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ResidualAttention, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.hidden_size = hidden_size

        # 自注意力层
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size, dropout=dropout_rate)

        # FFN 层
        self.fc1 = keras.layers.Dense(hidden_size, activation="relu")
        self.fc2 = keras.layers.Dense(hidden_size)

        # 残差连接层
        self.addnorm1 = AddNorm(dropout_rate)
        self.addnorm2 = AddNorm(dropout_rate)
    
    def call(self, X : tf.Tensor, **kwargs) -> tf.Tensor:
        # X 形状 (batch_size, state_len, hidden_size)
        # 自注意力层
        Y = self.addnorm1(X, self.attention(X, X, **kwargs), **kwargs)
        # FFN 层
        Y = self.addnorm2(Y, self.fc2(self.fc1(Y)), **kwargs)

        return Y

class ResidualMLP(tf.keras.layers.Layer):
    def __init__(self, hidden_size : int, dropout_rate : float=0.0, 
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ResidualMLP, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # 全连接层
        self.fc1 = keras.layers.Dense(hidden_size)
        self.fc2 = keras.layers.Dense(hidden_size)

        # BatchNorm 层
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(dropout_rate)
    
    def call(self, X : tf.Tensor, **kwargs) -> tf.Tensor:
        Y = keras.activations.relu(self.bn1(self.fc1(X), **kwargs))
        Y = self.bn2(self.fc2(Y), **kwargs)

        return tf.nn.relu(X + self.dropout(Y, **kwargs))

class ResidualIrregularConv1D(tf.keras.layers.Layer):
    def __init__(self, perms : list, hidden_size : int,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        Parameters
        ----------
        perms : list
            List of permutation actions represented by `array_form`,\n
            with shape `(num_of_act, state_len)`
        """
        super(ResidualIrregularConv1D, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        perms = tf.constant(perms)
        self.num_of_act, self.state_len = perms.shape
        self.hidden_size = hidden_size

        # create mask
        masks = []
        offset = self.state_len
        for k,perm in enumerate(perms):
            masks.extend([k*offset + i for i in range(self.state_len) if i != perm[i]])
        masks = tf.constant(masks, dtype=tf.int32)
        # masks is a row vector with shape `(1, num_of_act * state_len)`
        # each `state_len` elements represent a mask for a permutation action
        self.masks = masks

        # create conv1x1 layer
        self.conv1x1_L1 = keras.layers.Conv1D(self.embed_dim, 1, padding='valid')
        self.conv1x1_L2 = keras.layers.Conv1D(self.embed_dim, 1, padding='valid')
        # create convPer layer
        self.convPer = keras.layers.Conv1D(self.state_len, 1, padding='valid')

        # batch normalization
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
    
    def call(self, X, **kwargs):
        """
        Parameters
        ----------
        X : tf.Tensor
            input state vector with shape `(batch_size, state_len, embed_dim)`
        """
        # inorder to calculate in parallel, we first repeat `X` at axis=1 `num_of_act` times
        Y = tf.tile(X, [1, self.num_of_act, 1]) # (batch_size, state_len * num_of_act, embed_dim)
        # use mask to select the elements of `Y` that need to be aggregated
        # Y with shape `(batch_size, num_of_agg, embed_dim)`
        Y = tf.gather(Y, self.masks, axis=1)
        
        # then transpose `Y` to shape `(batch_size, embed_dim, num_of_agg)`
        # so the convPer aggregation the state vector and move in the embed_dim axis
        Y = tf.transpose(Y, perm=[0, 2, 1])
        # use convPer layer to aggregate irregular convolution
        Y = self.convPer(Y) # (batch_size, embed_dim, state_len)
        Y = tf.nn.relu(self.bn1(Y, **kwargs))

        # transpose back to shape `(batch_size, state_len, embed_dim)`
        Y = tf.transpose(Y, perm=[0, 2, 1])
        # use conv1x1 layer to imporve the model capacity
        Y = tf.nn.relu(self.bn2(self.conv1x1_L1(Y), **kwargs))
        Y = self.bn3(self.conv1x1_L2(Y), **kwargs)

        # residual connection
        Y = tf.nn.relu(X + Y)
        
        return X

class DeepCube(tf.keras.Model):
    def __init__(self, 
                 state_len : int, 
                 state_depth : int, 
                 perms : list,
                 embed_size : int, 
                 hidden_size : int, 
                 num_layers : int=3, 
                 dropout_rate : float=0.0, 
                 num_heads : int=1,
                 use_one_hot : bool=False,
                 residual : str="mlp",
                 positional_embedding : str="learnable",
                 **kwargs) -> None:
        super(DeepCube, self).__init__(**kwargs)
        # 使用 one-hot 编码时，residual 必须为  mlp
        residual = "mlp" if use_one_hot else residual

        self.state_len = state_len
        self.state_depth = state_depth
        self.perms = perms
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_one_hot = use_one_hot
        self.residual = residual
        self.positional_embedding = positional_embedding

        # input layer (embedding + flatten + mlp)
        if not use_one_hot:
            self.input_layer = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=(state_len, ), dtype=tf.int32),
                keras.layers.Embedding(state_depth, embed_size)
            ],name="input_layer")
            if positional_embedding == "learnable":
                self.input_layer.add(LearnablePE(embed_size, state_len, dropout_rate))
            elif positional_embedding == "Fixed":
                self.input_layer.add(PositionalEmbedding(embed_size, state_len, dropout_rate))
            elif positional_embedding == "None":
                pass
            
            if residual == "mlp":
                self.input_layer.add(keras.layers.Flatten())
            self.input_layer.add(keras.layers.Dense(2*hidden_size))
            self.input_layer.add(keras.layers.BatchNormalization())
            self.input_layer.add(keras.layers.Activation("relu"))
            self.input_layer.add(keras.layers.Dense(hidden_size))
            self.input_layer.add(keras.layers.BatchNormalization())
            self.input_layer.add(keras.layers.Activation("relu"))
        else:
            self.input_layer = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=(state_len, ), dtype=tf.int32),
                keras.layers.Lambda(lambda X : tf.one_hot(X, state_depth, dtype=tf.float32)),
                keras.layers.Flatten(),
                keras.layers.Dense(2*hidden_size),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dense(hidden_size),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
            ],name="input_layer")


        # residual blocks
        self.residual_layers = []
        for i in range(num_layers):
            if residual == "mlp":
                self.residual_layers.append(
                    ResidualMLP(hidden_size=hidden_size, dropout_rate=dropout_rate))
            elif residual == "attention":
                self.residual_layers.append(
                    ResidualAttention(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate))
            elif residual == "irregular":
                self.residual_layers.append(
                    ResidualIrregularConv1D(perms=perms, hidden_size=hidden_size))

        # output layer
        self.output_layer = keras.models.Sequential([],name="output_layer")
        if residual == "mlp":
            self.output_layer.add(keras.layers.Dense(1))
        elif residual == "attention" or residual == "irregular":
            self.output_layer.add(keras.layers.GlobalAveragePooling1D())
            # self.output_layer.add(keras.layers.Flatten())
            self.output_layer.add(keras.layers.Dense(1))
    
    def call(self, X : tf.Tensor, **kwargs) -> tf.Tensor:
        """
        X : tf.Tensor
            input state with shape = (batch_size, state_len)
        """
        # do embedding
        X = self.input_layer(X, **kwargs)

        # residual blocks
        for layer in self.residual_layers:
            X = layer(X, **kwargs)
        
        return self.output_layer(X)

    def get_config(self):
        config = {
            "state_len" : self.state_len,
            "state_depth" : self.state_depth,
            "perms" : self.perms,
            "embed_size" : self.embed_size,
            "hidden_size" : self.hidden_size,
            "num_layers" : self.num_layers,
            "dropout_rate" : self.dropout_rate,
            "num_heads" : self.num_heads,
            "use_one_hot" : self.use_one_hot,
            "residual" : self.residual,
            "positional_embedding" : self.positional_embedding
        }
        base_config = super(DeepCube, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PuzzleDataset:
    def __init__(self, agent : Agent, K : int=30, batch_size : int=10000, M : int=1000) -> None:
        """
        Parameters
        ----------
        agent : Agent
            用于生成 puzzle dataset 的 Agent
        K : int, default = 30
            每个 puzzle 的 scrambling steps
        batch_size : int, default = 10000
            每个 batch 的大小
        M : int, default = 1000
            设置 checking rounds，每次触发 checking round 时，Agent 会学习 batch_size * M 个状态
        """
        self.agent = agent
        self.K = K
        self.batch_size = batch_size
        self.M = M

        self.dataset = self.create_puzzle_dataset()


    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32)])
    def generate_batch(self, tf_actions, goal_state, num_of_act, state_len, K):
        # 从 [1, K] 的均匀分布中采样 scrambling steps
        steps = tf.random.uniform(shape=(self.batch_size, 1), minval=1, maxval=K+1, dtype=tf.int32)

        # 随机采样 (batch_size, K) 个置换
        # scrambs 形状为 (batch_size, K, state_len)
        scrambs_idx = tf.random.uniform(
            minval=0, maxval=num_of_act, 
            shape=(self.batch_size, K), dtype=tf.int32)
        scrambs = tf.gather(tf_actions, scrambs_idx)

        # 生成单位置换，形状为 (batch_size, K, state_len)
        identity = tf.range(state_len)
        identity = tf.tile(identity[None, None, :], [self.batch_size, K, 1])

        # 根据 steps 创建 mask
        # 对于每一组置换，超过 steps 的位置的 scrambs 都被置为 identity
        # 第一步，利用广播机制，得到 mask 的形状为 (batch_size, K)
        mask = tf.range(K)[None, :] < steps
        # 第二步，扩展 mask 维度，得到形状为 (batch_size, K, state_len)
        mask = tf.tile(mask[:, :, None], [1, 1, state_len])
        mask = tf.cast(mask, tf.int32)

        # 根据 mask 将 scrambs 中超过 steps 的位置置为 identity
        scrambs = scrambs * mask + identity * (1 - mask)

        # 最后，在 goal_state 上应用所有的置换，得到提供给模型的初始状态
        init_state = tf.tile(goal_state[None, :], [self.batch_size, 1])
        for k in range(K):
            init_state = tf.gather(init_state, scrambs[:, k, :], batch_dims=1)

        # 利用 agent 的 cost_model f 评估 cost(x_i, A(x_i, a)) + f(A(x_i, a))
        # 迭代更新 value function，构造标签 y_i = min_a cost(x_i, A(x_i, a)) + f(A(x_i, a))
        # 注意对所有 puzzle，cost(x_i, A(x_i, a)) = 1
        _, best_cost = self.agent.bfs_search(init_state)
        labels = best_cost + 1

        return init_state, labels, steps

    def create_puzzle_dataset(self):
        def create_generator(tf_actions, goal_state, num_of_act, state_len):
            for _ in range(self.M):
                init_state, labels, steps = self.generate_batch(
                    tf_actions, goal_state, num_of_act, state_len, self.K)
                yield init_state, labels, steps

        # 获取目标状态
        goal_state = tf.constant(self.agent.puzzle.goal_state_vec, dtype=tf.int32)
        state_len = self.agent.puzzle.state_length
        num_of_act = len(self.agent.actions)
        tf_actions = tf.stack(list(self.agent.actions.tf_actions.values()))
        
        dataset = tf.data.Dataset.from_generator(
            generator=lambda : create_generator(tf_actions, goal_state, num_of_act, state_len),
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, state_len), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.int32))
            )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def __iter__(self):
        yield from self.dataset
    
    def set_scrambling_steps(self, K : int):
        try:
            assert K > 0
            self.K = K
        except AssertionError:
            self.K = 30
        
        # 更新 dataset
        self.dataset = self.create_puzzle_dataset()
from functools import reduce
import plotly.graph_objects as go
from math import sqrt

import tensorflow as tf

import pandas as pd
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from datetime import datetime

import os
from typing import Union, List, Tuple

CUBE_COLORS = {"A": "white", "B": "#3588cc", "C": "red", "D": "green", "E": "orange", "F": "yellow"}


def gpu_limitation_config(memory : int=30, device : Union[int,list]=0):
    """
    ### 设置所使用的 GPU，以及 GPU 显存\n
    这在多人共用GPU时，能限制Tensorflow所使用的显存资源，方便多人共用GPU\n
    你可以指定所使用的 GPU 编号，以及所使用的显存大小\n

    Parameters
    ----------
    memory : int, default = 30
        设置所使用的GPU显存，单位GB，默认使用 30GB. \n
    device : int, default = 0
        设置所使用的 GPU 编号，默认使用第 0 块 GPU
    """
    # 设置所使用的 GPU
    if device is not None:
        if isinstance(device, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        elif isinstance(device, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in device])
        elif isinstance(device, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 设置所使用的显存大小
    memory = min(memory, 30)
    GPUS = tf.config.list_physical_devices("GPU")

    if GPUS:
        for gpu in GPUS:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory*1024)]
                )
            except RuntimeError as e:
                print(e)


class Animator:
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), 
                 nrows=1, ncols=1, figsize=(5, 3), title : str=None):
        """
        Parameters
        ----------
        xlabel, ylabel, xlim, ylim, xscale, yscale
            横，纵坐标轴相关设置
        legend : list of str
            图例
        title : str
            图标标题
        nrows, ncols, figsize
            子图行数和列数，图像画布大小
        fmts : tuple
            图中每条线的格式配置，例如`g-.`表示用绿色(`green`)绘制点划线`-.`
        """
        # 设置绘图相关信息
        def set_axes(ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.legend(legend)
            ax.set_title(title)
            ax.grid()
            plt.tight_layout()
        
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
            legend = (legend,)
            fmts = (fmts,)
        else:
            self.axes = self.axes.flatten()
        # lambda 函数将配置参数的信息保存到 set_axes() 函数
        self.config_axes = lambda ax : \
            set_axes(self.axes[ax], xlabel, ylabel, xlim, ylim, xscale, yscale, legend[ax], title)
        self.X, self.Y = [None for _ in self.axes], [None for _ in self.axes]
        self.fmts = fmts # 初始化

    def add(self, x, y : list, ax : int=0):
        """
        在现有的图上添加新的点
        """
        # 如果 y 不是序列类型，就转换为列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y) # 共有 n 条线

        # 如果 x 不是序列类型，就转换为列表
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        # 初始化 X
        if not self.X[ax]:
            self.X[ax] = [[] for _ in range(n)] 
        # 初始化 Y
        if not self.Y[ax]:
            self.Y[ax] = [[] for _ in range(n)] 
        
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[ax][i].append(a) # 添加横坐标
                self.Y[ax][i].append(b) # 添加纵坐标

        self.axes[ax].cla() # 清空画布
        for x, y, fmt in zip(self.X[ax], self.Y[ax], self.fmts[ax]):
            self.axes[ax].plot(x, y, fmt) # 绘制曲线
        
        self.config_axes(ax) # 配置画布
        display.display(self.fig) # 展示画布
        display.clear_output(wait=True) # 延迟清除

class Logger:
    def __init__(self, path_dir : str, name : str=None):
        self.path_dir = path_dir
        self.name = name
        self.log_file = f"{self.path_dir}/{self.name}.log"
        
        # 如果存在日志文件，就删除 
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # 创建日志文件
        f = open(self.log_file, "w").close()
        self.epoch = 0
    
    def info(self, epoch : int, msg : str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f = open(self.log_file, "a")
        f.write(f"[{t}] epoch [{epoch}]: {msg}\n")
        f.close()
        self.epoch = epoch
    

def create_standard_goal(puzzles : pd.DataFrame):
    # 找到每种 puzzle_type 的标准目标状态
    standard_goal = {}
    for puzzle_type in puzzles.puzzle_type.unique():
        df = puzzles.loc[puzzles.puzzle_type == puzzle_type]
        if "A" in df["solution_state"].values[0]:
            goal_state = df["solution_state"].values[0]
            standard_goal[puzzle_type] = goal_state
    
    return standard_goal

def transform_into_standard_goal(goal_state : str, init_state : str, puzzle_type : str, standard_goal : dict):
    if "N0" in goal_state:
        goal_state, init_state = goal_state.split(";"), init_state.split(";")
        standard_goal_state = standard_goal[puzzle_type].split(";")
        n = len(goal_state)
        # 创建一个从 N{k} 到标准目标状态的映射字典
        mapping = {goal_state[i]: standard_goal_state[i] for i in range(n)}
        
        # 将 init_state 中的 N{k} 映射到标准目标状态
        init_state = [mapping[init_state[i]] for i in range(n)]
        
        # 将 goal_state 替换为 standard_goal_state
        # 并将 goal_state 和 init_state 重新拼接成字符串
        goal_state = ";".join(standard_goal_state)
        init_state = ";".join(init_state)

    return goal_state, init_state


class PuzzleShowMaker:
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def show_cube(facelets : Union[str, List, Tuple], n : int=3, width=600, height=600, **kwargs):
        facelets = facelets.split(";") if type(facelets) == str and ";" in facelets else facelets

        fs = n**2       # Number of facelets per face
        vs = (n+1)**2   # Number of vertices per face        
        
        # calculate x, y, z coordinates for each face
        vertices = [(x, y, 0) for y in range(n, -1, -1) for x in range(n+1)]        # dn down
        vertices.extend([(x, 0, z) for z in range(n+1) for x in range(n+1)])        # f0 front
        vertices.extend([(n, y, z) for z in range(n+1) for y in range(n+1)])        # rn right
        vertices.extend([(x, n, z) for z in range(n+1) for x in range(n, -1, -1)])  # fn back
        vertices.extend([(0, y, z) for z in range(n+1) for y in range(n, -1, -1)])  # r0 left
        vertices.extend([(x, y, n) for y in range(n+1) for x in range(n+1)])        # d0 up

        # get facelet colors
        facelet_colors = [CUBE_COLORS[facelet] for facelet in facelets]
        facecolor = []
        for f_color in facelet_colors:
            facecolor.extend([f_color, f_color])

        # Building the mesh for the cube with triangles made out of 3 vertices (i, j, k) 
        # and each facelet is made out of 2 triangles
        ivs = []
        for i in range(vs):
            if (i+1) % (n+1) != 0 and i+1 < n*(n+1): ivs.extend([i, i])

        jvs = []
        for i, j in zip([i for i in range(vs) if i % (n+1) != 0 and i < n*(n+1)], [j for j in range(vs) if (j+1) % (n+1) != 0 and j+1 > n+1]):
            jvs.extend([i, j])

        kvs = []
        for i in range(vs):
            if (i) % (n+1) != 0 and i+1 > n+1: kvs.extend([i, i])

        fig = go.Figure(data=[
            go.Mesh3d(
                x=[v[0] for v in vertices],
                y=[v[1] for v in vertices],
                z=[v[2] for v in vertices],
                i=reduce(lambda x, y: x.extend(y) or x, [[v+vs*i for v in ivs] for i in range(6)]),
                j=reduce(lambda x, y: x.extend(y) or x, [[v+vs*i for v in jvs] for i in range(6)]),
                k=reduce(lambda x, y: x.extend(y) or x, [[v+vs*i for v in kvs] for i in range(6)]),
                facecolor=facecolor,
                opacity=1,
                hoverinfo='none'
            )
        ])

        # add the black lines to the cube
        lines_seq = [[0, n, n, 0, 0], [0, 0, n, n, 0]]

        for i in range(n+1):
            # Z axis lines
            fig.add_trace(go.Scatter3d(x=lines_seq[0], y=lines_seq[1], z=[i]*5, 
                                       mode="lines", line=dict(width=5, color="black"), hoverinfo="none"))

            # Y axis lines
            fig.add_trace(go.Scatter3d(x=lines_seq[1], y=[i]*5, z=lines_seq[0],
                                       mode="lines", line=dict(width=5, color="black"), hoverinfo="none"))

            # X axis lines
            fig.add_trace(go.Scatter3d(x=[i]*5, y=lines_seq[1], z=lines_seq[0],
                                       mode="lines", line=dict(width=5, color="black"), hoverinfo="none"))

        # add the axis texts
        fig.add_trace(go.Scatter3d(
            x=[n/2, n/2, n+1.5+n*0.5], y=[n/2, -1.5-n*0.5, n/2], z=[n+1+n*0.5, n/2, n/2], 
            mode="text", text=["UP", "FRONT", "RIGHT"], textposition="middle center", 
            textfont=dict(size=15+n*2,)))

        # set the layout and removing the legend, background, grid, ticks, etc.
        fig.update_layout(
            showlegend=False,
            autosize=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, title_text="", showspikes=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, title_text="", showspikes=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, title_text="", showspikes=False),
                camera=dict(
                    eye=dict(x=0.8, y=-1.2, z=0.8)
                )
            ),
            width=width,
            height=height,
            **kwargs
        )

        fig.show()

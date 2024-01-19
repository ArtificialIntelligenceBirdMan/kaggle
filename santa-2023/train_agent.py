from module.agent import Agent, train_deepcube_agent
from module.puzzle import Puzzle, PuzzleInfo, PuzzleAction
from module.model import DeepCube
from module import utils
import pandas as pd
import tensorflow as tf

import os

import json
import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    gpu_parser = parser.add_argument_group('gpu')
    puzzle_parser = parser.add_argument_group('puzzle')
    model_parser = parser.add_argument_group('model')
    train_parser = parser.add_argument_group('train')

    gpu_parser.add_argument('--gpu', type=int, default=0)
    gpu_parser.add_argument('--memory', type=int, default=30)

    # 设置 puzzle
    puzzle_parser.add_argument('--puzzle_type', type=str, default='cube_2/2/2')
    puzzle_parser.add_argument('--agent_name', type=str, default='cube_2x2x2_agent')
    
    # 设置模型参数
    model_parser.add_argument('--embed_size', type=int, default=4)
    model_parser.add_argument('--hidden_size', type=int, default=512)
    model_parser.add_argument('--num_layers', type=int, default=3)
    model_parser.add_argument('--dropout_rate', type=float, default=0.0)
    model_parser.add_argument('--residual', type=str, default='attention', choices=['attention', 'mlp'])
    model_parser.add_argument('--use_one_hot', action='store_true')
    model_parser.add_argument('--positional_embedding', type=str, default='learnable', choices=['learnable', 'fixed'])
    model_parser.add_argument('--num_heads', type=int, default=1)

    # 训练参数
    train_parser.add_argument('--K', type=int, default=15)
    train_parser.add_argument('--M', type=int, default=50)
    train_parser.add_argument('--eps', type=float, default=0.05)
    train_parser.add_argument('--epochs', type=int, default=10000)
    train_parser.add_argument('--batch_size', type=int, default=10000)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--verbose', type=int, default=20)
    train_parser.add_argument('--save_epoch', type=int, default=1000)

    args = parser.parse_args()
    # 保存模型参数
    model_params = {}
    for action in model_parser._group_actions:
        param = action.dest
        model_params[param] = getattr(args, param)
    model_dir = f"./model/{args.puzzle_type.replace('/', 'x')}"
    os.makedirs(model_dir, exist_ok=True)
    json.dump(model_params, open(f"{model_dir}/{args.agent_name}_cost_model.json", 'w'), indent=4)


    # 设置 GPU
    utils.gpu_limitation_config(device=args.gpu, memory=args.memory)

    # 读取数据
    puzzle_info = pd.read_csv('./data/puzzle_info.csv')
    puzzle_info.set_index('puzzle_type', inplace=True)
    puzzles = pd.read_csv('./data/puzzles.csv')
    logging.info("all puzzles: %s"%(puzzle_info.index.values))

    # 找到每种 puzzle_type 的标准目标状态
    standard_goal = {}
    for puzzle_type in puzzles.puzzle_type.unique():
        df = puzzles.loc[puzzles.puzzle_type == puzzle_type]
        if "A" in df["solution_state"].values[0]:
            goal_state = df["solution_state"].values[0]
            standard_goal[puzzle_type] = goal_state
            # 添加到 puzzle_info 中
            puzzle_info.loc[puzzle_type, "goal_state"] = goal_state
    
    # 创建 Puzzle 环境
    puzzle_type = args.puzzle_type
    puzzle_info_obj = PuzzleInfo(
        puzzle_type=puzzle_type, goal_state=puzzle_info.loc[puzzle_type, "goal_state"],
    )
    puzzle_act_obj = PuzzleAction(
        puzzle_type=puzzle_type,
        moves=eval(puzzle_info.loc[puzzle_type, "allowed_moves"]),
    )

    # 创建 model 和 agent
    cost_model = DeepCube(
        state_len=puzzle_info_obj.state_length, 
        state_depth=puzzle_info_obj.state_depth,
        perms=list(puzzle_act_obj.tf_actions.values()),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        residual=args.residual,
        use_one_hot=args.use_one_hot,
        positional_embedding=args.positional_embedding,
        num_heads=args.num_heads
    )
    cost_model.build(input_shape=(None, puzzle_info_obj.state_length))
    
    agent = Agent(puzzle_info_obj, puzzle_act_obj, cost_model, 
                  name=args.agent_name)
    
    # 训练
    agent = train_deepcube_agent(
        agent=agent, 
        M=args.M,
        eps=args.eps, 
        K=args.K,
        Epochs=args.epochs, 
        batch_size=args.batch_size,
        lr=args.lr,
        warm_up=3, 
        show=False, 
        verbose=args.verbose,
        save_epoch=args.save_epoch,
    )
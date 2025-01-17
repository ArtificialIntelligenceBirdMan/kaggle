from module.agent import Agent, train_deepcube_agent
from module.puzzle import Puzzle, PuzzleInfo, PuzzleAction, get_puzzle_sub_type
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

    gpu_parser.add_argument('--gpu', type=str, default="0")
    gpu_parser.add_argument('--data_gpu', type=str, default=None)
    gpu_parser.add_argument('--memory', type=int, default=30)

    # 设置 puzzle
    puzzle_parser.add_argument('--puzzle_type', type=str, default='cube_2/2/2')
    puzzle_parser.add_argument('--sub_type', type=str, default='S')
    puzzle_parser.add_argument('--agent_name', type=str, default='cube_2x2x2_S_agent')
    
    # 设置模型参数
    model_parser.add_argument('--embed_size', type=int, default=4)
    model_parser.add_argument('--hidden_size', type=int, default=512)
    model_parser.add_argument('--num_layers', type=int, default=3)
    model_parser.add_argument('--dropout_rate', type=float, default=0.0)
    model_parser.add_argument('--residual', type=str, default='attention', choices=['attention', 'mlp'])
    model_parser.add_argument('--use_one_hot', action='store_true')
    model_parser.add_argument('--positional_embedding', type=str, default='learnable', choices=['learnable', 'fixed', 'none'])
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
    train_parser.add_argument('--pre_train', type=str, default=None)

    args = parser.parse_args()
    # 保存模型参数
    model_params = {}
    for action in model_parser._group_actions:
        param = action.dest
        model_params[param] = getattr(args, param)
    model_dir = f"./model/{args.puzzle_type.replace('/', 'x')}_{args.sub_type}"
    os.makedirs(model_dir, exist_ok=True)
    json.dump(model_params, open(f"{model_dir}/{args.agent_name}_cost_model.json", 'w'), indent=4)


    # 设置 GPU
    utils.gpu_limitation_config(device=args.gpu, memory=args.memory)

    # 读取数据
    puzzle_info = pd.read_csv('./data/puzzle_info.csv')
    puzzle_info.set_index('puzzle_type', inplace=True)
    puzzles = pd.read_csv('./data/puzzles.csv')

    # 确定 sub_type
    all_puzzles = pd.DataFrame(puzzles.groupby(by=["puzzle_type", "solution_state"]).indices.keys())
    all_puzzles.columns = ["puzzle_type", "goal_state"]
    all_puzzles["sub_type"] = all_puzzles.apply(lambda x : get_puzzle_sub_type(x.iloc[0], x.iloc[1].split(";")), axis=1)
    all_puzzles.set_index(keys=["puzzle_type","sub_type"], inplace=True)
    
    # 创建 Puzzle 环境
    puzzle_type = args.puzzle_type
    sub_type = args.sub_type
    with tf.device("/GPU:0"):
        puzzle_info_obj = PuzzleInfo(
            puzzle_type=puzzle_type, goal_state=all_puzzles.loc[(puzzle_type, sub_type), "goal_state"], sub_type=sub_type
        )
        puzzle_act_obj = PuzzleAction(
            puzzle_type=puzzle_type,
            moves=eval(puzzle_info.loc[puzzle_type, "allowed_moves"]),
        )

        # 创建 model 和 agent
        cost_model = DeepCube(
            state_len=puzzle_info_obj.state_length, 
            state_depth=puzzle_info_obj.state_depth,
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
        data_gpu=args.data_gpu,
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
        pre_train=args.pre_train
    )
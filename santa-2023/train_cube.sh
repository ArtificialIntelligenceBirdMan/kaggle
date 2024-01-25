# nohup python train_agent.py --gpu=0 --puzzle_type="cube_2/2/2" --agent_name="cube_2x2x2_agent" \
#     --K=15 --residual="mlp" --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=512 --embed_size=4 \
#     --lr=1e-3 --epochs=20000 --batch_size=10000    > ./logs/train_cube_2x2x2_agent.log 2>&1 &

# nohup python train_agent.py --gpu=1 --puzzle_type="cube_3/3/3" --agent_name="cube_3x3x3_agent" \
#     --K=30 --residual="mlp"  --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=512 --embed_size=4 \
#     --lr=1e-3 --epochs=30000 --batch_size=10000    > ./logs/train_cube_3x3x3_agent.log 2>&1 &

# nohup python train_agent.py --gpu=0 --puzzle_type="cube_4/4/4" --agent_name="cube_4x4x4_agent" \
#     --K=40 --residual="mlp"  --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=512 --embed_size=4 \
#     --lr=1e-3 --epochs=50000 --batch_size=10000    > ./logs/train_cube_4x4x4_agent.log 2>&1 &


# nohup python train_agent.py --gpu=0 --puzzle_type="cube_10/10/10" --agent_name="cube_10x10x10_agent" \
#     --K=200 --residual="mlp" --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
#     --lr=1e-3 --epochs=200000 --batch_size=10000    > ./logs/train_cube_10x10x10_agent.log 2>&1 &


nohup python train_agent.py --gpu=0 --puzzle_type="cube_5/5/5" --agent_name="cube_5x5x5_agent" \
    --K=100 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --M=50 --save_epoch=1000 --epochs=30000 --batch_size=10000    > ./logs/train_cube_5x5x5_agent.log 2>&1 &

nohup python train_agent.py --gpu=1 --puzzle_type="cube_6/6/6" --agent_name="cube_6x6x6_agent" \
    --K=150 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --M=50 --save_epoch=500 --epochs=50000 --batch_size=10000    > ./logs/train_cube_6x6x6_agent.log 2>&1 &

nohup python train_agent.py --gpu=2 --puzzle_type="cube_7/7/7" --agent_name="cube_7x7x7_agent" \
    --K=200 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --M=50 --save_epoch=500 --epochs=50000 --batch_size=10000    > ./logs/train_cube_7x7x7_agent.log 2>&1 &

nohup python train_agent.py --gpu=7 --puzzle_type="cube_4/4/4" --agent_name="cube_4x4x4_agent" \
    --K=100 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --M=50 --save_epoch=500 --epochs=50000 --batch_size=10000    > ./logs/train_cube_4x4x4_agent.log 2>&1 &
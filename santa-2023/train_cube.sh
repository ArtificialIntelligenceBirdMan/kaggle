nohup python train_agent.py --gpu=0 --puzzle_type="cube_2/2/2" --agent_name="cube_2x2x2_agent" \
    --K=20 --residual="mlp" --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=512 --embed_size=4 \
    --lr=1e-3 --epochs=20000 --batch_size=10000    > ./logs/train_cube_2x2x2_agent.log 2>&1 &

nohup python train_agent.py --gpu=1 --puzzle_type="cube_3/3/3" --agent_name="cube_3x3x3_agent" \
    --K=30 --residual="mlp"  --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=512 --embed_size=4 \
    --lr=1e-4 --epochs=30000 --batch_size=10000    > ./logs/train_cube_3x3x3_agent.log 2>&1 &
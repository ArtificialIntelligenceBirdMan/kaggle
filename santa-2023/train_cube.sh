nohup python train_agent.py --gpu="4,5,6,7" --data_gpu="1,2,3" --puzzle_type="cube_3/3/3" --sub_type="S"  --agent_name="cube_3x3x3_S_parallel_agent" \
    --K=30 --residual="mlp" --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=512 --embed_size=4 \
    --lr=1e-3 --M=500  --epochs=20000 --batch_size=10000  > ./logs/train_cube_3x3x3_S_parallel_agent.log 2>&1 &
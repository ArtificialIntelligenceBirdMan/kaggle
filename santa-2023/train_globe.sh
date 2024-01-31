# nohup python train_agent.py --gpu="4,5,6,7" --data_gpu="1,2,3" --puzzle_type="globe_1/8" --sub_type="S"  --agent_name="globe_1x8_S_agent" \
#     --K=40 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
#     --lr=1e-3 --M=200 --save_epoch=1000 --epochs=30000 --batch_size=10000    > ./logs/train_globe_1x8_S_agent.log 2>&1 &

nohup python train_agent.py --gpu="0,1,2,3" --data_gpu="1,2,3" --puzzle_type="globe_1/8" --sub_type="N"  --agent_name="globe_1x8_N_agent" \
    --K=40 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --M=200 --save_epoch=1000 --epochs=30000 --batch_size=10000    > ./logs/train_globe_1x8_N_agent.log 2>&1 &
# nohup python train_agent.py --gpu=0 --puzzle_type="wreath_6/6" --agent_name="wreath_6x6_agent" \
#     --K=40 --residual="mlp"  --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
#     --lr=1e-3 --M=50 --save_epoch=1000 --epochs=30000 --batch_size=10000    > ./logs/train_wreath_6x6_agent.log 2>&1 &

# nohup python train_agent.py --gpu=1 --puzzle_type="wreath_7/7" --agent_name="wreath_7x7_agent" \
#     --K=50 --residual="mlp"  --positional_embedding="learnable" --num_layers=3 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
#     --lr=1e-3 --M=50 --save_epoch=1000 --epochs=30000 --batch_size=10000    > ./logs/train_wreath_7x7_agent.log 2>&1 &

nohup python train_agent.py --gpu=4 --puzzle_type="wreath_12/12" --agent_name="wreath_12x12_agent" \
    --K=150 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --eps=5e-3  --M=50 --save_epoch=500 --epochs=50000 --batch_size=10000    > ./logs/train_wreath_12x12_agent.log 2>&1 &

nohup python train_agent.py --gpu=5 --puzzle_type="wreath_21/21" --agent_name="wreath_21x21_agent" \
    --K=200 --residual="mlp"  --positional_embedding="learnable" --num_layers=4 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --eps=5e-3  --M=50 --save_epoch=500 --epochs=50000 --batch_size=10000    > ./logs/train_wreath_21x21_agent.log 2>&1 &

nohup python train_agent.py --gpu=6 --puzzle_type="wreath_33/33" --agent_name="wreath_33x33_agent" \
    --K=300 --residual="mlp"  --positional_embedding="learnable" --num_layers=5 --dropout_rate=0.0 --hidden_size=1024 --embed_size=4 \
    --lr=1e-3 --eps=5e-3  --M=50 --save_epoch=500 --epochs=50000 --batch_size=10000    > ./logs/train_wreath_33x33_agent.log 2>&1 &
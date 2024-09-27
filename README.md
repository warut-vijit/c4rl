## Playing around with RL for connect 4

### Command for training

```
python3 mcts/model.py \
  --parquet_paths_comma_separated mcts/data/2024092701_003330_003330.parquet \
  --checkpoint_save_parent_dir mcts/checkpoints/ \
  --checkpoint_load_path mcts/checkpoints/old_model/iter_0001337.pt \
```

### Command for self-play

```
python3 mcts/simulate.py \
  --player_plus_repr cnn:mcts/checkpoints/model/iter_0000199.pt:0.2 \
  --player_minus_repr cnn:mcts/checkpoints/model/iter_0000199.pt:0.2 \
  --num_simulations 3200 \
  --output_dir mcts/data
```

### Command for playing against model

```
python3 mcts/simulate.py \
  --player_plus_repr human \
  --player_minus_repr cnn:mcts/checkpoints/model/iter_0000199.pt:0.2 \
  --num_simulations 1 \
  --output_dir /dev/null
```

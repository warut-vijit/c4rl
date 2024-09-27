from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, random_split

from game import AnnotatedState, HEIGHT, WIDTH
from utils import get_random_name


class CNNResidualBlock(nn.Module):
    def __init__(self):
        super(CNNResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state):
        conv1_out = self.conv1(hidden_state)
        norm1_out = F.rms_norm(conv1_out, normalized_shape=(HEIGHT, WIDTH))
        relu1_out = F.relu(norm1_out)
        conv2_out = self.conv2(relu1_out)
        norm2_out = F.rms_norm(conv2_out, normalized_shape=(HEIGHT, WIDTH))
        skip_out = hidden_state + norm2_out
        relu_out = F.relu(skip_out)
        return relu_out


class ValueCNN(nn.Module):
    def __init__(self):
        super(ValueCNN, self).__init__()
        self.first_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.cnnr_block_list = nn.ModuleList([CNNResidualBlock() for _ in range(4)])
        self.fc = nn.Linear(16 * HEIGHT * WIDTH, 1)

    def forward(self, x):
        x = self.first_conv(x)
        for cnnr in self.cnnr_block_list:
            x = cnnr(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = torch.tanh(x)
        return x


def state_to_model_input(state: npt.NDArray) -> torch.Tensor:
    """Converts batch of states to batch of model inputs
    Input: npt.NDArray[(height, width), dtype=np.int8]
    Output: npt.NDArray[(3, height, width), dtype=torch.float32]
    """
    # (h, w, c=3)
    one_hot_state = np.eye(3)[state]
    # (c, h, w)
    one_hot_transposed = rearrange(one_hot_state, "h w c -> c h w")
    one_hot_contiguous = np.ascontiguousarray(one_hot_transposed)
    return torch.tensor(one_hot_contiguous, dtype=torch.float32)


class TrajectoryStateDataset(Dataset):
    def __init__(self, parquet_paths: List[str], discount_factor: float) -> None:
        self.parquet_paths = parquet_paths
        parquet_num_rows = np.array([pq.read_metadata(parquet_path).num_rows for parquet_path in parquet_paths])
        self.num_rows = np.sum(parquet_num_rows)
        self.parquet_offsets = np.cumsum(parquet_num_rows) - parquet_num_rows
        self.log_discount_factor = np.log(discount_factor)

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # parquet_offsets contains sum of rows in all preceding parquets.
        # so if there are two parquets with 100 rows, parquet_offsets = [0, 100]
        # and indices 0-99 maps to parquet_index=0 and 100-199 maps to parquet_index=1
        parquet_index = np.searchsorted(self.parquet_offsets, idx, side="right") - 1
        parquet_path = self.parquet_paths[parquet_index]
        data = pd.read_parquet(parquet_path)
        local_index = idx - self.parquet_offsets[parquet_index]

        row = AnnotatedState(**data.iloc[local_index])
        steps_to_end = row.trajectory_length - row.step_index - 1
        discounted_score = np.exp(steps_to_end * self.log_discount_factor) * row.score
        model_input = state_to_model_input(row.state)
        return model_input, torch.tensor([discounted_score], dtype=torch.float32)


def train(
    parquet_paths_comma_separated: str,
    checkpoint_save_parent_dir: str,
    learning_rate: float = 1e-3,
    discount_factor: float = 0.99,
    batch_size: int = 32,
    iterations: int = -1,
    eval_every: int = 10,
    save_every: int = -1,
    checkpoint_load_path: Optional[str] = None,
):
    now = datetime.now()
    checkpoint_name = get_random_name()
    checkpoint_save_dir = Path(checkpoint_save_parent_dir) / now.strftime(f"%Y%m%d%H_{checkpoint_name}")
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = parquet_paths_comma_separated.split(",")
    full_dataset = TrajectoryStateDataset(parquet_paths=parquet_paths, discount_factor=discount_factor)
    train_dataset, valid_dataset = random_split(full_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    train_batch_iterator = iter(train_dataloader)
    valid_batch_iterator = iter(valid_dataloader)

    # Infer iterations as number of batches if not specified
    if iterations == -1:
        iterations = len(train_dataset) // batch_size
        print("Inferred iterations as {len(train_dataset)}/{batch_size} = {iterations}")

    # Initialize the model,
    model = ValueCNN()
    if checkpoint_load_path is not None:
        state_dict = torch.load(checkpoint_load_path)
        model.load_state_dict(state_dict)

    # Initial loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []
    progress_bar = tqdm.tqdm(range(iterations))
    for iteration in progress_bar:
        try:
            train_model_input, train_targets = next(train_batch_iterator)

            # Forward pass
            train_output = model(train_model_input)
            train_loss = criterion(train_output, train_targets)
            train_loss_val = train_loss.item()
            train_losses.append((iteration, train_loss_val))

            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()

            # Gradient descent step
            optimizer.step()

            if eval_every > 0 and not iteration % eval_every:
                model.eval()
                with torch.no_grad():
                    valid_model_input, valid_targets = next(valid_batch_iterator)
                    valid_output = model(valid_model_input)
                    valid_loss = criterion(valid_output, valid_targets)
                    valid_loss_val = valid_loss.item()
                    valid_losses.append((iteration, valid_loss_val))
                model.train()
            else:
                valid_loss_val = None

            # Save state dict
            if (save_every > 0 and not iteration % save_every) or (iteration == iterations - 1):
                checkpoint_save_path = checkpoint_save_dir / f"iter_{iteration:0>7}.pt"
                state_dict = model.state_dict()
                torch.save(state_dict, checkpoint_save_path)

            # Update progress bar description
            last_valid_loss = valid_losses[-1][1] if valid_losses else "N/A"
            progress_bar.set_description(f"train={train_loss_val:.4} valid={last_valid_loss:.4}")
        except StopIteration:
            break

    # Save loss plots to file
    plt.plot(*zip(*train_losses), label="train loss")
    plt.plot(*zip(*valid_losses), label="valid loss")
    plt.title(checkpoint_save_dir.as_posix())
    plt.legend()
    plot_save_path = checkpoint_save_dir / f"plot.jpg"
    plt.savefig(plot_save_path)
    print(f"Saved plot to {plot_save_path.as_posix()}")


if __name__ == "__main__":
    # python3 model.py data/20240925_002104.parquet checkpoints/
    fire.Fire(train)

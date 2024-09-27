import itertools
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, TypeVar

import fire
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm

from game import AnnotatedState, GameState, HEIGHT, Sign, Trajectory, WIDTH
from player import AbstractPlayer, CNNPlayer, HumanPlayer, RandomPlayer

logger = logging.getLogger(__name__)


def player_repr_to_player(player_repr: str) -> AbstractPlayer:
    if player_repr == "random":
        return RandomPlayer()
    if player_repr == "human":
        return HumanPlayer()
    if player_repr.startswith("cnn:"):
        _, checkpoint_path, epsilon_str = player_repr.split(":")
        checkpoint_path = Path(checkpoint_path)
        return CNNPlayer(checkpoint_path, float(epsilon_str))


def simulate_one(player_plus: AbstractPlayer, player_minus: AbstractPlayer) -> Trajectory:
    trajectory_states = []
    state = GameState.empty()
    for player, sign, do_inv in itertools.cycle([(player_plus, 1, False), (player_minus, -1, True)]):
        player_state = state.inv if do_inv else state
        move = player.move(player_state)
        state.apply_move(move, sign)
        trajectory_states.append(state.state.copy())
        logger.info(state.show())
        logger.info("")
        if state.score != 0:
            logger.info(f"Player {state.score} won.")
            break
        if not np.any(state.valid_move_mask):
            break
    return trajectory_states, state.score


def save_trajectories(trajectories: List[Trajectory], output_path: Path) -> None:
    rows_to_save = []
    for trajectory_index, (trajectory_states, score) in enumerate(trajectories):
        for step_index, state in enumerate(trajectory_states):
            state_bytes = state.tobytes()
            row_to_save = AnnotatedState(
                trajectory_index=trajectory_index,
                score=score,
                step_index=step_index,
                trajectory_length=len(trajectory_states),
                state_bytes=state_bytes,
            )
            rows_to_save.append(row_to_save)
    dataframe = pd.DataFrame(rows_to_save, columns=AnnotatedState._fields)
    dataframe.to_parquet(output_path.as_posix())


def simulate(
    player_plus_repr: str,
    player_minus_repr: str,
    num_simulations: int,
    output_dir: str,
) -> None:
    logging.basicConfig(level=logging.WARNING)
    player_plus = player_repr_to_player(player_plus_repr)
    player_minus = player_repr_to_player(player_minus_repr)

    trajectories: List[Trajectory] = []
    for simulation_idx in tqdm.tqdm(range(num_simulations)):
        trajectory = simulate_one(player_plus, player_minus)
        trajectories.append(trajectory)

    now = datetime.now()
    output_path = Path(output_dir) / now.strftime(f"%Y%m%d%H_{player_plus.name}_{player_minus.name}.parquet")
    save_trajectories(trajectories, output_path)
    print(f"Saved trajectories to {output_path}")


if __name__ == "__main__":
    # python3 simulate.py random random 10 data/
    fire.Fire(simulate)

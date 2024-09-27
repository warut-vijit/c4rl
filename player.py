from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from game import GameState, Sign
from model import ValueCNN, state_to_model_input


class AbstractPlayer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def move(self, state: GameState) -> int:
        pass


class RandomPlayer(AbstractPlayer):
    @property
    def name(self) -> str:
        return "random"

    def move(self, state: GameState) -> int:
        valid_moves = state.valid_move_mask
        return np.random.choice(np.nonzero(valid_moves)[0])


class HumanPlayer(AbstractPlayer):
    @property
    def name(self) -> str:
        return "human"

    def move(self, state: GameState) -> int:
        move_str = input("Player move: ")
        return int(move_str)


class CNNPlayer(AbstractPlayer):
    def __init__(self, checkpoint_path: Path, epsilon: float):
        self._name = checkpoint_path.parent.name
        self.epsilon = epsilon

        self.model = ValueCNN()
        state_dict = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @property
    def name(self) -> str:
        return self._name

    def move(self, state: GameState) -> int:
        valid_moves = state.valid_move_mask
        # with probability epsilon, pick random move
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.nonzero(valid_moves)[0])
        best_move = 0
        best_score = -np.inf
        for valid_move in np.nonzero(valid_moves)[0]:
            state_copy = deepcopy(state)
            state_copy.apply_move(valid_move, 1)
            # add batch dimension
            model_input = state_to_model_input(state_copy.state).unsqueeze(0)
            score = self.model(model_input).item()
            if score > best_score:
                best_move = valid_move
                best_score = score
        return best_move

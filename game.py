from typing import List, Literal, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.signal as ss
from typing import NamedTuple

WIDTH = 7
HEIGHT = 6

Sign: TypeVar = Literal[-1, 1]
Trajectory: TypeVar = Tuple[List[npt.NDArray], int]


class GameState(NamedTuple):
    state: npt.NDArray
    count: npt.NDArray

    @classmethod
    def empty(cls) -> "GameState":
        state = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        count = np.zeros((WIDTH,), dtype=np.int8)
        return cls(state, count)

    @property
    def score(self) -> int:
        h_kern = np.ones((1, 4), dtype=np.int8)
        v_kern = np.ones((4, 1), dtype=np.int8)
        d_kern = np.eye(4, dtype=np.int8)

        h_conv_state = ss.convolve2d(self.state, h_kern, mode="valid")
        v_conv_state = ss.convolve2d(self.state, v_kern, mode="valid")
        d1_conv_state = ss.convolve2d(self.state, d_kern, mode="valid")
        d2_conv_state = ss.convolve2d(self.state, d_kern[::-1], mode="valid")

        if np.any(h_conv_state == 4) or np.any(v_conv_state == 4) or np.any(d1_conv_state == 4) or np.any(d2_conv_state == 4):
            return 1
        if np.any(h_conv_state == -4) or np.any(v_conv_state == -4) or np.any(d1_conv_state == -4) or np.any(d2_conv_state == -4):
            return -1
        return 0

    def apply_move(self, move: int, sign: Sign) -> None:
        assert self.count[move] < HEIGHT
        self.state[self.count[move], move] = sign
        self.count[move] += 1

    @property
    def valid_move_mask(self) -> npt.NDArray:
        return self.count < HEIGHT

    def show(self) -> str:
        _val_to_sym = np.vectorize(lambda v: ["-", " ", "+"][v+1])
        sym_list_list = _val_to_sym(self.state).tolist()
        return "\n".join("|".join(sym_list) for sym_list in sym_list_list[::-1])

    @property
    def inv(self) -> "GameState":
        return GameState(self.state * -1, self.count)


class AnnotatedState(NamedTuple):
    trajectory_index: int
    score: int
    step_index: int
    trajectory_length: int
    state_bytes: bytes

    @property
    def state(self) -> npt.NDArray:
        return np.frombuffer(self.state_bytes, dtype=np.int8).reshape(HEIGHT, WIDTH)

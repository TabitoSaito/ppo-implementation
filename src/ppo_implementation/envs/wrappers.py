from ..utils.constants import Identifier
from gymnasium import Env, ObservationWrapper
from typing import Any
import numpy as np

class OneHotEncodeBoard(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: Any) -> Any:
        board = observation["board"]

        temp_arrays = []

        for i in range(9):
            temp = np.zeros(board.shape)
            temp[board == i] = 1
            temp_arrays.append(temp)
        
        stacked = np.stack(temp_arrays)
        return stacked
    
    def render(self, q_values=None):
        return self.env.render(q_values=q_values)
    
class FeatureEncoder(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: Any) -> Any:
        board = observation["board"]

        pad_board = np.pad(board, pad_width=1, mode="constant", constant_values=Identifier.OFF_BOARD.value)
        windows = np.lib.stride_tricks.sliding_window_view(pad_board, (3, 3))
        windows = windows.reshape(64, 9)

        return windows
    
    def render(self, q_values=None):
        return self.env.render(q_values=q_values)
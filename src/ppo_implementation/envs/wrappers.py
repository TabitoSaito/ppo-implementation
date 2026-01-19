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
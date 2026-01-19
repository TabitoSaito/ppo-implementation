from ppo_implementation.train.train import train_loop
from ppo_implementation.utils.helper import build_agent
from ppo_implementation.envs.minesweeper import MinesweeperEnv
from ppo_implementation.envs.wrappers import OneHotEncodeBoard
import yaml
import os

env = MinesweeperEnv(render_mode="rgb_array")
env = OneHotEncodeBoard(env)

state, info = env.reset()

obs_shape = state.shape
act_shape = env.action_space.n

current_dir = os.path.dirname(os.path.abspath(__file__))
config = "config"
file = "default.yaml"
with open(os.path.join(current_dir, config, file)) as stream:
    config = yaml.safe_load(stream)

agent = build_agent(obs_shape, act_shape, config)

train_loop(agent, env, config, episodes=0, video_interval=50)

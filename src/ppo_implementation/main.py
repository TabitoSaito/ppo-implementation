from ppo_implementation.train.train import train_loop
from ppo_implementation.train.evaluate import eval_agent
from ppo_implementation.utils.helper import build_agent
import yaml
import gymnasium as gym
import datetime
import os

env = gym.make("CartPole-v1", render_mode="rgb_array")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

current_dir = os.path.dirname(os.path.abspath(__file__))
config = "config"
file = "default.yaml"
with open(os.path.join(current_dir, config, file)) as stream:
        config = yaml.safe_load(stream)

agent = build_agent(obs_dim, act_dim, config)

train_loop(agent, env, episodes=0, override=True, storage=datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S"), video_interval=20)

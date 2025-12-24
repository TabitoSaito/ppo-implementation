from ppo_implementation.train.train import train_loop
from ppo_implementation.agents.ppo_agent import PPOAgent
from ppo_implementation.networks.networks import PolicyNet, ValueNet
from ppo_implementation.buffers.buffer import RolloutBuffer
from ppo_implementation.train.evaluate import eval_agent
import gymnasium as gym

env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy = PolicyNet(obs_dim, act_dim)
value = ValueNet(obs_dim)

buffer = RolloutBuffer(64 * 32, obs_dim)

agent = PPOAgent(policy, value, buffer)

train_loop(agent, env)

print(eval_agent(agent, env))

env = gym.make("CartPole-v1", render_mode="human")
eval_agent(agent, env, episodes=1)

input()

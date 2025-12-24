import torch
import numpy as np


def eval_agent(agent, env, episodes=10):
    scores = []
    for i in range(episodes):
        obs, _ = env.reset()
        score = 0
        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, _, _ = agent.act(obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            obs = next_obs
            score += reward
            if done:
                scores.append(score)
                break

    return np.mean(scores)

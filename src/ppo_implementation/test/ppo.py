import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from itertools import count
from .networks import PolicyNet, ValueNet
from .buffer import RolloutBuffer


def train_loop():
    env = gym.make("CartPole-v1")

    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim)
    value_fn = ValueNet(obs_dim)

    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()), lr=3e-4
    )

    clip_eps = 0.2
    batch_size = 64
    buffer_size = batch_size * 32
    epochs = 10

    steps = 0
    for t in range(20):
        obs, _ = env.reset()

        buffer = RolloutBuffer(buffer_size, obs_dim)

        # Rollout
        for _ in range(buffer_size):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            dist = policy(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = value_fn(obs_t)

            next_obs, reward, done, _, _ = env.step(action.item())
            buffer.add(obs_t, action, reward, done, log_prob.detach(), value.detach())

            obs = next_obs
            if done:
                obs, _ = env.reset()

            steps += 1

        buffer.compute_gae()

        # PPO Update
        for _ in range(epochs):
            idx = torch.randperm(buffer_size)
            for i in range(0, buffer_size, batch_size):
                b = idx[i : i + batch_size]

                dist = policy(buffer.obs[b])
                new_logp = dist.log_prob(buffer.actions[b])
                ratio = torch.exp(new_logp - buffer.log_probs[b])

                loss_pi = -torch.mean(
                    torch.min(
                        ratio * buffer.advantages[b],
                        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                        * buffer.advantages[b],
                    )
                )

                loss_v = torch.mean((value_fn(buffer.obs[b]) - buffer.returns[b]) ** 2)

                loss = loss_pi + 0.5 * loss_v - 0.01 * dist.entropy().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scores = []
        for ep in range(10):
            obs, _ = env.reset()
            score = 0
            while True:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                dist = policy(obs_t)
                action = dist.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action.item())

                obs = next_obs
                score += float(reward)
                if terminated or truncated:
                    scores.append(score)
                    break

        print(f"Steps: {steps}\t\tAvg score: {np.mean(scores):.2f}")

    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    score = 0
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        dist = policy(obs_t)
        action = dist.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        obs = next_obs
        score += float(reward)
        if terminated or truncated:
            print(f"Ended with reward {score}")
            break

    input()

from itertools import count
import torch


def train_loop(agent, env, episodes=0, batch_size=64):
    steps = 0
    try:
        for it in count():
            obs, _ = env.reset()
            agent.buffer.reset()
            for _ in range(agent.buffer.size):
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, value = agent.act(obs_t)

                next_obs, reward, terminated, truncated, _ = env.step(action.item())

                done = terminated or truncated

                agent.remember(obs_t, action, reward, done, log_prob, value)

                obs = next_obs
                if done:
                    obs, _ = env.reset()
                steps += 1
                if steps % 100 == 0:
                    print(f"\r{steps}", end="")

            agent.update(batch_size)

            if episodes == 0:
                continue
            if it >= episodes:
                break
    except KeyboardInterrupt:
        pass
    return agent

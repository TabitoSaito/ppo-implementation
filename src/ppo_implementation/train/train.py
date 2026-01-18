from itertools import count
import torch
from .evaluate import eval_agent
from ..utils.writer import writer, GLOBAL_STEPS
from ..utils.helper import save_video, get_mask
from ..agents.ppo_agent import PPOAgent


def train_loop(agent: PPOAgent, env, episodes=0, video_interval=100):
    baseline = eval_agent(agent, env, episodes=50)

    episode = 0

    try:
        for it in count():
            obs, info = env.reset()
            mask = get_mask(info)
            agent.buffer.reset()
            score = 0
            for _ in range(agent.buffer.size):
                GLOBAL_STEPS["env_steps"] += 1
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, value, logits = agent.act(obs_t, mask)

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                score += reward

                done = terminated or truncated

                agent.remember(obs_t, action, reward, done, log_prob, value, mask)

                obs = next_obs
                mask = get_mask(info)

                if done:
                    episode += 1
                    obs, info = env.reset()
                    writer.add_scalars(
                        "Reward", {"Reward": score, "Baseline": baseline}, episode
                    )
                    score = 0
                    try:
                        mask = info["mask"]
                        mask = torch.tensor(mask, dtype=torch.bool)
                    except KeyError:
                        mask = None

            agent.update()

            if (it + 1) % video_interval == 0:
                save_video(agent, env, GLOBAL_STEPS["env_steps"], "Video")
            if episodes == 0:
                continue
            if it >= episodes:
                break
        print("finish")
    except KeyboardInterrupt:
        pass
    return agent

from itertools import count
import torch
import numpy as np
from .evaluate import eval_agent
from ..utils.writer import writer, GLOBAL_STEPS
from ..utils.helper import save_video, get_mask
from ..agents.ppo_agent import PPOAgent


def train_loop(agent: PPOAgent, env, config, episodes=0, video_interval=100):
    baseline = eval_agent(agent, env, episodes=50)

    try:
        for it in count():
            obs, info = env.reset()
            mask = get_mask(info)
            agent.buffer.reset()
            score = 0

            action_value_mean_buffer = []
            action_value_min_buffer = []
            action_value_max_buffer = []
            action_value_std_buffer = []
            state_value_buffer = []

            for _ in range(agent.buffer.size):
                GLOBAL_STEPS["env_steps"] += 1
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, value, raw_logits = agent.act(obs_t, mask)

                action_value_mean_buffer.append(raw_logits.mean().item())
                action_value_min_buffer.append(raw_logits.min().item())
                action_value_max_buffer.append(raw_logits.max().item())
                action_value_std_buffer.append(raw_logits.std().item())

                state_value_buffer.append(value.item())

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                score += reward

                done = terminated or truncated

                agent.remember(obs_t, action, reward, done, log_prob, value, mask)

                obs = next_obs
                mask = get_mask(info)

                if done:
                    GLOBAL_STEPS["env_episode"] += 1
                    obs, info = env.reset()

                    writer.add_scalar(
                        "Reward/reward",
                        score,
                        GLOBAL_STEPS["env_episode"],
                    )
                    writer.add_scalar(
                        "Action value/mean",
                        np.mean(action_value_mean_buffer),
                        GLOBAL_STEPS["env_episode"],
                    )
                    writer.add_scalar(
                        "Action value/min",
                        np.mean(action_value_min_buffer),
                        GLOBAL_STEPS["env_episode"],
                    )
                    writer.add_scalar(
                        "Action value/max",
                        np.mean(action_value_max_buffer),
                        GLOBAL_STEPS["env_episode"],
                    )
                    writer.add_scalar(
                        "Action value/std",
                        np.mean(action_value_std_buffer),
                        GLOBAL_STEPS["env_episode"],
                    )
                    writer.add_scalar(
                        "State value/mean",
                        np.mean(state_value_buffer),
                        GLOBAL_STEPS["env_episode"],
                    )

                    score = 0
                    mask = get_mask(info)

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

    rewards = eval_agent(agent, env, episodes=50)

    writer.add_hparams(
        config,
        {
            "Steps": GLOBAL_STEPS["env_steps"],
            "Reward mean": np.mean(rewards),
            "Reward max": np.max(rewards),
            "Reward min": np.min(rewards),
            "Reward std": np.std(rewards),
            "Reward baseline": np.mean(baseline)
        },
    )

    return agent

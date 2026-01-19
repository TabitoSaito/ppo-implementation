import torch
import cv2
import subprocess
import os
from itertools import count
from ..utils.helper import get_mask
from ..agents.ppo_agent import PPOAgent


def eval_agent(agent: PPOAgent, env, episodes=10):
    scores = []
    for i in range(episodes):
        obs, info = env.reset()
        mask = get_mask(info)

        score = 0
        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, _, _, _ = agent.act(obs_t, mask)
            next_obs, reward, terminated, truncated, info = env.step(action.item())

            done = terminated or truncated

            obs = next_obs
            mask = get_mask(info)

            score += reward
            if done:
                scores.append(score)
                break

    return scores


def render_run(agent: PPOAgent, env, file_path):
    assert env.render_mode == "rgb_array"

    frames = []
    obs, info = env.reset()
    mask = get_mask(info)

    score = 0
    for t in count():
        frame = env.render()
        frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, _, _, _ = agent.act(obs_t, mask)
        obs, reward, terminated, truncated, info = env.step(action.item())

        mask = get_mask(info)

        score += reward
        if terminated or truncated:
            break

    frames.append(env.render())

    tmp_path = file_path + ".tmp.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        tmp_path,
        fourcc,
        env.metadata["render_fps"],
        (frames[0].shape[1], frames[0].shape[0]),
    )

    for f in frames:
        out.write(f[:, :, ::-1])

    out.release()
    env.close()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            tmp_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-loglevel",
            "panic",
            file_path,
        ],
        check=True,
    )

    os.remove(tmp_path)

    metadata = {
        "reward": score,
        "steps": t,
    }

    return metadata

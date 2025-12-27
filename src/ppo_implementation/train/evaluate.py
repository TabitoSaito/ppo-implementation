import torch
import numpy as np
import cv2
import subprocess
import os


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


def render_run(agent, env, file_path):
    assert env.render_mode == "rgb_array"

    frames = []
    obs, _ = env.reset()
    score = 0
    while True:
        frame = env.render()
        frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, _, _ = agent.act(obs_t)
        obs, reward, terminated, truncated, _ = env.step(action.item())

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

    return score
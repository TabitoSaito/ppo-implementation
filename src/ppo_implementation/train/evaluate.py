import torch
import numpy as np
import cv2


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
    while True:
        frame = env.render()
        frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, _, _ = agent.act(obs_t)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        done = terminated or truncated

        obs = next_obs
        if done:
            break

    frame = env.render()
    frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        file_path,
        fourcc,
        env.metadata["render_fps"],
        (frame.shape[1], frame.shape[0]),
    )
    for frame in frames:
        out.write(frame[:, :, ::-1])
    out.release()
    env.close()
import random
import torch
import numpy as np
from ..agents.ppo_agent import PPOAgent
from ..networks.networks import CNNPolicyNet, CNNValueNet, ScoringPolicyNet, ScoringValueNet
from ..buffers.buffer import RolloutBuffer
from .writer import writer


def build_agent(obs_shape, act_shape, config):
    obs_dim = obs_shape[0]

    match config["POLICY_NET"].lower():
        case "cnn":
            policy = CNNPolicyNet(obs_dim, act_shape)
        case "score":
            policy = ScoringPolicyNet(9)
        case _:
            raise ValueError(f"No policy network with name '{config["POLICY_NET"].lower()}'")

    match config["VALUE_NET"].lower():
        case "cnn":
            value = CNNValueNet(obs_dim, act_shape)
        case "score":
            value = ScoringValueNet(9)
        case _:
            raise ValueError(f"No value network with name '{config["VALUE_NET"].lower()}'")


    buffer = RolloutBuffer(obs_shape, act_shape, config)

    agent = PPOAgent(policy, value, buffer, config)

    return agent


def action_to_index(action, shape: tuple):
    row = int(action / shape[1])
    col = action % shape[1]
    return (row, col)


def generate_unique_coordinates(
    n, upper_bound_x, upper_bound_y, lower_bound_x=0, lower_bound_y=0, except_=[]
):
    cords = []

    for _ in range(n):
        while True:
            x = random.randint(lower_bound_x, upper_bound_x)
            y = random.randint(lower_bound_y, upper_bound_y)

            cord = [x, y]
            if cord in cords:
                continue
            elif cord[0] == except_[0] and cord[1] == except_[1]:
                continue
            else:
                cords.append(cord)
                break
    return list(map(list, zip(*cords)))


def index_in_bound(index: tuple[int, int], bound: tuple[int, int]):
    if not 0 <= index[0] < bound[0]:
        return False
    if not 0 <= index[1] < bound[1]:
        return False
    return True


def get_mask(info):
    try:
        mask = info["mask"]
        mask = torch.tensor(mask, dtype=torch.bool)
    except KeyError:
        mask = None
    return mask


def save_video(agent: PPOAgent, env, episode: int, name: str):
    obs, info = env.reset()

    mask = get_mask(info)

    frames = []
    done = False
    frame = env.render()
    frames.append(frame)
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action, _, _, _ = agent.act(obs_t, mask)
        next_obs, _, terminated, truncated, info = env.step(action.item())

        obs = next_obs
        mask = get_mask(info)

        done = terminated or truncated
        frame = env.render()
        frames.append(frame)

    frame_t = torch.from_numpy(np.stack(frames)).permute((0, 3, 1, 2)).unsqueeze(0)

    writer.add_video(name, frame_t, episode, fps=env.metadata["render_fps"])

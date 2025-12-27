from itertools import count
import torch
from .evaluate import eval_agent
import copy
from .plot import GraphPlotter
import multiprocessing


def train_loop(agent, env, episodes=0, batch_size=64, storage="run1", override=False):
    q = multiprocessing.Queue()
    styles = [
        {
            "name": "Reward",
            "color": "#1f77b4",
            "linestyle": "-",
            "hl": [{"y": 200, "color": "red", "style": "--", "label": "Target"}],
        }
    ]
    plotter = GraphPlotter(q, styles)
    plotter.get_storage(storage, override)
    plotter.start()

    eval_env = copy.deepcopy(env)
    try:
        for it in count():
            obs, _ = env.reset()
            agent.buffer.reset()
            score = 0
            for _ in range(agent.buffer.size):
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, value = agent.act(obs_t)

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                score += reward

                done = terminated or truncated

                agent.remember(obs_t, action, reward, done, log_prob, value)

                obs = next_obs
                
                if done:
                    obs, _ = env.reset()
                    q.put([score])
                    score = 0

            agent.update(batch_size)

            if episodes == 0:
                continue
            if it >= episodes:
                break
    except KeyboardInterrupt:
        pass
    plotter.stop()
    return agent

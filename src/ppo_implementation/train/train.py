from itertools import count
import torch
from .plot import GraphPlotter
import multiprocessing


def train_loop(agent, env, episodes=0, storage="run1", override=False, video_interval=100):
    stat_q = multiprocessing.Queue()
    video_q = multiprocessing.Queue()
    styles = [
        {
            "name": "Reward",
            "color": "#1f77b4",
            "linestyle": "-",
            "hl": [{"y": 200, "color": "red", "style": "--", "label": "Target"}],
        }
    ]
    plotter = GraphPlotter(stat_q, video_q, styles)
    plotter.get_storage(storage, override)
    plotter.start()

    episode = 0

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
                    episode += 1
                    obs, _ = env.reset()
                    stat_q.put([score])
                    score = 0

            agent.update()
            if it % video_interval == 0:
                video_q.put((agent, env, episode))
            if episodes == 0:
                continue
            if it >= episodes:
                break
    except KeyboardInterrupt:
        pass
    plotter.stop()
    return agent

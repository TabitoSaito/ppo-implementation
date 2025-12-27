from itertools import count
import torch
from .plot import GraphPlotter
from ..utils.helper import StateQueue, QueueBatch
import multiprocessing


def train_loop(agent, env, episodes=0, storage="run1", override=False, video_interval=100):
    video_q = multiprocessing.Queue()
    styles1 = [
        {
            "name": "Reward",
            "color": "#1f77b4",
            "linestyle": "-",
            "hl": [{"y": 200, "color": "red", "style": "--", "label": "Target"}],
        }
    ]
    styles2 = [
        {
            "name": "Action Loss",
            "color": "#1f77b4",
            "linestyle": "-",
        },
        {
            "name": "Value Loss",
            "color": "#1f77b4",
            "linestyle": "-",
        },
        {
            "name": "Entropy",
            "color": "#1f77b4",
            "linestyle": "-",
        },
        {
            "name": "Loss",
            "color": "#1f77b4",
            "linestyle": "-",
        }
    ]

    q1 = StateQueue(styles=styles1, unit="Episode")
    q2 = StateQueue(styles=styles2, unit="Updates")

    q_batch = QueueBatch([q1, q2])

    plotter = GraphPlotter(q_batch, video_q)
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
                    q1.put([score])
                    score = 0

            loss_pi, loss_v, entropy, loss = agent.update()
            for update in zip(loss_pi, loss_v, entropy, loss):
                q2.put(update)
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

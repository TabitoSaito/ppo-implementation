import os
import numpy as np
import multiprocessing
from ppo_implementation.agents.ppo_agent import PPOAgent
from ppo_implementation.networks.networks import PolicyNet, ValueNet
from ppo_implementation.buffers.buffer import RolloutBuffer


def create_folder_on_marker(folder: str, marker="src"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    while True:
        if marker in os.listdir(project_root):
            break
        parent = os.path.dirname(project_root)
        if parent == project_root:
            raise FileNotFoundError(
                f"Projekt-Root mit Marker '{marker}' nicht gefunden!"
            )
        project_root = parent

    target = os.path.join(project_root, marker, folder)
    os.makedirs(target, exist_ok=True)
    return target


def minmax_downsample(x, y, max_points=2000):
    n = len(x)
    if n <= max_points:
        return x, y

    bins = np.linspace(0, n, max_points, dtype=int)
    xs, ys = [], []

    for i in range(len(bins) - 1):
        seg = slice(bins[i], bins[i + 1])
        y_seg = y[seg]
        if len(y_seg) == 0:
            continue

        i_min = np.argmin(y_seg)
        i_max = np.argmax(y_seg)

        xs.extend([x[seg][i_min], x[seg][i_max]])
        ys.extend([y_seg[i_min], y_seg[i_max]])

    return np.array(xs), np.array(ys)


def build_agent(obs_dim, act_dim, config):
    policy = PolicyNet(obs_dim, act_dim)
    value = ValueNet(obs_dim)

    buffer = RolloutBuffer(obs_dim, config)

    agent = PPOAgent(policy, value, buffer, config)

    return agent


class StateQueue():
    def __init__(self, styles, unit) -> None:
        self.q = multiprocessing.Queue()
        self.styles = styles
        self.unit = unit

        self.cols = [[] for _ in styles]
        self.x = []
        self.steps = 0

    def put(self, *arg, **kwarg):
        self.q.put(*arg, **kwarg)

    def empty(self):
        return self.q.empty()

    def fetch_states(self):
        try:
            while True:
                item = self.q.get_nowait()
                self.steps += 1
                self.x.append(self.steps)
                for i, v in enumerate(item):
                    self.cols[i].append(v)
        except Exception:
            pass


class QueueBatch:
    def __init__(self, queues: list[StateQueue]) -> None:
        self.queues = queues

    def check_empty(self) -> bool:
        """check if all queues are empty

        Returns:
            bool: True if all empty
        """
        for q in self.queues:
            if q.q.empty():
                continue
            else:
                return False
        return True
    
    def fetch_states(self):
        for q in self.queues:
            q.fetch_states()

    def __iter__(self):
        for q in self.queues:
            yield q

    def __len__(self):
        return len(self.queues)

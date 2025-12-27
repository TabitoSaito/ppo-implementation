import os
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
from ..utils.helper import create_folder_on_marker, minmax_downsample, QueueBatch
from .evaluate import render_run


class GraphPlotter(multiprocessing.Process):
    def __init__(
        self, stat_queue_batch: QueueBatch, video_queue, interval=5, plt_size=(6, 6), max_points=2000
    ) -> None:
        super().__init__()
        self.stat_queue_batch = stat_queue_batch
        self.video_queue = video_queue
        self.interval = interval
        self.running = multiprocessing.Event()
        self.running.set()

        self.plt_size = plt_size
        self.max_points = max_points

        self.steps = 0
        self.df = None
        self.full_storage = None

    def get_storage(self, storage="run1", override=False):
        instance_dir = create_folder_on_marker("static", "server")
        self.storage = storage
        self.full_storage = os.path.join(instance_dir, storage)
        if os.path.exists(self.full_storage):
            if not override:
                raise FileExistsError(
                    "Run with that name already exists. To override set 'override' flag to true"
                )
        else:
            os.makedirs(os.path.join(self.full_storage, "imgs"))
            os.makedirs(os.path.join(self.full_storage, "vids"))
            with open(os.path.join(self.full_storage, "index.json"), "w") as f:
                json.dump({"imgs": [], "vids": []}, f)

    def run(self):
        while self.running.is_set() or not self.stat_queue_batch.check_empty() or not self.video_queue.empty():
            self.stat_queue_batch.fetch_states()

            self.plot_graphs()

            try:
                while True:
                    agent, env, episode = self.video_queue.get_nowait()
                    self.save_video(agent, env, episode)
            except Exception:
                pass

            time.sleep(self.interval)

    def plot_graphs(self):
        imgs = []
        for q in self.stat_queue_batch:
            x = np.asarray(q.x)
            cols = [np.asarray(c) for c in q.cols]

            for y, style in zip(cols, q.styles):
                xd, yd = minmax_downsample(x, y)

                fig, ax = plt.subplots(figsize=self.plt_size)
                ax.plot(
                    xd,
                    yd,
                    color=style.get("color", "blue"),
                    linestyle=style.get("linestyle", "-"),
                    label=style["name"]
                )

                hl_metadata = {}

                for h in style.get("hl", []):
                    ax.axhline(
                        h["y"],
                        color=h.get("color", "red"),
                        linestyle=h.get("linestyle", "--"),
                        label=h.get("label")
                    )

                    hl_metadata[h.get("label")] = h["y"]


                ax.set_title(style["name"])
                ax.grid(alpha=0.3)
                ax.legend()
                ax.legend(loc="upper left")
                ax.set_xlabel(q.unit)
                ax.grid(axis = 'y')
                ax.margins(0)
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                fig.tight_layout()

                out_file = os.path.join(self.full_storage, "imgs", f"{style["name"].replace(" ", "_")}.png")
                fig.savefig(out_file, dpi=150)
                plt.close(fig)

                if len(y) < 1:
                    metadata = hl_metadata
                else:
                    metadata = {
                        "min": y.min(),
                        "max": y.max(),
                        "mean": y.mean(),
                        "std": y.std(),
                        "min_100": y[-100:].min(),
                        "max_100": y[-100:].max(),
                        "mean_100": y[-100:].mean(),
                        "std_100": y[-100:].std(),
                        **hl_metadata
                    }

                img = {
                    "src": os.path.join(self.storage, "imgs", f"{style["name"].replace(" ", "_")}.png"),
                    "metadata": metadata
                }
                imgs.append(img)

        with open(os.path.join(self.full_storage, "index.json"), "r") as f:
            index = json.load(f)

        index["imgs"] = imgs
        with open(os.path.join(self.full_storage, "index.json"), "w") as f:
            json.dump(index, f)

    def save_video(self, agent, env, episode):
        agent = copy.deepcopy(agent)
        env = copy.deepcopy(env)

        out_file = os.path.join(self.full_storage, "vids", f"{episode}.mp4")
        try:
            data = render_run(agent, env, out_file)
        except Exception as e:
            data = {}
            print("Render failed with exception. ", e)

        with open(os.path.join(self.full_storage, "index.json"), "r") as f:
            index = json.load(f)

        vid = {
            "src": os.path.join(self.storage, "vids", f"{episode}.mp4"),
            "metadata": {
                "episode": episode,
                **data
            }
        }
        index["vids"].append(vid)
        with open(os.path.join(self.full_storage, "index.json"), "w") as f:
            json.dump(index, f)

    def stop(self):
        self.running.clear()
        self.join(timeout=self.interval + 4)



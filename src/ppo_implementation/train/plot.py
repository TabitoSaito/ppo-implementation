import os
import pandas as pd
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from ..utils.helper import create_folder_on_marker, minmax_downsample


class GraphPlotter(multiprocessing.Process):
    def __init__(
        self, queue, styles: list, interval=5, plt_size=(6, 6), max_points=2000
    ) -> None:
        super().__init__()
        self.queue = queue
        self.interval = interval
        self.running = multiprocessing.Event()
        self.running.set()

        self.plt_size = plt_size
        self.max_points = max_points

        self.steps = 0
        self.df = None
        self.storage = None
        self.styles = styles

        self.cols = [[] for _ in styles]
        self.x = []

    def get_storage(self, storage="run1", override=False):
        instance_dir = create_folder_on_marker("static", "server")
        self.storage = os.path.join(instance_dir, storage)
        if os.path.exists(self.storage):
            if not override:
                raise FileExistsError(
                    "Run with that name already exists. To override set 'override' flag to true"
                )
        else:
            os.makedirs(self.storage)

    def run(self):
        while self.running.is_set() or not self.queue.empty():
            try:
                while True:
                    item = self.queue.get_nowait()
                    self.steps += 1
                    self.x.append(self.steps)
                    for i, v in enumerate(item):
                        self.cols[i].append(v)
            except Exception:
                pass

            self.plot_graphs()
            time.sleep(self.interval)

    def plot_graphs(self):
        x = np.asarray(self.x)
        cols = [np.asarray(c) for c in self.cols]

        for y, style in zip(cols, self.styles):
            xd, yd = minmax_downsample(x, y)

            fig, ax = plt.subplots(figsize=self.plt_size)
            ax.plot(
                xd,
                yd,
                color=style.get("color", "blue"),
                linestyle=style.get("linestyle", "-"),
                label=style["name"]
            )

            for h in style.get("hl", []):
                ax.axhline(
                    h["y"],
                    color=h.get("color", "red"),
                    linestyle=h.get("linestyle", "--"),
                    label=h.get("label")
                )

            ax.set_title(style["name"])
            ax.grid(alpha=0.3)
            ax.legend()
            ax.legend(loc="upper left")
            ax.set_xlabel("Episode")
            ax.grid(axis = 'y')
            ax.margins(0)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            fig.tight_layout()

            out_file = os.path.join(self.storage, f"{style["name"].replace(" ", "_")}.png")
            fig.savefig(out_file, dpi=150)
            plt.close(fig)


    def stop(self):
        self.running.clear()
        self.join(timeout=self.interval + 4)

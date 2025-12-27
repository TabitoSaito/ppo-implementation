import os
import numpy as np
import pandas as pd


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

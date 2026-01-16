import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from ppo_implementation.envs.minesweeper import MinesweeperEnv

logdir = "runs"
host = "127.0.0.1"
port = 6007

tb = program.TensorBoard()
tb.configure(
    argv=[
        None,
        "--logdir", logdir,
        "--host", host,
        "--port", str(port),
    ]
)

url = tb.launch()
print(f"TensorBoard läuft unter: {url}")

env = MinesweeperEnv(render_mode="rgb_array")

env.reset()
frames = []
done = False
frame = env.render()
frames.append(frame)
while not done:
    _, _, terminated, truncated, _ = env.step(-1)
    done = terminated or truncated
    frame = env.render()
    frames.append(frame)

frame_t = torch.from_numpy(np.stack(frames)).permute((0, 3, 1, 2)).unsqueeze(0)

writer = SummaryWriter()

writer.add_video("test video", frame_t)

writer.close()

input()
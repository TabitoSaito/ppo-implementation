from torch.utils.tensorboard import SummaryWriter
import datetime

GLOBAL_STEPS = {
    "update_episodes": 0,
    "env_steps": 0,
    "env_episode": 0
}

writer = SummaryWriter(f"runs/{datetime.datetime.now()}")

layout = {
        "Charts": {
            "Action Value": ["Multiline", ["Action value/mean", "Action value/min", "Action value/max"]],
        },
    }

writer.add_custom_scalars(layout)

from torch.utils.tensorboard import SummaryWriter

GLOBAL_STEPS = {
    "update_episodes": 0,
    "env_steps": 0,
}

writer = SummaryWriter()

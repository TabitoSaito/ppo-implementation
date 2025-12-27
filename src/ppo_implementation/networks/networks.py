import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        if len(x.shape) < 1:
            x = x.unsqueeze(0)
        logits = self.net(x)
        return Categorical(logits=logits)


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if len(x.shape) < 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)

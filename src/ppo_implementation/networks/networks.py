import torch.nn as nn
import torch


class CNNPolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(CNNPolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(obs_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x, mask=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        logits = self.net(x)
        logits = torch.flatten(logits, start_dim=1)
        raw_logits = logits.clone()
        if mask is not None:
            logits.masked_fill_(mask, float("-inf"))
        return logits, raw_logits


class CNNValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(CNNValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(obs_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.net(x)
        x = x.mean(dim=(2, 3))
        return x.squeeze(-1)
    

class ScoringPolicyNet(nn.Module):
    def __init__(self, features, hidden_dim=64) -> None:
        super(ScoringPolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        batch_size, num_elements, num_features = x.shape
        x_flat = x.view(-1, num_features)

        scores = self.net(x_flat)

        logits = scores.view(batch_size, num_elements)

        raw_logits = logits.clone()
        if mask is not None:
            logits.masked_fill_(mask, float("-inf"))
        return logits, raw_logits
    

class ScoringValueNet(nn.Module):
    def __init__(self, features, hidden_dim=64) -> None:
        super(ScoringValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        batch_size, num_elements, num_features = x.shape
        x_flat = x.view(-1, num_features)

        scores = self.net(x_flat)

        logits = scores.view(batch_size, num_elements)

        values = logits.mean(1)

        return values.squeeze(-1)
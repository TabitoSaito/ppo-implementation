import torch


class RolloutBuffer:
    def __init__(self, obs_dim, config):
        self.size = config["BATCH_SIZE"] * config["BATCH_NUM"]
        self.obs_dim = obs_dim
        self.config = config
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.size, self.obs_dim)
        self.actions = torch.zeros(self.size, dtype=torch.long)
        self.rewards = torch.zeros(self.size)
        self.dones = torch.zeros(self.size)
        self.log_probs = torch.zeros(self.size)
        self.values = torch.zeros(self.size)

        self.advantages = torch.zeros(self.size)
        self.returns = torch.zeros(self.size)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self):
        adv = 0
        for t in reversed(range(self.ptr)):
            delta = (
                self.rewards[t]
                + self.config["GAMMA"]
                * (1 - self.dones[t])
                * (self.values[t + 1] if t + 1 < self.ptr else 0)
                - self.values[t]
            )
            adv = delta + self.config["GAMMA"] * self.config["LAMBDA"] * (1 - self.dones[t]) * adv
            self.advantages[t] = adv
            self.returns[t] = adv + self.values[t]

        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

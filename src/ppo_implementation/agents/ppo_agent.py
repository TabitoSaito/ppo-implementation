import torch.optim as optim
import torch


class PPOAgent:
    def __init__(self, policy_net, value_net, buffer, update_episodes=10, clip_eps=0.2) -> None:
        self.policy_net = policy_net
        self.value_net = value_net

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=3e-4,
        )

        self.buffer = buffer
        self.update_episodes = update_episodes
        self.clip_eps = clip_eps

    def act(self, state):
        dist = self.policy_net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)

        return action, log_prob, value
    
    def remember(self, state, action, reward, done, log_prob, value):
        self.buffer.add(state, action, reward, done, log_prob.detach(), value.detach())

    def update(self, batch_size):
        self.buffer.compute_gae()
        for _ in range(self.update_episodes):
            idx = torch.randperm(self.buffer.size)
            for i in range(0, self.buffer.size, batch_size):
                b = idx[i : i + batch_size]

                dist = self.policy_net(self.buffer.obs[b])
                new_logp = dist.log_prob(self.buffer.actions[b])
                ratio = torch.exp(new_logp - self.buffer.log_probs[b])

                loss_pi = -torch.mean(
                    torch.min(
                        ratio * self.buffer.advantages[b],
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * self.buffer.advantages[b],
                    )
                )

                loss_v = torch.mean((self.value_net(self.buffer.obs[b]) - self.buffer.returns[b]) ** 2)

                loss = loss_pi + 0.5 * loss_v - 0.01 * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

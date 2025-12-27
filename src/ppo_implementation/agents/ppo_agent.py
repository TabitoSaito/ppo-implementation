import torch.optim as optim
import torch
import numpy as np


class PPOAgent:
    def __init__(self, policy_net, value_net, buffer, config) -> None:
        self.policy_net = policy_net
        self.value_net = value_net

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=config["LR"],
        )

        self.config = config
        self.buffer = buffer

        

    def act(self, state):
        dist = self.policy_net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)

        return action, log_prob, value
    
    def remember(self, state, action, reward, done, log_prob, value):
        self.buffer.add(state, action, reward, done, log_prob.detach(), value.detach())

    def update(self):
        loss_pi_batch = []
        loss_v_batch = []
        entropy_batch = []
        loss_batch = []
        self.buffer.compute_gae()
        for _ in range(self.config["UPDATE_EPISODES"]):
            idx = torch.randperm(self.buffer.size)
            loss_pi_buffer = []
            loss_v_buffer = []
            entropy_buffer = []
            loss_buffer = []
            for i in range(0, self.buffer.size, self.config["BATCH_SIZE"]):
                b = idx[i : i + self.config["BATCH_SIZE"]]

                dist = self.policy_net(self.buffer.obs[b])
                new_logp = dist.log_prob(self.buffer.actions[b])
                ratio = torch.exp(new_logp - self.buffer.log_probs[b])

                loss_pi = -torch.mean(
                    torch.min(
                        ratio * self.buffer.advantages[b],
                        torch.clamp(ratio, 1 - self.config["CLIP"], 1 + self.config["CLIP"])
                        * self.buffer.advantages[b],
                    )
                )

                loss_v = torch.mean((self.value_net(self.buffer.obs[b]) - self.buffer.returns[b]) ** 2)

                loss = loss_pi + self.config["VALUE_DISCOUNT"] * loss_v - self.config["ENTROPY_COEF"] * dist.entropy().mean()

                loss_pi_buffer.append(loss_pi.item())
                loss_v_buffer.append(loss_v.item())
                entropy_buffer.append(dist.entropy().mean().item())
                loss_buffer.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_pi_batch.append(np.mean(loss_pi_buffer))
            loss_v_batch.append(np.mean(loss_v_buffer))
            entropy_batch.append(np.mean(entropy_buffer))
            loss_batch.append(np.mean(loss_buffer))

        return loss_pi_batch, loss_v_batch, entropy_batch, loss_batch
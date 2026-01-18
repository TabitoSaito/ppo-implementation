import torch.optim as optim
import torch
import numpy as np
from torch.distributions import Categorical
from ..utils.writer import writer, GLOBAL_STEPS


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

    def act(self, state, mask=None):
        logits = self.policy_net(state, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)

        return action, log_prob, value, logits

    def remember(self, state, action, reward, done, log_prob, value, mask=None):
        self.buffer.add(
            state, action, reward, done, log_prob.detach(), value.detach(), mask
        )

    def update(self):
        self.buffer.compute_gae()
        for _ in range(self.config["UPDATE_EPISODES"]):
            idx = torch.randperm(self.buffer.size)
            loss_pi_buffer = []
            loss_v_buffer = []
            entropy_buffer = []
            loss_buffer = []
            for i in range(0, self.buffer.size, self.config["BATCH_SIZE"]):
                b = idx[i : i + self.config["BATCH_SIZE"]]

                logits = self.policy_net(self.buffer.obs[b], self.buffer.masks[b])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(self.buffer.actions[b])
                ratio = torch.exp(new_logp - self.buffer.log_probs[b])

                loss_pi = -torch.mean(
                    torch.min(
                        ratio * self.buffer.advantages[b],
                        torch.clamp(
                            ratio, 1 - self.config["CLIP"], 1 + self.config["CLIP"]
                        )
                        * self.buffer.advantages[b],
                    )
                )

                loss_v = torch.mean(
                    (self.value_net(self.buffer.obs[b]) - self.buffer.returns[b]) ** 2
                )

                loss = (
                    loss_pi
                    + self.config["VALUE_DISCOUNT"] * loss_v
                    - self.config["ENTROPY_COEF"] * dist.entropy().mean()
                )

                loss_pi_buffer.append(loss_pi.item())
                loss_v_buffer.append(loss_v.item())
                entropy_buffer.append(dist.entropy().mean().item())
                loss_buffer.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            GLOBAL_STEPS["update_episodes"] += 1

            writer.add_scalar(
                "Agent/Action loss",
                np.mean(loss_pi_buffer),
                GLOBAL_STEPS["update_episodes"],
            )
            writer.add_scalar(
                "Agent/Value loss",
                np.mean(loss_v_buffer),
                GLOBAL_STEPS["update_episodes"],
            )
            writer.add_scalar(
                "Agent/Entropy loss",
                np.mean(entropy_buffer),
                GLOBAL_STEPS["update_episodes"],
            )
            writer.add_scalar(
                "Agent/loss", np.mean(loss_buffer), GLOBAL_STEPS["update_episodes"]
            )

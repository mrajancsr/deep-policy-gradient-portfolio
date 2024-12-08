from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ddpgportfolio.agent.models import Actor, Critic
from ddpgportfolio.dataset import (
    KrakenDataSet,
)
from ddpgportfolio.memory.memory import ExperienceReplayMemory, PortfolioVectorMemory
from ddpgportfolio.portfolio.portfolio import Portfolio
from utilities.noise import OrnsteinUhlenbeckNoise

torch.set_default_device("mps")


@dataclass
class DDPGAgent:
    """Implementation of Deep Deterministic Policy Gradient for the Agent"""

    portfolio: Portfolio
    batch_size: int
    window_size: int
    step_size: int
    n_epochs: int
    learning_rate: Optional[float] = 3e-5
    betas: Optional[Tuple[float, float]] = (0.0, 0.9)
    device: Optional[str] = "mps"
    actor: Actor = field(init=False)
    critic: Critic = field(init=False)
    target_actor: nn.Module = field(init=False)
    target_critic: nn.Module = field(init=False)
    actor_optimizer: torch.optim = field(init=False)
    critic_optimizer: torch.optim = field(init=False)
    loss_fn: nn.modules.loss.MSELoss = field(init=False)
    dataloader: DataLoader = field(init=False)
    pvm: PortfolioVectorMemory = field(init=False)
    replay_memory: ExperienceReplayMemory = field(init=False)
    ou_noise: OrnsteinUhlenbeckNoise = field(init=False)
    gamma: float = 0.90
    tau: float = 0.005

    def __post_init__(self):
        # create dataset and dataloaders for proper iteration
        kraken_ds = KrakenDataSet(self.portfolio, self.window_size, self.step_size)

        self.dataloader = DataLoader(
            kraken_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            generator=torch.Generator(device=self.device),
        )

        # create actor and critic networks and specify optimizers and learning rates
        m_noncash_assets = self.portfolio.m_noncash_assets
        self.actor = Actor(3, m_noncash_assets)
        self.critic = Critic(3, m_noncash_assets)
        self.target_actor = self.clone_network(self.actor)
        self.target_critic = self.clone_network(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

        # loss function for the critic
        self.loss_fn = nn.MSELoss()

        # initializing pvm with all cash initially
        self.pvm = PortfolioVectorMemory(self.portfolio.n_samples, m_noncash_assets)
        self.pvm.update_memory_stack(
            torch.zeros(m_noncash_assets), self.window_size - 2
        )
        self.replay_memory = ExperienceReplayMemory()
        self.update_target_networks()
        self.ou_noise = OrnsteinUhlenbeckNoise(1)

    def clone_network(self, network):
        cloned_network = type(network)()
        cloned_network.load_state_dict(network.state_dict())
        return cloned_network

    def select_action(self, state):
        """Select action using the actor's policy (deterministic action)"""
        self.actor.eval()  # Ensure the actor is in evaluation mode
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        return action

    def update_target_networks(self):
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

    def soft_update(
        self, target_network: nn.Module, main_network: nn.Module, tau: float
    ):
        """_summary_

        Parameters
        ----------
        target_network : nn.Module
            _description_
        main_network : nn.Module
            _description_
        tau : float
            _description_
        """
        for target_param, main_param in zip(
            target_network.parameters(), main_network.parameters()
        ):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )

    def train_actor(self, state: Tuple[torch.tensor, torch.tensor]):
        self.actor_optimizer.zero_grad()
        predicted_actions = self.actor(state)

        # compute the actor loss using deterministic policy gradient
        q_values = self.critic(state, predicted_actions).mean()
        actor_loss = -q_values.mean()

        # perform backprop
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def train_critic(self, state, action, reward, next_state):
        """Train the critic network by minimizing the loss based on TD Error

        Parameters
        ----------
        state : _type_
            _description_
        action : _type_
            _description_
        reward : _type_
            _description_
        next_state : _type_
            _description_
        done : function
            _description_
        """
        self.critic_optimizer.zero_grad()
        # get predicted q values from current batch
        predicted_q_values = self.critic(state, action)

        with torch.no_grad():
            # get the next q values using the target critic and next state (from target network)
            next_action = self.target_actor(next_state)
            next_q_values = self.target_critic(next_state, next_action)

            # calculate target q values using bellman equation
            target_q_values = reward + self.gamma * next_q_values

        # compute the critic loss using MSE between predicted Q-values and target Q-values
        # Hence we are minimizing the TD Error
        critic_loss = self.loss_fn(predicted_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def train(self):
        print("Training Started for DDPG Agent")
        # scheduler to perform learning rate decay
        critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=0.95
        )
        actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=0.95
        )
        for epoch in range(self.n_epochs):

            total_actor_loss = 0
            total_critic_loss = 0
            total_reward = 0
            total_batches = 0

            for idx_batch, (xt_batch, prev_index_batch) in enumerate(self.dataloader):
                batch_actor_loss = 0
                batch_critic_loss = 0
                mini_batch_size = xt_batch.shape[0]
                for i in range(1, mini_batch_size):
                    xt, xt_next = xt_batch[i - 1], xt_batch[i]
                    prev_index, next_index = (
                        prev_index_batch[i - 1],
                        prev_index_batch[i],
                    )

                    # get the previous weights from portfolio vector memory
                    previous_action = self.pvm.get_memory_stack(prev_index)
                    state = (xt, previous_action)

                    # get current weight from actor network given s = (Xt, wt_prev)
                    action = self.select_action(state)

                    # store the current action back into pvm
                    self.pvm.update_memory_stack(action.detach(), next_index)

                    # get the relative price vector from price tensor to calculate reward
                    yt = 1 / xt[0, :, -2]
                    reward = (
                        self.portfolio.get_reward(action, yt, previous_action)
                        * self.portfolio.get_initial_portfolio_value()
                    )
                    next_state = (xt_next, action)
                    self.replay_memory.add(state, action, reward, next_state)

                    # we sample every 10 time steps
                    if i % 10 == 0:
                        state, action, reward, next_state = self.replay_memory.sample()
                    # train the critic
                    critic_loss = self.train_critic(state, action, reward, next_state)
                    batch_critic_loss += critic_loss
                    # train the actor
                    actor_loss = self.train_actor(state)
                    batch_actor_loss += actor_loss
                    self.update_target_networks()
                    total_reward += reward.item()
                total_batches += 1

                print(
                    f"Batch {idx_batch + 1} - Actor Loss: {batch_actor_loss / mini_batch_size:.4f}, Critic Loss: {batch_critic_loss / self.batch_size:.4f}"
                )

            total_critic_loss += batch_critic_loss / mini_batch_size
            total_actor_loss += batch_actor_loss / mini_batch_size
            # avg_reward = total_reward / total_batches

            # Log batch-specific losses (optional)
            print(
                f"Batch {idx_batch + 1} - Actor Loss: {batch_actor_loss / self.batch_size:.4f}, Critic Loss: {batch_critic_loss / self.batch_size:.4f}"
            )
            # Step the scheduler to adjust the learning rate
            critic_scheduler.step()
            actor_scheduler.step()
        # After processing the entire dataset or epoch, log total losses
        print(f"Total Actor Loss for Epoch: {total_actor_loss:.4f}")
        print(f"Total Critic Loss for Epoch: {total_critic_loss:.4f}")
        print("Done Training")
        print("Stop")


def plot_losses(actor_losses, critic_losses):
    """Plot the actor and critic losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Actor and Critic Losses during Training")
    plt.legend()
    plt.show()

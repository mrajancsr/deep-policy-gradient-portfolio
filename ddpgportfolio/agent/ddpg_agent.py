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
from ddpgportfolio.memory.memory import (
    Experience,
    PortfolioVectorMemory,
    PrioritizedReplayMemory,
)
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
    n_episodes: int
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
    replay_memory: PrioritizedReplayMemory = field(init=False)
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
        self.replay_memory = PrioritizedReplayMemory(
            capacity=100000,
        )
        self.update_target_networks()
        self.ou_noise = OrnsteinUhlenbeckNoise(1)

    def clone_network(self, network):
        cloned_network = type(network)()
        cloned_network.load_state_dict(network.state_dict())
        return cloned_network

    def select_action(self, state, exploration: bool = False):
        """Select action using the actor's policy (deterministic action)"""
        self.actor.eval()  # Ensure the actor is in evaluation mode
        with torch.no_grad():
            action_logits = self.actor(state)
        if exploration:
            noise = self.ou_noise.sample()
            action = torch.softmax(action_logits.view(-1) + noise, dim=0)
        else:
            action = torch.softmax(action_logits, dim=0)
        # return all non-cash weights
        return action[1:]

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

    def train_actor(self, experience: Experience, is_weights: torch.tensor):
        self.actor_optimizer.zero_grad()
        logits = self.actor(experience.state)
        predicted_actions = torch.softmax(logits.view(-1), dim=-1)
        xt, previous_noncash_actions = experience.state
        cash_weight_previous = 1 - previous_noncash_actions.sum()
        previous_action = torch.cat(
            [cash_weight_previous.unsqueeze(0), previous_noncash_actions], dim=0
        )
        state = (xt, previous_action)
        # compute the actor loss using deterministic policy gradient
        q_values = self.critic(state, predicted_actions).mean()
        actor_loss = -q_values.mean()
        actor_loss = (actor_loss * is_weights).mean()
        # perform backprop

        actor_loss.backward()
        self.actor_optimizer.step()
        return predicted_actions[1:], actor_loss.item()

    def train_critic(self, experience: Experience, is_weights):
        """Train the critic by minimizing the loss based on TD Error

        Parameters
        ----------
        experience : Experience
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.critic_optimizer.zero_grad()
        # critic needs to evaluate good an action is in a state
        # hence we need to add the cash weight back otherwise its biased
        xt, previous_noncash_actions = experience.state
        reward = torch.tensor(experience.reward, dtype=torch.float32)
        cash_weight_previous = 1 - previous_noncash_actions.sum()

        # previous action includes cash weight now
        previous_action = torch.cat(
            [cash_weight_previous.unsqueeze(0), previous_noncash_actions], dim=0
        )
        # construct st = (Xt, wt-1)
        state = (xt, previous_action)

        # we need to do the same for action wt at time t
        noncash_actions = experience.action
        cash_weight_action = 1 - noncash_actions.sum()
        actions = torch.cat([cash_weight_action.unsqueeze(0), noncash_actions], dim=0)
        predicted_q_values = self.critic(state, actions)

        with torch.no_grad():
            # the target actor uses next state from replay buffer
            logits = self.target_actor(experience.next_state)
            next_target_action = torch.softmax(logits.view(-1), dim=-1)
            # since previous action in next state is current action in current state
            xt_next = experience.next_state[0]
            next_state = (xt_next, actions)
            next_q_values = self.target_critic(next_state, next_target_action)

            # calculate target q values using bellman equation
            td_target = reward + self.gamma * next_q_values

        # compute the critic loss using MSE between predicted Q-values and target Q-values
        # Hence we are minimizing the TD Error
        td_error = td_target - predicted_q_values
        critic_loss = torch.mean(td_error**2)
        critic_loss = critic_loss * is_weights

        critic_loss.backward()
        self.critic_optimizer.step()
        return td_error, critic_loss.item()

    def pre_train(self):
        """Pretraining the ddpg agent by populating the experience replay buffer"""
        print("pre-training ddpg agent started...")
        print("ReplayMemoryBuffer populating with experience...")
        kraken_ds = KrakenDataSet(self.portfolio, self.window_size, self.step_size)
        n_samples = len(kraken_ds)
        for i in range(1, n_samples):
            xt, prev_index = kraken_ds[i - 1]
            previous_action = self.pvm.get_memory_stack(prev_index)
            state = (xt, previous_action)
            # get current weight from actor network given s = (Xt, wt_prev)
            action = self.select_action(state, exploration=True).detach()
            # store the current action into pvm
            self.pvm.update_memory_stack(action, prev_index + 1)
            # get the relative price vector from price tensor to calculate reward
            yt = 1 / xt[0, :, -2]
            reward = self.portfolio.get_reward(action, yt, previous_action)
            xt_next, _ = kraken_ds[i]
            next_state = (xt_next, action)
            experience = Experience(
                state, action, reward.item(), next_state, prev_index
            )
            self.replay_memory.add(experience=experience, reward=reward.item())
        print("pretraining done")
        print(f"buffer size: {len(self.replay_memory)}")
        # we subtract one since each experience consists of current state and next state
        assert len(self.replay_memory) == n_samples - 1

    def train(self):
        if len(self.replay_memory) == 0:
            raise Exception("replay memory is empty.  Please pre-train agent")

        print("Training Started for DDPG Agent")
        # scheduler to perform learning rate decay
        critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=0.95
        )
        actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=0.95
        )
        for episode in range(self.n_episodes):

            # total_actor_loss = 0
            # total_critic_loss = 0
            # total_reward = 0
            # total_batches = 0
            # batch_actor_loss = 0
            # batch_critic_loss = 0

            experiences, indices, is_weights = self.replay_memory.sample(
                self.batch_size
            )
            for experience in experiences:
                # get the td errors and update replay memory
                td_errors, critic_loss = self.train_critic(experience)
                self.replay_memory.update_priorities(indices, td_errors)
                action, actor_loss = self.train_actor(experience)
                self.pvm.update_memory_stack(
                    action.detach(), experience.previous_index + 1
                )

                # compute relative price vector to calculate reward
                yt = 1 / experience.state[0][0, :, -2]
                reward = self.portfolio.get_reward(
                    action.detach(), yt, experience.state[1]
                )

                # create a new experience and add it to replay memory
                xt, previous_action = experience.state
                state = (xt, previous_action)
                xt_next = experience.next_state[0]
                next_state = (xt_next, action.detach())
                previous_index = experience.previous_index
                new_experience = Experience(
                    state, action, reward, next_state, previous_index
                )
                self.replay_memory.add(experience=new_experience, reward=reward)

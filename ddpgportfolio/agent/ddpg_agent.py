from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
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
from utilities.pg_utils import (
    OrnsteinUhlenbeckNoise,
    RewardNormalizer,
    normalize_batch_rewards,
    plot_performance,
)

torch.set_default_device("mps")


@dataclass
class DDPGAgent:
    """Implementation of Deep Deterministic Policy Gradient for the Agent"""

    portfolio: Portfolio
    batch_size: int
    window_size: int
    step_size: int
    learning_rate: Optional[float] = 3e-5
    betas: Optional[Tuple[float, float]] = (0.0, 0.9)
    device: Optional[str] = "mps"
    actor: Actor = field(init=False)
    critic: Critic = field(init=False)
    target_actor: nn.Module = field(init=False)
    target_critic: nn.Module = field(init=False)
    actor_optimizer: torch.optim = field(init=False)
    critic_optimizer: torch.optim = field(init=False)
    dataloader: DataLoader = field(init=False)
    pvm: PortfolioVectorMemory = field(init=False)
    replay_memory: PrioritizedReplayMemory = field(init=False)
    ou_noise: OrnsteinUhlenbeckNoise = field(init=False)

    gamma: float = 0.7
    tau: float = 0.01
    epsilon: float = 1.0
    epsilon_max: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay_rate: float = 1e-5
    episode_count: int = 0
    warmup_steps: int = 6000
    min_buffer_warmup_steps: int = 5000
    ou_params: Dict[str, float] = field(
        default_factory=lambda: {"sigma": 0.4, "theta": 0.2}
    )

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
        m_assets = self.portfolio.m_assets
        m_noncash_assets = self.portfolio.m_noncash_assets
        self.actor = Actor(3, m_noncash_assets)
        self.critic = Critic(3, m_assets)
        self.target_actor = self.clone_network(self.actor)
        self.target_critic = self.clone_network(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=3e-5,
            weight_decay=1e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=5e-5,
            weight_decay=1e-5,
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

        # ou noise initialization
        theta = self.ou_params["theta"]
        sigma = self.ou_params["sigma"]
        self.ou_noise = OrnsteinUhlenbeckNoise(size=m_assets, theta=theta, sigma=sigma)

        # initializing pvm
        self.pvm = PortfolioVectorMemory(self.portfolio.n_samples, m_noncash_assets)

        self.replay_memory = PrioritizedReplayMemory(
            capacity=20000,
        )
        self.update_target_networks()

    def clone_network(self, network):
        cloned_network = type(network)()
        cloned_network.load_state_dict(network.state_dict())
        return cloned_network

    def select_random_action(self, m):
        uniform_vec = np.random.rand(m)
        return torch.tensor(uniform_vec / np.sum(uniform_vec), dtype=torch.float32)

    def select_action(
        self,
        state: Tuple[torch.tensor, torch.tensor],
        exploration: bool = False,
        action_type: Union[str, str] = "greedy",
    ):
        """Select action using the actor's policy (deterministic action)

        Parameters
        ----------
        state : Tuple[torch.tensor, torch.tensor]
            contains (Xt, wt-1)
        exploration : bool, optional
            whether to explore with agent, by default False

        Returns
        -------
        if no exploration, returns deterministic_actions
        otherwise, returns (noise, deterministic_actions)
        """
        with torch.no_grad():
            self.actor.eval()
            action_logits = self.actor(state)
            actions = torch.softmax(action_logits.view(-1), dim=-1)
        if exploration:
            if action_type == "hybrid":
                if self.episode_count < self.warmup_steps:
                    random_action = self.select_random_action(self.portfolio.m_assets)
                    self.episode_count += 1
                    return random_action, actions
                else:
                    # transition to ou noise after warm up steps
                    action_type = "ou"

            if action_type == "ou":
                noise = self.ou_noise.sample()
                noisy_logits = action_logits + noise
                noisy_actions = torch.softmax(noisy_logits.view(-1), dim=-1)
                return noisy_actions, actions

            elif action_type == "greedy" and np.random.rand() < self.epsilon:
                random_actions = self.select_random_action(self.portfolio.m_assets)
                self.update_epsilon()
                return random_actions, actions
        return actions

    def update_epsilon(self):
        if self.episode_count < self.warmup_steps:
            self.epsilon = (
                self.epsilon_max
                - ((self.epsilon_max - 0.5) / self.warmup_steps) * self.episode_count
            )

        else:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon
                * np.exp(
                    -self.epsilon_decay_rate * (self.episode_count - self.warmup_steps)
                ),
            )
        self.episode_count += 1

    def update_target_networks(self):
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

    def soft_update(
        self, target_network: nn.Module, main_network: nn.Module, tau: float
    ):
        """Peroforms a softupdate of target networks using ewma

        Parameters
        ----------
        target_network : nn.Module
            one of actor or critic
        main_network : nn.Module
           one of actor or critic
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
        """trains the actor network by maximizing the Q Value from Critic

        Parameters
        ----------
        experience : Experience
            _description_
        is_weights : torch.tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.actor.train()
        self.actor_optimizer.zero_grad()
        xt, previous_action = experience.state
        state = (xt, previous_action[:, 1:])
        action_logits = self.actor(state)
        predicted_actions = torch.softmax(action_logits, dim=-1)

        # actor has to choose action that maximizes the q value
        # hence we compute the q value and maximize this value
        q_values = self.critic(experience.state, predicted_actions)
        actor_loss = -q_values.mean()
        actor_loss = (actor_loss * is_weights).mean()

        # perform backprop
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        return actor_loss.item()

    def _normalize_batch_rewards(self, rewards):
        mean = rewards.mean()
        std = rewards.std()
        return (rewards - mean) / (std + 1e-5)

    def train_critic(self, experience: Experience, is_weights):
        """Train the critic by minimizing loss based on TD Error

        Parameters
        ----------
        experience : Experience
            an object consisting of (st, at, rt, st+1)
            st = (Xt, at-1)
            size of each experience is given by batch_size
        is_weights : bool
            importance sampling weights

        Returns
        -------
        Tuple[torch.tensor, float]
            (td error, critic loss)
        """
        self.critic.train()
        self.critic_optimizer.zero_grad()

        xt, previous_action = experience.state
        reward = experience.reward
        # construct st = (Xt, wt-1)
        state = (xt, previous_action)

        # we need to do the same for action wt at time t
        noisy_actions = experience.action
        # q(st, at)
        predicted_q_values = self.critic(state, noisy_actions)

        with torch.no_grad():
            # the target actor uses next state from replay buffer
            xt_next, prev_action_next = experience.next_state
            # remove cash from action and reconstruct the state
            next_state = (xt_next, prev_action_next[:, 1:])
            # the actor only takes non cash assets as input
            next_target_logits = self.target_actor(next_state)
            next_target_action = torch.softmax(next_target_logits, dim=1)
            # q(st+1, at+1)
            target_q_values = self.target_critic(
                experience.next_state, next_target_action
            )

            # calculate target q values using bellman equation
            td_target = reward + self.gamma * target_q_values

        # compute the critic loss using MSE between predicted Q-values and target Q-values
        # Hence we are minimizing the TD Error
        td_error = td_target - predicted_q_values
        critic_loss = torch.mean(td_error**2)
        critic_loss = (critic_loss * is_weights).mean()

        assert not torch.any(torch.isnan(td_error)), "NaN in TD error!"

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        return td_error, critic_loss.item()

    def train(self, num_episodes: int = 2, action_type: str = "ou"):
        """Train the agent by training the actor and critic networks

        Parameters
        ----------
        n_episodes : int, optional
            _description_, by default 50
        n_iterations_per_episode : int, optional
            _description_, by default 20

        Raises
        ------
        Exception
            _description_
        """

        print("Training Started for DDPG Agent")

        # scheduler to perform learning rate decay
        critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=100, gamma=0.9
        )
        actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=100, gamma=0.9
        )

        # store the losses and rewards from updating actor and critic networks
        critic_losses = []
        actor_losses = []
        rewards = []

        # dataset to get the price tensor and previous weights
        kraken_ds = KrakenDataSet(self.portfolio, self.window_size, self.step_size)
        n_samples = len(kraken_ds) + 49

        batch_size = self.batch_size

        for episode in range(num_episodes):

            # Initialize accumulators for the losses
            episode_actor_loss = 0
            episode_critic_loss = 0
            total_normalized_reward = 0
            true_episodic_reward = 0

            # initialize weights in pvm starting with cash only
            self.pvm = PortfolioVectorMemory(
                self.portfolio.n_samples, self.portfolio.m_assets
            )
            all_cash = torch.zeros(self.portfolio.m_assets)
            all_cash[0] = 1.0
            self.pvm.update_memory_stack(all_cash, self.window_size - 2)

            # reset noise parameters
            self.epsilon = 1.0
            self.ou_noise.reset_sigma(episode)
            print(f"episode {episode + 1} sigma: {self.ou_noise.sigma}")
            self.episode_count = 0

            # equity curve of agent
            previous_portfolio_value = self.portfolio.get_initial_portfolio_value()
            equity_curve = [previous_portfolio_value]

            # normalizer to normalize rewards
            reward_normalizer = RewardNormalizer()

            # Loop over timesets in current episode
            for t in range(1, n_samples):
                # get the price tensor from dataset and construct state
                Xt, prev_index = kraken_ds[t - 1]
                previous_action = self.pvm.get_memory_stack(prev_index)
                state = (Xt, previous_action[1:])

                # select actions by exploring using ou policy
                noisy_action, action = self.select_action(
                    state, exploration=True, action_type=action_type
                )

                # decay sigma
                dt = t / n_samples
                self.ou_noise.decay_sigma(dt, 0.0003)

                # execute action and compute rewards
                # get the relative price vector from price tensor to calculate reward
                yt = 1 / Xt[0, :, -2]
                reward = self.portfolio.get_reward(action, yt, previous_action)
                true_episodic_reward += reward.item()

                # get current portfolio value
                current_portfolio_value = self.portfolio.update_portfolio_value(
                    previous_portfolio_value, reward
                )
                equity_curve.append(current_portfolio_value.item())
                previous_portfolio_value = current_portfolio_value

                # normalize the rewards
                reward_normalizer.update(reward.item())
                normalized_reward = reward_normalizer.normalize(reward.item())

                # create state and next state
                state = (Xt, previous_action)
                Xt_next, _ = kraken_ds[t]
                next_state = (Xt_next, action)

                # construct transition to store into replay buffer
                experience = Experience(
                    state,
                    noisy_action,
                    normalized_reward,
                    next_state,
                    prev_index,
                )

                # update the portfolio vector memory with current action
                self.pvm.update_memory_stack(action.detach(), prev_index + 1)

                self.replay_memory.add(experience=experience)

                if len(self.replay_memory) < self.min_buffer_warmup_steps:
                    continue

                # sample a random mini-batch from replay memory
                experiences, indices, is_weights = self.replay_memory.sample(
                    batch_size=batch_size
                )

                # Update critic (TD Error)
                td_error, critic_loss = self.train_critic(experiences, is_weights)

                # Update actor (deterministic policy gradient)
                actor_loss = self.train_actor(experiences, is_weights)

                if t % 50 == 0:
                    self.update_target_networks()

                # Update priorities in the replay buffer (for prioritized experience replay)
                self.replay_memory.update_priorities(indices, td_error, 0.3)

                # Accumulate the losses over the iterations for logging
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                total_normalized_reward += experiences.reward.sum()

                if t % 2000 == 0:
                    print(f"iteration {t} sigma: {self.ou_noise.sigma}")

            # Update the learning rate scheduler
            critic_scheduler.step()
            actor_scheduler.step()

            # After training during the episode, log the average losses
            avg_episode_actor_loss = episode_actor_loss / n_samples
            avg_episode_critic_loss = episode_critic_loss / n_samples
            actor_losses.append(avg_episode_actor_loss)
            critic_losses.append(avg_episode_critic_loss)
            rewards.append(true_episodic_reward)

            print(
                f"Episode {episode + 1} - Actor Loss: {avg_episode_actor_loss:.4f}, Critic Loss: {avg_episode_critic_loss:.4f}, Total Normalized Reward in Episode: {total_normalized_reward:.4f}, True Episodic Reward: {true_episodic_reward:.4f}"
            )

        print("Training complete!")
        # performance plots
        plot_performance(actor_losses, critic_losses, rewards)

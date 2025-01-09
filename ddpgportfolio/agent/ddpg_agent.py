from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ddpgportfolio.agent.models import Actor, Critic
from ddpgportfolio.dataset import (
    KrakenDataSet,
)
from ddpgportfolio.environment.environment import TradeEnv
from ddpgportfolio.memory.memory import (
    Experience,
    PortfolioVectorMemory,
    PrioritizedReplayMemory,
)
from ddpgportfolio.portfolio.portfolio import Portfolio
from utilities.pg_utils import (
    OrnsteinUhlenbeckNoise,
    RewardNormalizer,
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
    pvm: PortfolioVectorMemory = field(init=False)
    replay_memory: PrioritizedReplayMemory = field(init=False)
    ou_noise: OrnsteinUhlenbeckNoise = field(init=False)

    gamma: float = 0.9
    tau: float = 0.001
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
        # create actor and critic networks and specify optimizers and learning rates
        m_assets = self.portfolio.m_assets
        m_noncash_assets = self.portfolio.m_noncash_assets
        window_size = self.window_size
        self.actor = Actor(3, window_size, m_noncash_assets)
        self.critic = Critic(3, window_size, m_assets)
        self.target_actor = self.clone_network(self.actor)
        self.target_critic = self.clone_network(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-5,
            weight_decay=1e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=1e-6,
            weight_decay=1e-5,
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        # self.target_critic.to(self.device)

        # ou noise initialization
        theta = self.ou_params["theta"]
        sigma = self.ou_params["sigma"]
        self.ou_noise = OrnsteinUhlenbeckNoise(size=m_assets, theta=theta, sigma=sigma)

        # initializing pvm
        self.pvm = PortfolioVectorMemory(self.portfolio.n_samples, m_noncash_assets)

        self.replay_memory = PrioritizedReplayMemory(
            capacity=500000,
        )
        self.update_target_networks()

    def clone_network(self, network):
        input_channel = network.input_channels
        look_back_window = network.look_back
        output_dim = network.output_dim
        cloned_network = type(network)(input_channel, look_back_window, output_dim)
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
            # input to actor is previous action with no cash
            curr_state = (state[0], state[1][1:])
            action_logits = self.actor(curr_state)
        if exploration:
            if action_type == "hybrid":
                if self.episode_count < self.warmup_steps:
                    random_action = self.select_random_action(self.portfolio.m_assets)
                    self.episode_count += 1
                    return random_action
                else:
                    # transition to ou noise after warm up steps
                    action_type = "ou"

            if action_type == "ou":
                noise = self.ou_noise.sample()
                noisy_logits = action_logits + noise
                noisy_actions = torch.softmax(noisy_logits.view(-1), dim=-1)
                return noisy_actions

            elif action_type == "greedy" and np.random.rand() < self.epsilon:
                random_actions = self.select_random_action(self.portfolio.m_assets)
                self.update_epsilon()
                return random_actions

        actions = torch.softmax(action_logits.view(-1), dim=-1)
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

        # input to actor is price tensor Xt and previous weights with no cash
        action_logits = self.actor((experience.state[0], experience.state[1][:, 1:]))
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

        reward = experience.reward
        # q(st, at)
        predicted_q_values = self.critic(experience.state, experience.action)

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
        """trains actor and critic networks

        Parameters
        ----------
        num_episodes : int, optional
            _description_, by default 2
        action_type : str, optional
            _description_, by default "ou"
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
        env = TradeEnv(kraken_ds, self.window_size)
        n_samples = len(kraken_ds)

        batch_size = self.batch_size

        for episode in range(num_episodes):
            state = env.reset()
            # Initialize accumulators for the losses
            episode_actor_loss = 0
            episode_critic_loss = 0
            true_episodic_reward = 0

            # reset noise parameters
            self.epsilon = 1.0
            self.ou_noise.reset_sigma(episode)
            self.episode_count = 0

            # normalizer to normalize rewards
            reward_normalizer = RewardNormalizer()

            self.batch_critic_losses = []
            self.batch_actor_losses = []

            # Interact with the environment
            done = False
            t = 0
            while not done:
                # select actions by exploring using ou policy
                noisy_action = self.select_action(
                    state, exploration=True, action_type=action_type
                )

                next_state, reward, done = env.step(noisy_action)
                true_episodic_reward += reward.item()

                # normalize the rewards
                reward_normalizer.update(reward.item())
                normalized_reward = reward_normalizer.normalize(reward.item())

                # construct transition to store into replay buffer
                experience = Experience(
                    state,
                    noisy_action,
                    normalized_reward,
                    next_state,
                )

                self.replay_memory.add(experience=experience)

                # decay sigma for next state
                t += 1
                dt = t / n_samples
                self.ou_noise.decay_sigma(dt, 0.0003)

                state = next_state

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

                self.batch_critic_losses.append(critic_loss)
                self.batch_actor_losses.append(actor_loss)

                if t % 50 == 0:
                    self.update_target_networks()

                if t % 2000 == 0:
                    print(
                        f"time-step: {t}, batch actor loss: {np.mean(self.batch_actor_losses)}, batch critic loss: {np.mean(self.batch_critic_losses)}"
                    )

                # Update priorities in the replay buffer (for prioritized experience replay)
                self.replay_memory.update_priorities(indices, td_error, 0.3)

                # Accumulate the losses over the iterations for logging
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss

            # Update the learning rate scheduler
            critic_scheduler.step()
            actor_scheduler.step()

            # After training during the episode, log the average losses
            actor_losses.append(np.mean(self.batch_actor_losses))
            critic_losses.append(np.mean(self.batch_critic_losses))
            rewards.append(true_episodic_reward)

            print(
                f"Episode {episode + 1} - Avg Actor Loss: {actor_losses[episode]:.4f}, Avg Critic Loss: {critic_losses[episode]:.4f}, True Episodic Reward: {true_episodic_reward:.4f}"
            )

        print("Training complete!")
        self.agent_perf = (actor_losses, critic_losses, rewards)

    def build_equity_curve(self, test_port: Portfolio):
        kraken_ds = KrakenDataSet(test_port, self.window_size, self.step_size)
        env = TradeEnv(kraken_ds, self.window_size)
        state = env.reset()
        previous_portfolio_value = test_port.get_initial_portfolio_value()
        agent_equity = [previous_portfolio_value]
        self.actions = [state[1]]
        done = False

        while not done:
            # select actions by exploring using ou policy
            action = self.select_action(state, exploration=False)
            self.actions.append(action)
            # reward is log return that incorporates penalty
            next_state, reward, done = env.step(action)
            current_portfolio_value = previous_portfolio_value * torch.exp(reward)
            agent_equity.append(current_portfolio_value.item())
            previous_portfolio_value = current_portfolio_value

            state = next_state
        return agent_equity

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
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
    OSBLSampler,
    PortfolioVectorMemory,
    PrioritizedReplayMemory,
)
from ddpgportfolio.portfolio.portfolio import Portfolio
from utilities.pg_utils import (
    OrnsteinUhlenbeckNoise,
    RewardNormalizer,
    compute_entropy,
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
    n_iter: int
    learning_rate: Optional[float] = 0.001
    betas: Optional[Tuple[float, float]] = (0.0, 0.9)
    device: Optional[str] = "mps"
    actor: Actor = field(init=False)
    actor_optimizer: torch.optim = field(init=False)
    loss_fn: nn.modules.loss.MSELoss = field(init=False)
    dataloader: DataLoader = field(init=False)
    pvm: PortfolioVectorMemory = field(init=False)
    osbl_sampler: OSBLSampler = field(init=False)
    ou_noise: OrnsteinUhlenbeckNoise = field(init=False)

    gamma: float = 0.9
    tau: float = 0.05
    epsilon: float = 1.0
    epsilon_max: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay_rate: float = 1e-5
    episode_count: int = 0
    warmup_steps: int = 6000
    entropy_beta: float = 0.20

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

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )

        self.actor.to(self.device)

        # loss function for the critic
        self.loss_fn = nn.MSELoss()

        # ou noise initialization
        self.ou_noise = OrnsteinUhlenbeckNoise(size=m_assets, theta=0.20, sigma=0.5)

        # initializing pvm with all cash initially
        self.pvm = PortfolioVectorMemory(self.portfolio.n_samples, m_noncash_assets)
        self.pvm.update_memory_stack(
            torch.zeros(m_noncash_assets), self.window_size - 2
        )
        self.osbl_sampler = OSBLSampler(20000, 50, 50)

    def select_uniform_action(self, m):
        uniform_vec = np.random.uniform(0, 1, size=m)
        return torch.tensor(uniform_vec, dtype=torch.float32)

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
            _description_
        exploration : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """

        action_logits = self.actor(state)

        if exploration:
            if action_type == "hybrid":
                if self.episode_count < self.warmup_steps:
                    noise = self.select_uniform_action(action_logits.shape)
                    action_logits += noise
                    self.episode_count += 1
                else:
                    # transition to ou noise after warm up steps
                    action_type = "ou"
            if action_type == "ou":
                noise = self.ou_noise.sample()
                action_logits += noise
                self.ou_noise.decay_sigma()

            elif action_type == "greedy" and np.random.rand() < self.epsilon:
                action = self.select_uniform_action(action_logits.shape)
                self.update_epsilon()
                return action[:, 1:]

        action = torch.softmax(action_logits, dim=1)

        # return all non-cash weights
        return action[:, 1:]

    def decay_entropy_beta(self, decay_rate=0.001):
        self.entropy_beta = self.entropy_beta * np.exp(-decay_rate * self.episode_count)

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

    def _normalize_batch_rewards(self, rewards):
        mean = rewards.mean()
        std = rewards.std()
        return (rewards - mean) / (std + 1e-5)

    def pre_train(self, total_steps: int = 2_000_000):
        """Pretraining the ddpg agent by populating the experience replay buffer"""
        print("pre-training ddpg agent started...")
        print("ReplayMemoryBuffer populating with experience...")
        kraken_ds = KrakenDataSet(self.portfolio, self.window_size, self.step_size)

        # resetting ou process in the event of a new run
        self.ou_noise.reset()

        n_samples = len(kraken_ds)

        curr_step = 0
        while curr_step < total_steps:
            for t in range(1, n_samples + 49):
                if curr_step >= total_steps:
                    break

                # Step 1: retrieve Xt and previous action
                xt, prev_index = kraken_ds[t - 1]
                previous_action = self.pvm.get_memory_stack(prev_index)

                # Step 2: construct the state
                state = (xt, previous_action)

                # step 3: compute deterministic action, remove cash
                action_logits = self.actor(state)
                action = torch.softmax(action_logits.view(-1), dim=-1)[1:]

                # step 4: get the relative price vector from price tensor to calculate reward
                yt = 1 / xt[0, :, -2]
                reward = self.portfolio.get_reward(action, yt, previous_action)

                # step 5: pre-train the actor
                # normalize reward by interval length
                J = reward.sum() / self.batch_size

                # compute policy gradient
                self.actor_optimizer.zero_grad()
                policy_loss = -J
                policy_loss.backward()
                self.actor_optimizer.step()

                xt_next, prev_index_next = kraken_ds[t]
                next_state = (xt_next, action.detach())
                experience = Experience(
                    state,
                    action.detach(),
                    reward.detach().item(),
                    next_state,
                    prev_index,
                )

                assert prev_index_next == prev_index + 1

                # step 6: Update pvm and osbl
                self.pvm.update_memory_stack(action.detach(), prev_index_next + 1)
                self.osbl_sampler.add(experience)

                # step 7: increment step
                curr_step += 1

        print("pretraining done")

        print(f"buffer size: {len(self.osbl_sampler)}")

        # we subtract one since each experience consists of current state and next state
        assert len(self.osbl_sampler) == n_samples + 48

    def train(self, n_episodes: int = 50):
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

        print("Training Started for DeepPG Agent")
        # scheduler to perform learning rate decay

        actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=100, gamma=0.9
        )

        actor_losses = []
        rewards = []

        # reset episode count
        self.episode_count = 0

        for episode in range(n_episodes):
            # Initialize accumulators for the losses
            episode_actor_loss = 0
            total_episodic_reward = 0
            t = episode
            # Sample a batch of experiences from the osbl sampler
            mini_batches = self.osbl_sampler.sample_mini_batches(t)
            for mini_batch in mini_batches:
                states, prev_index = zip(
                    *[(exp.state, exp.previous_index) for exp in mini_batch]
                )
                Xt = torch.stack([s[0] for s in states])
                previous_actions = torch.stack([s[1] for s in states])
                states = (Xt, previous_actions)
                prev_index = torch.tensor(prev_index)

                # train the actor
                actions = self.select_action(
                    states, exploration=True, action_type="hybrid"
                )
                # calculate reward based on action and state
                yt = 1 / Xt[:, 0, :, -2]
                reward = self.portfolio.get_reward(actions, yt, previous_actions)

                # normalize reward by interval length
                J = reward.sum() / self.batch_size

                # compute policy gradient
                self.actor_optimizer.zero_grad()
                policy_loss = -J
                policy_loss.backward()
                self.actor_optimizer.step()

                # update memory and accumulate metrics
                self.pvm.update_memory_stack(actions.detach(), prev_index + 1)
                episode_actor_loss += policy_loss.item()
                total_episodic_reward += reward.sum().item()

            # decay learning rate
            actor_scheduler.step()

            # Log episode metrics
            avg_episode_actor_loss = episode_actor_loss / len(self.dataloader)
            avg_reward_per_batch = total_episodic_reward / (len(self.dataloader))

            actor_losses.append(avg_episode_actor_loss)
            rewards.append(avg_reward_per_batch)

            print(
                f"Episode {episode + 1} - Actor Loss: {avg_episode_actor_loss:.4f}, Avg Reward: {avg_reward_per_batch:.4f}"
            )

        print("Training complete!")
        # performance plots
        plot_performance(actor_losses, rewards)

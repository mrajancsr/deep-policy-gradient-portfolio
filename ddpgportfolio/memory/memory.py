from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PortfolioVectorMemory:
    """Implements the Portfolio Vector Memory inspired by the idea of experience replay memory (Mnih et al., 2013),
    see pg. 13-14 of paper
    A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
    A PortfolioVectorMemory contains only non cash assets.
    """

    n_samples: int
    m_noncash_assets: int
    initial_weight: Optional[torch.tensor] = None
    memory: torch.tensor = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.memory = torch.ones(self.n_samples, self.m_noncash_assets) / (
            self.m_noncash_assets + 1
        )
        self.memory = self.memory.to(self.device)

    def update_memory_stack(self, new_weights: torch.tensor, indices: torch.tensor):
        self.memory[indices] = new_weights

    def get_memory_stack(self, indices):
        return self.memory[indices]


@dataclass
class ExperienceReplayMemory:
    """Implements the Experience Replay Memory by (Mnih et al. 2013)
    c.f https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    by storing the experience (Xt, w(t-1), wt, rt, X(t+1), done)"""

    buffer: deque = field(init=False)

    def __post_init__(self):
        self.buffer = deque(maxlen=1000000)

    def __repr__(self):
        return ""

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def add(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)

    def sample(self) -> Tuple:
        idx = np.random.choice(len(self.buffer), replace=False)
        return self.buffer[idx]


@dataclass
class Experience:
    state: Tuple[torch.tensor, torch.tensor]
    action: torch.tensor
    reward: torch.tensor
    next_state: Tuple[torch.tensor, torch.tensor]
    previous_index: torch.tensor

    def __repr__(self):
        return "Experience"


@dataclass
class PrioritizedReplayMemory:
    """Implements the Prioritized Experience Replay of Schaul, Quan et al (2016)
    https://arxiv.org/pdf/1511.05952
    by prioritizing experiences with better rewards
    """

    capacity: int
    alpha: Optional[float] = 0.6
    epsilon: Optional[float] = 1e-5
    beta_decay_rate: Optional[float] = 0.01
    alpha_decay_rate: Optional[float] = 0.001
    min_priority: float = 0.1
    device: Optional[str] = "mps"
    buffer: List[Experience] = field(init=False, default_factory=lambda: [])
    pos: int = field(init=False, default=0)
    priorities: List[float] = field(init=False, default_factory=lambda: [])

    def __repr__(self):
        return f"Total Experiences: {len(self.buffer)}"

    def __len__(self):
        return len(self.buffer)

    def peek_buffer(self):
        return self.buffer[-1]

    def peek_priorities(self):
        return self.priorities[-1]

    def add(self, experience: Experience, reward: float):
        """_summary_

        Parameters
        ----------
        experience : Experience
            _description_
        reward : torch.tensor
            _description_
        """
        priority = abs(reward) + self.epsilon

        if len(self.buffer) < self.capacity:
            # add experiencs and their priority
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # overwrite experience at position pos
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority

        self.priorities[self.pos] = max(priority, self.min_priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size, beta=0.4, p_recent=0.5
    ) -> Tuple[Tuple[Experience, torch.tensor, np.ndarray]]:

        # Decay alpha and beta over time
        self.alpha = max(
            0.1, self.alpha - self.alpha_decay_rate
        )  # Ensure alpha doesn't go below 0.1
        self.beta = min(1.0, beta + self.beta_decay_rate)  # Gradually increase beta

        # Partition the buffer into recent and older sections
        recent_cutoff = int(len(self.buffer) * 0.4)  # Top 20% of the buffer is recent
        recent_indices = list(range(len(self.buffer) - recent_cutoff, len(self.buffer)))
        older_indices = list(range(0, len(self.buffer) - recent_cutoff))

        # Number of samples from each partition
        n_recent = int(batch_size * p_recent)
        n_older = batch_size - n_recent

        # Normalize the priorities within each partition
        recent_priorities = np.array(self.priorities)[recent_indices] ** self.alpha
        older_priorities = np.array(self.priorities)[older_indices] ** self.alpha

        # Calculate probabilities for sampling
        prob_recent = recent_priorities / recent_priorities.sum()
        prob_older = older_priorities / older_priorities.sum()

        # Sample indices from each partition
        sampled_recent = np.random.choice(recent_indices, size=n_recent, p=prob_recent)
        sampled_older = np.random.choice(older_indices, size=n_older, p=prob_older)

        # Combine sampled indices and shuffle
        indices = np.concatenate([sampled_recent, sampled_older])
        np.random.shuffle(indices)

        # Compute importance-sampling weights for combined indices
        combined_priorities = np.array(self.priorities)[indices]
        prob_combined = (
            combined_priorities**self.alpha / (combined_priorities**self.alpha).sum()
        )
        weights = (len(self.buffer) * prob_combined) ** -self.beta
        weights /= weights.max()  # Normalize to avoid large weights

        # Extract the actual experiences from the buffer using the indices
        experiences = [self.buffer[idx] for idx in indices]

        xt, prev_action, action, reward, xt_next, prev_index = zip(
            *[
                (
                    *(exp.state[0], exp.state[1]),
                    exp.action,
                    exp.reward,
                    exp.next_state[0],
                    exp.previous_index,
                )
                for exp in experiences
            ]
        )

        xt = torch.stack(xt)
        prev_action = torch.stack(prev_action)
        action = torch.stack(action)
        reward = torch.tensor(reward, dtype=torch.float32)
        xt_next = torch.stack(xt_next)
        prev_index = torch.tensor(prev_index)

        experience = Experience(
            (xt, prev_action), action, reward, (xt_next, action), prev_index
        )

        weights = torch.tensor(
            weights, dtype=torch.float32
        )  # Convert weights to torch tensor

        return experience, indices, weights

    def update_priorities(self, indices, td_errors):
        # Ensure priorities are non-negative and account for epsilon to avoid zero priority
        td_error_priorities = td_errors.abs() + self.epsilon

        # Normalize recency bias
        recency_bias = 1 - (
            np.array(indices) / len(self.buffer)
        )  # Most recent = 1, oldest = 0

        # Weight TD errors and recency bias
        for idx, (priority, recency) in zip(
            indices, zip(td_error_priorities, recency_bias)
        ):
            combined_priority = self.alpha * priority + (1 - self.alpha) * recency
            self.priorities[idx] = max(combined_priority.item(), self.min_priority)


class OSBLSampler:
    def __init__(self, buffer_size=10000, batch_size=50, num_batches=10, beta=0.9):
        """
        Online Stochastic Batch Learning Trainer for managing data.

        Args:
            buffer_size (int): Maximum size of the training data buffer.
            batch_size (int): Size of each mini-batch.
            num_batches (int): Number of mini-batches to sample per training step.
            beta (float): Decay rate for prioritizing recent data (0 < beta < 1).
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.beta = beta
        self.buffer = deque(maxlen=buffer_size)  # Training data buffer

    def __len__(self):
        return len(self.buffer)

    def add(self, experience: Experience):
        """
        Add pre-training data to the buffer.

        Args:
            dataset (list or iterable): Pre-training dataset, each item being (state, action, reward).
        """
        self.buffer.append(experience)

    def geometric_sampling_probs(self, t, buffer_size, beta=5e-5, batch_size=50):
        """
        Compute geometric distribution probabilities for batch selection.

        Args:
            t (int): Current time step.
            buffer_size (int): Size of the buffer.
            beta (float): Success probability (similar to decay rate in geometric decay).
            batch_size (int): Mini-batch size.

        Returns:
            np.array: Sampling probabilities for each valid batch.
        """
        valid_starts = np.arange(buffer_size - batch_size)
        k = t - valid_starts - batch_size  # Temporal distance

        # Compute geometric probabilities
        log_probs = k * np.log(1 - beta) + np.log(beta)

        # Normalize in the log domain to prevent overflow
        log_probs -= np.max(
            log_probs
        )  # Subtract max log-probability for numerical stability
        probs = np.exp(log_probs)  # Convert back to probabilities
        probs /= probs.sum()  # Normalize to sum to 1

    def sample_mini_batches(self, t):
        """
        Sample mini-batches using geometric sampling.

        Args:
            t (int): Current time step.

        Returns:
            list: List of sampled mini-batches.
        """
        buffer_size = len(self.buffer)
        if buffer_size < self.batch_size:
            raise ValueError("Not enough data in the buffer to sample a batch.")

        # Compute sampling probabilities
        probs = self.geometric_sampling_probs(
            t, buffer_size, beta=self.beta, batch_size=self.batch_size
        )

        # Randomly sample mini-batch starting indices
        batch_starts = np.random.choice(
            np.arange(buffer_size - self.batch_size),
            size=self.num_batches,
            p=probs,
            replace=False,
        )

        # Extract mini-batches
        mini_batches = []
        for start in batch_starts:
            batch = list(self.buffer)[start : start + self.batch_size]
            mini_batches.append(batch)

        return mini_batches

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
        self.buffer = deque(maxlen=20000)

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
    capacity: int
    alpha: float = 1.0  # Controls prioritization (0: uniform, 1: fully prioritized)
    beta: float = 0.4  # Importance-sampling bias correction
    epsilon: float = 1e-5  # Avoid zero priority
    beta_decay_rate: float = 0.01  # Rate of beta increment
    alpha_decay_rate: float = 0.001  # Rate of alpha decrement
    min_priority: float = 0.1  # Minimum allowable priority
    device: str = "mps"  # Device for tensor operations
    buffer: List = field(default_factory=list)
    priorities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    position: int = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, experience: Tuple, initial_td_error: Optional[float] = None):
        """
        Adds an experience to the buffer, initializing its priority based on TD error or default.
        """
        # Compute initial priority based on TD error or use max priority
        # Compute initial priority
        if initial_td_error is not None:
            priority = abs(initial_td_error) + self.epsilon
        else:
            priority = max(
                self.priorities.max() if len(self.priorities) > 0 else 1.0, 1.0
            )

        # If buffer is not full, append experience and priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            if len(self.priorities) < self.capacity:
                self.priorities = np.append(self.priorities, priority)
        else:
            # Replace the oldest experience and update priority
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        # Ensure priority meets minimum threshold
        self.priorities[self.position] = max(priority, self.min_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: Optional[float] = None
    ) -> Tuple[List, np.ndarray, torch.Tensor]:
        """
        Samples a batch of experiences based on priorities, with importance-sampling weights.
        """
        beta = beta or self.beta
        self.beta = min(
            1.0, self.beta + self.beta_decay_rate
        )  # Increment beta over time

        # Normalize priorities to get probabilities
        scaled_priorities = self.priorities[: len(self.buffer)] ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

<<<<<<< HEAD
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
=======
        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.buffer), size=batch_size, p=sampling_probabilities
>>>>>>> ddpg_change
        )
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

        # Compute importance-sampling weights
        weights = (len(self.buffer) * sampling_probabilities[indices]) ** -beta
        weights /= weights.max()  # Normalize for numerical stability

        # Convert weights to tensor
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return experience, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: torch.tensor,
        recency_weight: Optional[float] = None,
    ):
        """
        Updates the priorities of sampled experiences based on TD errors and optionally recency bias.
        """
        assert not torch.any(torch.isnan(td_errors)), "NaN in TD errors!"
        assert not np.any(indices >= len(self.buffer)), "Indices out of range!"

        # Calculate new priorities
        td_error_priorities = (abs(td_errors) + self.epsilon).flatten()

        if recency_weight is not None:
            # Incorporate recency bias
            recency_bias = 1 - (indices / len(self.buffer))  # Recent: high, Old: low
            combined_priorities = (
                recency_weight * torch.tensor(recency_bias, dtype=torch.float32)
                + (1 - recency_weight) * td_error_priorities
            )
        else:
            combined_priorities = td_error_priorities

        # Update priorities in buffer, ensuring minimum threshold
        for idx, priority in zip(indices, combined_priorities):
            self.priorities[idx] = max(priority, self.min_priority)

    def decay_alpha(self):
        """
        Gradually decreases alpha for smoother transitions over time.
        """
        self.alpha = max(0.1, self.alpha - self.alpha_decay_rate)

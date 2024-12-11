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
        return f"Close Price: {self.state[0][0, :2, -2]}\
            \naction {self.state[1][:2]}\
            \nreward {self.reward:.4f}"


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
    device: Optional[str] = "mps"
    __buffer: List[Experience] = field(init=False, default_factory=lambda: [])
    pos: int = field(init=False, default=0)
    __priorities: List[float] = field(init=False, default_factory=lambda: [])

    def __repr__(self):
        return f"Total Experiences: {len(self.__buffer)}"

    def __len__(self):
        return len(self.__buffer)

    def peek(self):
        return self.__buffer[-1]

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

        if len(self.__buffer) < self.capacity:
            # add experiencs and their priority
            self.__buffer.append(experience)
            self.__priorities.append(priority)
        else:
            # overwrite experience at position pos
            self.__buffer[self.pos] = experience
            self.__priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size, beta=0.4
    ) -> Tuple[Tuple[Experience, torch.tensor, np.ndarray]]:
        """Sample a batch of experiences based on priority."""
        # Decay alpha and beta over time
        self.alpha = max(
            0.1, self.alpha - self.alpha_decay_rate
        )  # Ensure alpha doesn't go below 0.1
        self.beta = min(1.0, beta + self.beta_decay_rate)  # Gradually increase beta

        # Normalize the priorities and calculate the probabilities
        priorities = np.array(self.__priorities) ** self.alpha
        prob = priorities / priorities.sum()

        # Sample batch of experiences based on the probabilities
        indices = np.random.choice(len(self), batch_size, p=prob)

        # Compute importance-sampling weights (using beta)
        weights = (len(self.__buffer) / prob[indices]) ** self.beta
        weights /= weights.max()  # Normalize to avoid large weights

        # Extract the actual experiences from the buffer using the indices
        experiences = [self.__buffer[idx] for idx in indices]

        weights = torch.tensor(
            weights, dtype=torch.float32
        )  # Convert weights to torch tensor

        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.__priorities[idx] = (
                abs(td_error) + self.epsilon
            )  # Update priority based on TD-error

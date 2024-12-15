import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3, device="mps"):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.device = device
        self.reset()

    def reset(self):
        self.state = np.full(self.size, self.mu, dtype=np.float32)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float32, device=self.device)

    def decay_sigma(self, decay_rate=0.99, min_sigma=0.05):
        self.sigma = max(self.sigma * decay_rate, min_sigma)


class RewardNormalizer:
    def __init__(self):
        self.mean = 0.0  # Running mean
        self.M2 = 0.0  # Sum of squared differences
        self.count = 0  # Number of rewards seen

    def update(self, reward):
        """
        Update the running statistics with a new reward.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        """
        Get the current mean and standard deviation.
        """
        if self.count < 2:
            return self.mean, float("inf")  # Std is undefined for count < 2
        variance = self.M2 / self.count
        std = variance**0.5
        return self.mean, std

    def normalize(self, reward):
        """
        Normalize a new reward using the current mean and standard deviation.
        """
        mean, std = self.get_stats()
        if std == float("inf") or std == 0:  # Handle edge case for very few rewards
            return reward
        return (reward - mean) / std


def normalize_batch_rewards(rewards):
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + 1e-5)


def plot_performance(
    actor_losses: List[float], critic_losses: List[float], total_rewards: List[float]
):

    episodes = range(1, len(actor_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot actor and critic losses
    plt.plot(episodes, actor_losses, label="Actor Loss", color="blue")
    plt.plot(episodes, critic_losses, label="Critic Loss", color="orange")
    plt.title("Actor and Critic Losses Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.plot(episodes, total_rewards, label="Total Reward", color="green")
    plt.plot(
        episodes, np.cumsum(total_rewards), label="Cumulative Reward", color="blue"
    )

    plt.title("Cumulative Reward (Log Scale)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    # plt.yscale("log")
    plt.show()


def set_seed(seed: int):
    """
    Set the seed for all relevant random number generators to ensure reproducibility.

    Parameters:
    seed (int): The seed value to set.
    """
    random.seed(seed)  # Python's built-in random
    np.random.seed(seed)  # Numpy's random
    torch.manual_seed(seed)  # Torch's random

    # If using a GPU or MPS
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # For deterministic behavior, optional
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

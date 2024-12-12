import numpy as np
import torch


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float32)


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

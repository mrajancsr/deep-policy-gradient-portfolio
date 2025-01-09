from dataclasses import dataclass, field
from typing import Tuple

import gym
import numpy as np
import torch

from ddpgportfolio.dataset import KrakenDataSet
from ddpgportfolio.memory.memory import PortfolioVectorMemory


@dataclass
class TradeEnv(gym.Env):
    dataset: KrakenDataSet
    window_size: int
    transaction_cost: float = 0.0018
    num_assets: int = field(init=False)
    current_step: int = field(init=False, default=0)
    portfolio_vector_memory: PortfolioVectorMemory = field(init=False)
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.num_assets = self.dataset.portfolio.m_assets

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_assets),
            dtype=np.float32,
        )
        self.n_samples = len(self.dataset)

    def reset(self):
        self.current_step = 0
        init_weights = torch.zeros(self.num_assets)
        init_weights[0] = 1.0
        n_samples = len(self.dataset) + 1000
        self.portfolio_vector_memory = PortfolioVectorMemory(
            n_samples, self.num_assets, init_weights
        )
        return self._get_state()

    def _get_state(self) -> Tuple[torch.tensor, torch.tensor]:
        price_tensor = self.dataset[self.current_step]
        previous_weights = self.portfolio_vector_memory.get_memory_stack(
            self.current_step
        )
        return (price_tensor, previous_weights)

    def step(self, action: torch.tensor):
        self.portfolio_vector_memory.update_memory_stack(
            action,
            self.current_step + 1,
        )
        xt, previous_action = self._get_state()
        yt = 1 / xt[0, :, -2]
        commission_rate = self.transaction_cost
        reward = self.dataset.portfolio.get_reward(
            action, yt, previous_action, commission_rate=commission_rate
        )
        self.current_step += 1
        done = self.current_step + self.window_size >= len(self.dataset)
        next_state = self._get_state()
        return next_state, reward, done

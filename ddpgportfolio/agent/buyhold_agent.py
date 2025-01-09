from dataclasses import dataclass
from typing import Tuple

import torch

from ddpgportfolio.dataset import KrakenDataSet
from ddpgportfolio.environment.environment import TradeEnv
from ddpgportfolio.portfolio.portfolio import Portfolio


@dataclass
class BuyHoldAgent:
    portfolio: Portfolio
    batch_size: int
    window_size: int
    step_size: int
    cash_weight: float = 0.2
    device: str = "mps"

    def __post_init__(self):
        m_assets = self.portfolio.m_assets
        self.weights = torch.ones(m_assets)
        self.weights[0] = self.cash_weight
        self.weights[1:] = (1 - self.cash_weight) / (m_assets - 1)
        assert torch.isclose(self.weights.sum(), torch.tensor(1.0))

    def select_action(self):
        return self.weights

    def build_equity_curve(self):
        kraken_ds = KrakenDataSet(
            self.portfolio, self.window_size, self.step_size, device=self.device
        )
        env = TradeEnv(kraken_ds, self.window_size)
        state = env.reset()
        previous_portfolio_value = self.portfolio.get_initial_portfolio_value()
        agent_equity = []
        done = False

        while not done:
            xt, prev_action = state
            # select actions by exploring using ou policy
            action = self.select_action()
            next_state, _, done = env.step(action)

            # get the relative price and compute reward
            yt = 1 / xt[0, :, -2]
            yt = torch.concat([torch.ones(1), yt], dim=-1)
            reward = torch.dot(prev_action, yt)

            # pt = pt-1 * <wt-1, yt>
            current_portfolio_value = previous_portfolio_value * reward
            agent_equity.append(current_portfolio_value.item())
            previous_portfolio_value = current_portfolio_value

            state = next_state
        return agent_equity

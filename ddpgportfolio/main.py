# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


from typing import List

import torch

from ddpgportfolio.agent.ddpg_agent import DDPGAgent
from ddpgportfolio.dataset import KrakenDataSet
from ddpgportfolio.portfolio.portfolio import Portfolio

torch.set_default_device("mps")


def main():
    BATCH_SIZE = 50  # training is done in mini-batches
    WINDOW_SIZE = 50  # last n timesteps for the price tensor
    STEP_SIZE = 1  # for rolling window batch sampler
    start_date = "2024-01-01"  # start date of trading
    # DEVICE = "mps"

    asset_names: List[str] = [
        "CASH",
        "SOL",
        "ADA",
        "USDT",
        "AVAX",
        "LINK",
        "DOT",
        "PEPE",
        "ETH",
        "XRP",
        "TRX",
        "MATIC",
    ]

    portfolio = Portfolio(asset_names=asset_names, start_date=start_date)
    # kraken_ds = KrakenDataSet(portfolio, WINDOW_SIZE)
    agent = DDPGAgent(portfolio, BATCH_SIZE, WINDOW_SIZE, STEP_SIZE, 100)

    # need to pretrain the agent to populate the replay buffer with experiences
    agent.pre_train()
    # train the agent
    agent.train()


if __name__ == "__main__":
    # used for debugging purposes
    main()

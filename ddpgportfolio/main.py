# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


import random
from typing import List

import numpy as np
import torch

from ddpgportfolio.agent.ddpg_agent import DDPGAgent
from ddpgportfolio.portfolio.portfolio import Portfolio
from utilities.pg_utils import set_seed

set_seed(111)
torch.set_default_device("mps")


def main():
    BATCH_SIZE = 50  # training is done in mini-batches
    WINDOW_SIZE = 50  # last n timesteps for the price tensor
    STEP_SIZE = 1  # for rolling window batch sampler
    start_date = "2024-01-01"  # start date of trading
    end_date = "2024-09-30"
    N_EPISODES = 100  # number of episodes to train the agent
    N_ITERATIONS_PER_EPISODE = 20
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

    portfolio = Portfolio(
        asset_names=asset_names, start_date=start_date, end_date=end_date
    )
    # kraken_ds = KrakenDataSet(portfolio, WINDOW_SIZE)
    agent = DDPGAgent(portfolio, BATCH_SIZE, WINDOW_SIZE, STEP_SIZE)

    # train the agent
    agent.train(5)


if __name__ == "__main__":
    # used for debugging purposes
    main()

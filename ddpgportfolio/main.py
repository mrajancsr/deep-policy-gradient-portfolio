# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


from typing import List

import torch

from adaptivepm.ddpg_agent import DDPGAgent
from adaptivepm.portfolio import Portfolio

torch.set_default_device("mps")


def main():
    BATCH_SIZE = 50  # training is done in mini-batches
    WINDOW_SIZE = 50  # last n trading days for the price tensor
    STEP_SIZE = 1  # for rolling window batch sampler
    DEVICE = "mps"

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

    portfolio = Portfolio(asset_names=asset_names)

    agent = DDPGAgent(portfolio, BATCH_SIZE, WINDOW_SIZE, STEP_SIZE, 2000000)
    agent.train()


if __name__ == "__main__":
    # used for debugging purposes
    main()

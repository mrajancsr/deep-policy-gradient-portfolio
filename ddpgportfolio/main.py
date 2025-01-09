# The purpose of this file is to create a price tensor for input into the neural network
# and to train the policy using Deep Deterministic Policy Gradient.
# Code is inspired by the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
# For more details, see: c.f https://arxiv.org/abs/1706.10059


from typing import List

import torch

from ddpgportfolio.agent.buyhold_agent import BuyHoldAgent
from ddpgportfolio.agent.ddpg_agent import DDPGAgent
from ddpgportfolio.dataset import (
    KrakenDataSet,
)
from ddpgportfolio.portfolio.portfolio import Portfolio
from utilities.pg_utils import set_seed

# set_seed(111)
torch.set_default_device("mps")


def main():
    BATCH_SIZE = 100  # training is done in mini-batches
    WINDOW_SIZE = 10  # last n timesteps for the price tensor
    STEP_SIZE = 1  # for rolling window batch sampler
    train_start_date = "2024-01-01"  # start date of trading
    train_end_date = "2024-07-30"
    test_start_date = "2024-08-01"
    test_end_date = "2024-09-30"
    N_EPISODES = 5  # number of episodes to train the agent
    DEVICE = "mps"

    asset_names: List[str] = ["CASH", "SOL", "PEPE", "ETH", "TRX", "MATIC", "LINK"]

    train_port = Portfolio(
        asset_names=asset_names, start_date=train_start_date, end_date=train_end_date
    )
    test_port = Portfolio(
        asset_names=asset_names, start_date=test_start_date, end_date=test_end_date
    )

    ddpg_agent = DDPGAgent(
        train_port, BATCH_SIZE, WINDOW_SIZE, STEP_SIZE, device=DEVICE
    )
    bh_agent = BuyHoldAgent(
        test_port, BATCH_SIZE, WINDOW_SIZE, STEP_SIZE, cash_weight=0.2, device=DEVICE
    )

    bh_equity = bh_agent.build_equity_curve()

    # train the agent
    ddpg_agent.train(N_EPISODES, action_type="hybrid")

    ddpg_equity = ddpg_agent.build_equity_curve(test_port)

    print(ddpg_equity)


if __name__ == "__main__":
    # used for debugging purposes
    main()

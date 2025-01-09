from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import pandas as pd
import torch

from ddpgportfolio import Asset

PATH_TO_PRICES_PICKLE = os.path.join(
    os.getcwd(), "datasets", "Kraken_pipeline_output", "prices.pkl"
)


@dataclass
class Portfolio:
    """Implements a Portfolio that holds CryptoCurrencies as Assets
    Parameters
    _ _ _ _ _ _ _ _ _ _


    Attributes:
    _ _ _ _ _ _ __ __ _ _
    __assets: Dict[name, Asset]
        dictionary of Asset objects whose keys are the asset names
    """

    asset_names: List[str]
    start_date: str
    end_date: str
    initial_cash: float
    __prices: Dict[str, pd.DataFrame] = field(init=False, default_factory=lambda: {})
    __assets: Dict[str, Asset] = field(init=False)
    m_assets: int = field(init=False, default=0)
    m_noncash_assets: int = field(init=False, default=0)
    __annualization_factor: int = field(init=False, default=365 * 48)

    def __post_init__(self):
        self._load_pickle_object()
        self.__assets = {
            asset_name: Asset(
                name=asset_name,
                open_price=self.__prices["open"][asset_name].loc[
                    self.start_date : self.end_date,
                ],
                close_price=self.__prices["close"][asset_name].loc[
                    self.start_date : self.end_date
                ],
                high_price=self.__prices["high"][asset_name].loc[
                    self.start_date : self.end_date
                ],
                low_price=self.__prices["low"][asset_name].loc[
                    self.start_date : self.end_date
                ],
            )
            for asset_name in self.asset_names
        }
        self.m_assets = len(self.__assets)
        self.m_noncash_assets = self.m_assets - 1
        self.n_samples = (
            self.__prices["close"].loc[self.start_date : self.end_date,].shape[0]
        )

    def get_annualization_factor(self):
        return self.__annualization_factor

    def _load_pickle_object(self):
        with open(PATH_TO_PRICES_PICKLE, "rb") as f:
            self.__prices.update(pickle.load(f))

    def __iter__(self) -> Iterator[Asset]:
        yield from self.assets()

    def __repr__(self) -> str:
        return f"Portfolio size: {self.m_assets} \
            \nm_assets: {[asset.name for asset in self.assets()]}"

    def get_initial_portfolio_value(self):
        return self.initial_cash

    def get_asset(self, name: str) -> Asset:
        """Returns the asset in the portfolio given the name of the asset

        Parameters
        ----------
        asset : str
            name of the asset

        Returns
        -------
        Asset
            contains information about the asset
        """
        return self.__assets.get(name.upper())

    def assets(self) -> Iterator[Asset]:
        yield from self.__assets.values()

    def get_relative_price(self):
        return self.__prices["relative_price"].loc[self.start_date : self.end_date,]

    def get_close_price(self):
        return self.__prices["close"].loc[self.start_date : self.end_date,]

    def get_high_price(self):
        return self.__prices["high"].loc[self.start_date : self.end_date,]

    def get_low_price(self):
        return self.__prices["low"].loc[self.start_date : self.end_date,]

    def get_end_of_period_weights(self, yt: torch.tensor, wt_prev: torch.tensor):
        """Computes the wt' which is portfolio weight at the end of period t
        c.f formula 7 in https://arxiv.org/pdf/1706.10059

        Parameters
        ----------
        yt : torch.tensor
            relative price vector representing market movement from t-1 to period t
            given by close_t / close(t-1)
            shape=(batch_size, m_noncash_assets)
        wt_prev : torch.tensor
            portfolio weight at the beginning of previous period
            shape=(batch_size, m_noncash_assets)
        """
        wt_prime = (yt * wt_prev) / (yt.dot(wt_prev) + 1e-5)
        return wt_prime

    def get_transacton_remainder_factor(
        self,
        wt: torch.tensor,
        yt: torch.tensor,
        wt_prev: torch.tensor,
        comission_rate: float = 0.0018,
        n_iter: int = 3,
    ):
        """Computes the transaction remainder factor via a iterative approach
        c.f formula 14 and 15 in https://arxiv.org/pdf/1706.10059
        This formula reduces the portfolio value from pt' to pt

        Parameters
        ----------
         Parameters
        ----------
        wt : torch.tensor
            portfolio vector weight at beginning of period t+1
            dim=(batch_size, m_noncash_assets)
        yt : torch.tensor
            relative price vector given by close_t / close_t-1
            dim=(batch_size, m_noncash_assets)
        wt_prev : torch.tensor
            portfolio vector weight at beginning of period t
            dim=(batch_size, m_noncash_assets)
        comission_rate : float, default = 0.26% (maximum)
            comission rate for purchasing and selling
        n_iter: int, default = 3
        number of iterations to compute the transaction remainder factor
        """
        wt_prime = self.get_end_of_period_weights(yt, wt_prev)

        # get end of period cash position for each example in batch
        wt_cash_prime = wt_prime[0]

        # get cash position for portfolio weight at period t+1
        wt_cash = wt[0]

        # initial transaction remainder factor
        ut_k = comission_rate * torch.abs(wt[1:] - wt_prime[1:]).sum()
        c = comission_rate
        for _ in range(n_iter):
            update_term = torch.relu(wt_prime[1:] - ut_k * wt[1:]).sum()
            ut_k = (
                1
                / (1 - c * wt_cash)
                * (1 - c * wt_cash_prime - c * (2 - c) * update_term)
            )
        return ut_k

    def get_reward(
        self,
        wt: torch.tensor,
        yt: torch.tensor,
        wt_prev: torch.tensor,
        risk_free_rate: float = 0.0425,
        beta: float = 0.0018,
        alpha: float = 0.2,
        commission_rate: float = 0.0018,
    ):
        """returns the immediate reward to the agent given by 11 and mentioned on pg 11
        given by rt = ln(ut*yt . w(t-1)) / batch_size

        Parameters
        ----------
        wt : torch.tensor
            portfolio vector for beginning of period t+1
        yt : torch.tensor
            relative price vector given by Close_t / Close(t-1)
        wt_prev : torch.tensor
            portfolio vector weight for beginning of period t
        """
        rf_period = risk_free_rate / self.get_annualization_factor()
        # Risk penalty (volatility or large weight changes)

        yt_with_cash = torch.concat(
            [torch.tensor(1 + rf_period).unsqueeze(0), yt], dim=-1
        )

        weight_change_penalty = torch.sum(
            torch.abs(wt - wt_prev), dim=-1
        )  # penalize large changes in portfolio weights

        # Compute transaction cost penalty: this is based on the change in portfolio weights
        transaction_penalty = beta * weight_change_penalty
        ut = self.get_transacton_remainder_factor(
            wt, yt_with_cash, wt_prev, commission_rate
        )
        # portfolio return before transaction cost

        portfolio_return = yt_with_cash.dot(wt_prev)

        # Avoid log(0) or negative values by adding a small epsilon
        epsilon = 1e-6
        reward = torch.log(ut * portfolio_return + epsilon)

        # profitability incentive
        alignment_incentive = torch.sum(wt_prev * torch.relu(yt_with_cash - 1))

        return reward + alpha * alignment_incentive

    def update_portfolio_value(self, previous_portfolio_value, reward: torch.tensor):
        return previous_portfolio_value * torch.exp(reward)

    def calculate_final_equity_return(self, equity_curve: List[float]):
        """calculates total return from equity curve

        Parameters
        ----------
        equity_curve : List[float]
            _description_

        Returns
        -------
        _type_
            _description_
        """
        V_start = equity_curve[0]
        V_end = equity_curve[-1]
        total_return = ((V_end - V_start) / V_start) * 100
        return total_return

    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        peak = equity_curve[0]
        max_drawdown = -sys.maxsize
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown * 100

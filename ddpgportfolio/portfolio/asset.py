"""Implementation of Asset Class"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List

import numpy as np
import pandas as pd


@dataclass
class Asset:
    name: str
    open_price: pd.Series
    close_price: pd.Series
    high_price: pd.Series
    low_price: pd.Series
    price_relative_vector: np.ndarray = field(init=False)
    size: int = field(init=False)
    returns_history: pd.Series = field(init=False)
    annualized_returns: float = field(init=False)
    expected_returns: float = field(init=False)
    all_assets: ClassVar[List[Asset]] = []

    def __post_init__(self) -> None:
        self.size = self.open_price.shape[0]
        self.price_relative_vector = (
            self.close_price / self.close_price.shift(1)
        ).to_numpy()
        self.returns_history = np.log(self.price_relative_vector)
        self.annualized_returns = (
            self.returns_history.sum()
            * Asset.get_annualization_factor()
            / (self.size * 48)
        )
        self.expected_returns = self._get_expected_returns()
        self.__class__.all_assets.append(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Asset) -> bool:
        return type(self) is type(other) and self.name == other.name

    def __ne__(self, other: Asset) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        return f"Asset Name: {self.name}, \
            \nexpected returns: {self.expected_returns:.5f}, \
            \nannualized returns: {self.annualized_returns:.5f}"

    @staticmethod
    def get_annualization_factor() -> int:
        # 365 days of trading days in a year
        # multiplied by 48 intervals daily of 30 minute bars
        return 365 * 48

    def _get_expected_returns(self):
        return Asset.get_annualization_factor() * self.returns_history.mean()

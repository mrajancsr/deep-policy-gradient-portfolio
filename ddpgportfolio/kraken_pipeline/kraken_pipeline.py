# Aborting the kraken project for now as the datasets are not in 30 minute interval but rather 14 minute
# interval.
# will need to build the ohlc data from rest api instead
import os
import pickle
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List

import pandas as pd

DATASET_PATH = os.path.join(os.getcwd(), "datasets")
OUTPUT_PATH = os.path.join(
    os.getcwd(), "datasets", "Kraken_pipeline_output", "prices.pkl"
)


@dataclass
class KrakenPipeLine:
    """Preprocessing coins in the kraken exchange"""

    folder: str
    period: int
    coins: List[str]
    file_list: List[str] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        period = self.period
        coins = self.coins
        folder = self.folder
        self.file_list = [
            os.path.join(folder, coin_name) + f"USD_{period}.csv" for coin_name in coins
        ]
        assert len(self.file_list) != 0

    def preprocess_coins(self, prices: Dict[str, pd.DataFrame]):
        columns = ["timestamp", "open", "high", "low", "close", "volume", "count"]
        keep_cols = ["open", "high", "low", "close"]
        result_dfs = []

        for coin_name, file_name in zip(self.coins, self.file_list):
            df = pd.read_csv(file_name, names=columns)
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            df.pop("timestamp")
            df.set_index("date", inplace=True)
            df = df[keep_cols]
            tuples = [(coin_name, col_name) for col_name in keep_cols]
            df.columns = pd.MultiIndex.from_tuples(tuples)
            result_dfs.append(df)
        combined_dfs = reduce(
            lambda A, B: pd.merge(A, B, on="date", how="outer"), result_dfs
        )
        combined_dfs = combined_dfs.stack(level=0, future_stack=True).unstack(level=1)
        combined_dfs = combined_dfs.fillna(method="ffill")
        prices["open"] = combined_dfs["open"]
        prices["close"] = combined_dfs["close"]
        prices["low"] = combined_dfs["low"]
        prices["high"] = combined_dfs["high"]

        # create cash position
        prices["open"]["CASH"] = 1.0
        prices["open"] = prices["open"][["CASH"] + self.coins]
        prices["high"]["CASH"] = 1.0
        prices["high"] = prices["high"][["CASH"] + self.coins]
        prices["low"]["CASH"] = 1.0
        prices["low"] = prices["low"][["CASH"] + self.coins]
        prices["close"]["CASH"] = 1.0
        prices["close"] = prices["close"][["CASH"] + self.coins]
        prices["relative_price"] = prices["close"][["CASH"] + self.coins].div(
            prices["close"][["CASH"] + self.coins].shift(1)
        )


def main():
    coins: List[str] = [
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
    folder: str = os.path.join(DATASET_PATH, "Kraken_OHLC")
    print(folder)
    pipe = KrakenPipeLine(folder=folder, period=30, coins=coins)
    prices = {}
    pipe.preprocess_coins(prices)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(prices, f)


if __name__ == "__main__":
    main()

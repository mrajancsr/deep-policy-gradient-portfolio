from dataclasses import dataclass, field
import glob
from typing import List
import pandas as pd
import os

DATASET_PATH = os.path.join(os.getcwd(), "datasets")

print(DATASET_PATH)


@dataclass
class PoloniexPipeLine:
    """Preprocessing coins in the Poloniex exchange"""

    folder: str
    file_list: List[str] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        self.file_list = glob.glob(self.folder + "/*.csv")

    def combine_files_into_dataframes(self):
        df = pd.read_csv(self.file_list[0])
        df["date"] = pd.to_datetime(df["date"], unit="s")
        print(df.head())


def main():
    folder: str = os.path.join(DATASET_PATH, "Poloniex_OHLC")

    # process the files in the folder and change dimension to 4xaxb
    pipe = PoloniexPipeLine(folder=folder)
    pipe.combine_files_into_dataframes()


if __name__ == "__main__":
    main()

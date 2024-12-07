from typing import Generator, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ddpgportfolio.portfolio.portfolio import Portfolio


class KrakenDataSet(Dataset):
    """Creates a tensor Xt as defined by equation 18 in paper to feed into ANN

    Parameters
    ----------
    Dataset : torch.utils.data.dataset.Dataset
        ABC from torch utils class that represents a dataset
    """

    def __init__(
        self,
        portfolio: Portfolio,
        window_size: int = 50,
        step_size: int = 1,
        device="mps",
    ):
        self.portfolio = portfolio
        self.window_size = window_size
        self.step_size = step_size
        self.start_index = self.window_size - 1
        self.close_pr = torch.tensor(
            self.portfolio.get_close_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.high_pr = torch.tensor(
            self.portfolio.get_high_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.low_pr = torch.tensor(
            self.portfolio.get_low_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.device = device

    def __len__(self):
        return self.portfolio.n_samples - self.start_index - self.window_size

    def __getitem__(self, idx):
        m_noncash_assets = self.portfolio.m_noncash_assets
        start = idx * self.step_size
        end = start + self.window_size

        if end >= len(self.close_pr):
            raise IndexError(f"End index {end} exceeds data length.")

        # the price tensor
        xt = torch.zeros(3, m_noncash_assets, self.window_size).to(self.device)
        xt[0] = (self.close_pr[start:end,] / self.close_pr[end - 1,]).T
        xt[1] = (self.high_pr[start:end,] / self.close_pr[end - 1,]).T
        xt[2] = (self.low_pr[start:end,] / self.close_pr[end - 1,]).T

        return xt, end - 2


class SlidingWindowBatchSampler(Sampler):
    """
    A custom BatchSampler that samples batches of size `batch_size` from KrakenDataSet
    using a sliding window approach.
    """

    def __init__(
        self, dataset: KrakenDataSet, batch_size: int = 50, step_size: int = 1
    ):
        """
        Parameters
        ----------
        dataset : KrakenDataSet
            The KrakenDataSet Object
        batch_size : int, optional
            size of the batch, default=50
        step_size : int, optional
            the step size for sliding window, default=1
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.step_size = step_size

        # we are getting data from dataset, so first index starts at index 0
        self.start_index = 0
        # The end index is dataset size
        self.end_index = len(dataset)  # the number of observations in your dataset

    def __iter__(self) -> Iterator[int]:
        """
        Yield batches of indices to sample from the dataset.
        Each batch contains indices from [i, i+batch_size).
        """
        # Start iterating from the start index (49)
        for i in range(
            self.start_index, self.end_index - self.batch_size + 1, self.step_size
        ):
            # Each batch will be a list of indices [i, i+batch_size)
            yield list(range(i, i + self.batch_size))

    def __len__(self) -> int:
        """Returns the number of batches."""
        return (self.end_index - self.batch_size) // self.step_size + 1


def get_current_and_next_batch(
    kraken_dl: DataLoader,
) -> Generator[torch.tensor, torch.tensor, torch.tensor]:
    """Creates an iterator for dataloader

    Parameters
    ----------
    kraken_dl : DataLoader
        dataloader that contains the current batch of price tensor Xt,
        and the previous index given by t-1

    Yields
    ------
    Generator[torch.tensor, torch.tensor, torch.tensor]
        current batch of price tensor Xt
        next batch of price tensor X(t+1)
        previous index given by t-1
    """
    iterator = iter(kraken_dl)

    # get the first batch
    xt_batch, previous_index_batch = next(iterator)

    while True:
        try:
            xt_next_batch, current_index_batch = next(iterator)

            yield xt_batch, xt_next_batch, previous_index_batch
            xt_batch, previous_index_batch = xt_next_batch, current_index_batch
        except StopIteration:
            break

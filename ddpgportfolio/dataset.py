import torch
from torch.utils.data import Dataset

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
        self.asset_names = self.portfolio.asset_names
        self.close_pr = torch.tensor(
            self.portfolio.get_close_price()[self.asset_names].values[:, 1:],
            dtype=torch.float32,
        ).to(device)
        self.high_pr = torch.tensor(
            self.portfolio.get_high_price()[self.asset_names].values[:, 1:],
            dtype=torch.float32,
        ).to(device)
        self.low_pr = torch.tensor(
            self.portfolio.get_low_price()[self.asset_names].values[:, 1:],
            dtype=torch.float32,
        ).to(device)
        self.device = device

    def __len__(self):
        return self.portfolio.n_samples

    def __getitem__(self, idx):
        m_noncash_assets = self.portfolio.m_noncash_assets
        start = idx * self.step_size
        end = start + self.window_size
        current = end - 1

        if current >= len(self.close_pr):
            raise IndexError(f"End index {end} exceeds data length.")

        # the price tensor
        xt = torch.zeros(3, m_noncash_assets, self.window_size).to(self.device)
        xt[0] = (self.close_pr[start:end,] / self.close_pr[current,]).T
        xt[1] = (self.high_pr[start:end,] / self.close_pr[current,]).T
        xt[2] = (self.low_pr[start:end,] / self.close_pr[current,]).T

        return xt

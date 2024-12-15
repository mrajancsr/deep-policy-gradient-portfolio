import torch
from torch.utils.data import Dataset

from ddpgportfolio.portfolio.portfolio import Portfolio


class KrakenDataSet(Dataset):
    """Creates a tensor Xt as defined by equation 18 in paper to feed into ANN."""

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
        self.device = device

        # Get price data and move to the device
        self.close_pr = torch.tensor(
            self.portfolio.get_close_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.high_pr = torch.tensor(
            self.portfolio.get_high_price().values[:, 1:], dtype=torch.float32
        ).to(device)
        self.low_pr = torch.tensor(
            self.portfolio.get_low_price().values[:, 1:], dtype=torch.float32
        ).to(device)

        # Dynamically calculate the total number of valid samples
        self.total_samples = (len(self.close_pr) - self.step_size) // self.step_size

    def __len__(self):
        # Calculate the total number of samples considering the step size
        return (len(self.close_pr) - self.window_size) // self.step_size + 1

    def __getitem__(self, idx):
        """Retrieve a single sample."""
        # Calculate start and end indices
        start = idx * self.step_size
        end = start + self.window_size

        if end > len(self.close_pr):
            # Adjust the start and end to include the last valid window
            end = len(self.close_pr)
            start = end - self.window_size

        # Prepare the price tensor
        m_noncash_assets = self.portfolio.m_noncash_assets
        xt = torch.zeros(3, m_noncash_assets, self.window_size).to(self.device)
        xt[0] = (self.close_pr[start:end] / self.close_pr[end - 1]).T
        xt[1] = (self.high_pr[start:end] / self.close_pr[end - 1]).T
        xt[2] = (self.low_pr[start:end] / self.close_pr[end - 1]).T

        # The second output (`end - 2`) is adjusted for consistency
        return xt, end - 2

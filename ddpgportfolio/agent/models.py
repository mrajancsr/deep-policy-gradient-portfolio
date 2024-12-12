# The purpose of this file is to create a deep deterministic policy gradient
# which is an ANN composed of actor and critic
# Actor will estimate the policy
# Critic will estimate the Q value function

from typing import Tuple

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Actor(nn.Module):
    def __init__(self, input_channels=3, output_dim=11):
        """_summary_

        Parameters
        ----------
        input_channels : int, optional
            number of features, default=3
        output_dim : int, optional
            number of asset weights, default=11
        """
        super(Actor, self).__init__()
        self.output_dim = output_dim
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                input_channels, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(2, 20, kernel_size=(1, 48), stride=1, padding=(0, 0)),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv_layer_with_weights = nn.Sequential(
            nn.Conv2d(21, 1, kernel_size=(1, 1), stride=1)
        )
        self.cash_bias = nn.Parameter(torch.full((1, 1, 1), 0.3))
        self.softmax = nn.Softmax(dim=1)
        self.apply(weights_init)

    def forward(self, state: Tuple[torch.tensor, torch.tensor]) -> torch.tensor:
        """performs a forward pass and returns portfolio weights at time t
        c.f pg 14-15 from https://arxiv.org/pdf/1706.10059

        Parameters
        ----------
        price_tensor : torch.tensor
            price tensor Xt comprised of close, high and low prices
            dim = (batch_size, kfeatures, massets, window_size)
            window_size pretains to last 50 trading prices
        pvm : torch.tensor
            weights w(t-1) from portfolio vector memory at time t-1
            dim = (batch_size, massets)

        Returns
        -------
        torch.tensor, dim = (batch_size, m_noncash_assets)
            action or current weights at time t
        """
        price_tensor, prev_weights = state
        x = self.conv_layer(price_tensor)

        # add previous weights to next conv layer
        if price_tensor.dim() == 3:
            # only one example
            dim = 0
            prev_weights = prev_weights.unsqueeze(0).unsqueeze(2)
            x = torch.cat([x, prev_weights], dim=dim)
            cash_bias = self.cash_bias.expand(1, 1, 1)
            dim += 1
        else:
            # we have a batch of examples now
            dim = 1
            prev_weights = prev_weights.unsqueeze(1).unsqueeze(3)
            x = torch.cat([x, prev_weights], dim=dim)
            batch_size = x.shape[0]
            cash_bias = self.cash_bias.expand(batch_size, 1, 1, 1)
            dim += 1

        x = self.conv_layer_with_weights(x)

        # add a cash bias to the layer before voting
        logits = torch.cat([cash_bias, x], dim=dim)
        # current_weights = self.softmax(x).view(-1)

        return logits


class Critic(nn.Module):
    def __init__(self, input_channels: int = 3, m_assets: int = 11):
        """Critic network for DDPG.  Given a state (Xt, w(t-1)), this network outputs the Q-Value

        Parameters
        ----------
        input_channels : int
            _description_
        output_dim : int
            _description_
        """
        super(Critic, self).__init__()
        self.m_assets = m_assets
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                input_channels, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(2, 20, kernel_size=(1, 48), stride=1, padding=(0, 0)),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Flatten(),
        )

        # fully connected layer for actions
        self.fc_layer = nn.Sequential(nn.Linear(24, 64), nn.ReLU(True))

        # q layer to evaluate the state and action
        self.q_layer = nn.Sequential(
            nn.Linear(20 * m_assets + 64, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

        self.apply(weights_init)

    def forward(
        self, state: Tuple[torch.tensor, torch.tensor], current_weights: torch.tensor
    ) -> torch.tensor:
        """_summary_

        Parameters
        ----------
        state : Tuple[torch.tensor, torch.tensor]
            composed of price tensor Xt and previous action wt_prev
            price_tensor : torch.tensor
                price tensor Xt comprised of close, high and low prices
                dim = (batch_size, kfeatures, m_assets, window_size)
                window_size pretains to last 50 trading prices
            wt_prev : torch.tensor
        current_weights : torch.tensor
            the current action wt

        Returns
        -------
        torch.tensor
            _description_
        """
        xt, prev_weights = state
        price_features = self.conv_layer(xt)

        # process actions
        actions = torch.cat([current_weights, prev_weights], dim=-1)
        action_features = self.fc_layer(actions)

        # combine features
        combined = torch.cat([price_features.view(-1), action_features], dim=-1)

        # estimate the q value
        q_value = self.q_layer(combined)
        return q_value

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class gaussianMLP(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, n_hidden=4):
        """
        DON'T CONFUSE THE VARIABLE NAMES USED HERE WITH THE ONES
        FROM THE PAPER/MODEL

        x_dim: dimension of observations (the data)
        z_dim: dimension of latent variables (the target multivariate gaussian)
        h_dim: dimension of hidden layer

        returns mu, log(sigma**2)
        """
        super(gaussianMLP, self).__init__()

        self.in_layer = nn.Linear(x_dim, h_dim)
        self.hidden = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(n_hidden)])
        self.mu_weights = nn.Linear(h_dim, z_dim)
        self.cov_diag_weights = nn.Linear(h_dim, z_dim)

    def get_pretrain_params(self):
        return (
          list(self.in_layer.parameters())+
          list(self.hidden.parameters())+
          list(self.mu_weights.parameters())
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = F.leaky_relu(x)
        for hid in self.hidden:
            x = hid(x)
            x = F.leaky_relu(x)

        mu = self.mu_weights(x)
        cov_diag = self.cov_diag_weights(x)

        return mu, cov_diag

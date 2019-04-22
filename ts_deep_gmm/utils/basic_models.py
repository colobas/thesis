import torch
import torch.nn as nn
import torch.nn.functional as F

class gaussianMLP(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, n_hidden=4, reg=0.01):
        """
        DON'T CONFUSE THE VARIABLE NAMES USED HERE WITH THE ONES
        FROM THE PAPER/MODEL

        x_dim: dimension of observations (the data)
        z_dim: dimension of latent variables (the target multivariate gaussian)
        h_dim: dimension of hidden layer
        """
        super(gaussianMLP, self).__init__()

        self.in_layer = nn.Linear(x_dim, h_dim)
        self.hidden = [nn.Linear(h_dim, h_dim) for _ in range(n_hidden)]
        self.mu_weights = nn.Linear(h_dim, z_dim)
        self.cov_diag_weights = nn.Linear(h_dim, z_dim)
        self.reg = reg

    def forward(self, x):
        x = self.in_layer(x)
        x = F.relu(x)
        for hid in self.hidden:
            x = hid(x)
            x = F.relu(x)

        mu = self.mu_weights(x)
        x = F.relu(x)
        cov_diag = F.relu(self.cov_diag_weights(x)).clamp(max=10) + self.reg # diag has to be positive

        return mu, cov_diag

class categMLP(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, n_hidden=4):
        """
        DON'T CONFUSE THE VARIABLE NAMES USED HERE WITH THE ONES
        FROM THE PAPER/MODEL

        x_dim: dimension of observations (the data)
        z_dim: dimension of latent variables (the target categ)
        h_dim: dimension of hidden layer
        """
        super(categMLP, self).__init__()

        self.in_layer = nn.Linear(x_dim, h_dim)
        self.hidden = [nn.Linear(h_dim, h_dim) for _ in range(n_hidden)]
        self.out_layer = nn.Linear(h_dim, z_dim)


    def forward(self, x):
        x = self.in_layer(x)
        x = F.relu(x)
        for hid in self.hidden:
            x = hid(x)
            x = F.relu(x)
        x = self.out_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x


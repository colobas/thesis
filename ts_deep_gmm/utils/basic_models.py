import torch
import torch.nn as nn
import torch.nn.functional as F

class gaussianMLP(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, reg=0.01):
        """
        DON'T CONFUSE THE VARIABLE NAMES USED HERE WITH THE ONES
        FROM THE PAPER/MODEL

        x_dim: dimension of observations (the data)
        z_dim: dimension of latent variables (the target multivariate gaussian)
        h_dim: dimension of hidden layer
        """
        super(gaussianMLP, self).__init__()

        self.hidden = nn.Linear(x_dim, z_dim)
        self.mu_weights = nn.Linear(h_dim, z_dim)
        self.cov_diag_weights = nn.Linear(h_dim, z_dim)
        self.reg = reg

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        mu = self.mu_weights(x)
        cov_diag = F.relu(self.cov_diag_weights(x)).clamp(max=10) + self.reg # diag has to be positive

        return mu, cov_diag

class categMLP(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        """
        DON'T CONFUSE THE VARIABLE NAMES USED HERE WITH THE ONES
        FROM THE PAPER/MODEL

        x_dim: dimension of observations (the data)
        z_dim: dimension of latent variables (the target categ)
        h_dim: dimension of hidden layer
        """
        super(categMLP, self).__init__()

        self.hidden1 = nn.Linear(x_dim, h_dim)
        self.hidden2 = nn.Linear(h_dim, z_dim)


    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.log_softmax(x, dim=-1)
        return x


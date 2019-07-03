import math
import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn

from torch.distributions.lowrank_multivariate_normal import (
    _batch_capacitance_tril,
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis
    )

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MultivariateGaussian(nn.Module):
    def __init__(self, dim, loc=None, sqrt_cov_diag=None, sqrt_cov_factor=None):
        super().__init__()

        if loc is None:
            loc = torch.zeros(dim)

        if sqrt_cov_diag is None:
            sqrt_cov_diag = torch.ones(dim)

        if sqrt_cov_factor is None:
            sqrt_cov_factor = torch.ones(dim).unsqueeze(-1)

        self.loc = nn.Parameter(loc)
        self.sqrt_cov_diag = nn.Parameter(sqrt_cov_diag)
        self.sqrt_cov_factor = nn.Parameter(sqrt_cov_factor)

        self._event_shape = self.loc.shape[-1:]

    def log_prob(self, value):
        diff = value - self.loc

        try:
            capacitance_tril = _batch_capacitance_tril(self.sqrt_cov_factor ** 2 + 0.1,
                                                       self.sqrt_cov_diag ** 2 + 0.1)
        except:
            print(self.sqrt_cov_factor)
            print(self.sqrt_cov_diag)
            assert False

        M = _batch_lowrank_mahalanobis(self.sqrt_cov_factor ** 2 + 0.1,
                                       self.sqrt_cov_diag ** 2 + 0.1,
                                       diff,
                                       capacitance_tril)

        log_det = _batch_lowrank_logdet(self.sqrt_cov_factor ** 2 + 0.1,
                                        self.sqrt_cov_diag ** 2 + 0.1,
                                        capacitance_tril)

        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + log_det + M)


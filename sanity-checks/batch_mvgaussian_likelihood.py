# -*- coding: utf-8 -*-
# %%
from IPython.display import HTML
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

# %%
import math

import torch
from torch.distributions import MultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis

# %%

def gen_mask(n_samples, dim):
    """
    returns n_samples of dimension dim, where each entry is randomly 1 or -1
    """
    return (
      torch.ones(n_samples, dim)
      - 2*torch.randint(0, 2, (n_samples, dim)).type(torch.FloatTensor)
    )

def gen_covs(N, dim):
    a = torch.randn(N, dim, dim)
    return a @ a.transpose(2, 1)


def log_prob(loc, scale_tril, value, dim):
    diff = value - loc
    M = _batch_mahalanobis(scale_tril, diff)
    half_log_det = scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    return -0.5 * (dim * math.log(2 * math.pi) + M) - half_log_det


n_samples = 100
dim = 2

mus = gen_mask(n_samples, dim) * (5 * (torch.rand(n_samples, dim) + 1))
covs = gen_covs(n_samples, dim)

data = torch.randn(n_samples, dim)

logprobs1 = MultivariateNormal(mus, covs).log_prob(data)

logprobs2 = torch.zeros(n_samples)

for i in range(n_samples):
    logprobs2[i] = MultivariateNormal(mus[i], covs[i]).log_prob(data[i])

assert torch.all(logprobs1 == logprobs2)

logprobs3 = log_prob(mus, torch.cholesky(covs), data, dim)

assert torch.all(logprobs1 == logprobs3)

n_components = 3

mus = gen_mask(n_components, dim) * (5 * (torch.rand(n_components, dim) + 1))
covs = gen_covs(n_components, dim)

logprobs1 = MultivariateNormal(mus, covs).log_prob(data.unsqueeze(1)).sum(dim=1)

logprobs2 = torch.zeros(n_samples)

for i in range(n_samples):
    for k in range(n_components):
        logprobs2[i] += MultivariateNormal(mus[k], covs[k]).log_prob(data[i])

assert torch.all(logprobs1 == logprobs2)

logprobs3 = log_prob(mus, torch.cholesky(covs), data.unsqueeze(1), dim).sum(dim=1)

assert torch.all(logprobs1 == logprobs3)


# -*- coding: utf-8 -*-
# %%
from IPython.display import HTML
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

# %%
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from pyro.distributions import RelaxedOneHotCategoricalStraightThrough
from sklearn import datasets

from tensorboardX import SummaryWriter
from tqdm import trange

use_cuda = torch.cuda.is_available()
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
class GMM_Generator(nn.Module):
    def __init__(self, n_clusters, ydim):
        super().__init__()

        self.ydim = ydim

        self.logits = nn.Parameter(torch.ones(n_clusters))
        self.mus = nn.Parameter(torch.stack([
            torch.randint(5, 7, (1,)).type(torch.FloatTensor) * 
            torch.randn(ydim) for i in range(n_clusters)
        ]))

        self.sigmas = nn.Parameter(
            torch.stack([torch.randn(ydim) for _ in range(n_clusters)])
        )

    def sample(self, n_samples, temperature=1, ret_z=False):
        oh = RelaxedOneHotCategoricalStraightThrough(
            logits=self.logits,
            temperature=temperature
        ).sample((n_samples, ))
        mus = (oh.unsqueeze(2) * self.mus).sum(dim=1)
        sigmas = (oh.unsqueeze(2) * self.sigmas).sum(dim=1) ** 2

        if ret_z:
            return Normal(mus, sigmas).sample((1, )).squeeze(0), oh.argmax(dim=1)
        else:
            return Normal(mus, sigmas).sample((1, )).squeeze(0)


    def rsample(self, n_samples, ret_z, temperature=0.1):
        oh = RelaxedOneHotCategoricalStraightThrough(
            logits=self.logits,
            temperature=temperature
        ).rsample((n_samples, ))
        mus = (oh.unsqueeze(2) * self.mus).sum(dim=1)
        sigmas = (oh.unsqueeze(2) * self.sigmas).sum(dim=1) ** 2
        
        if ret_z:
            return Normal(mus, sigmas).rsample((1, )).squeeze(0), oh.argmax(dim=1)
        else:
            return Normal(mus, sigmas).rsample((1, )).squeeze(0)

class Discriminator(nn.Module):
    def __init__(self, ydim, hdim, n_hidden=1):
        super().__init__()

        net_modules = (
          [nn.Linear(ydim, hdim), nn.LeakyReLU()] +
          sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, 1), nn.Sigmoid()]
        )

        self.net = nn.Sequential(*net_modules)

    def forward(self, y):
        return self.net(y)
# %%
true_model = GMM_Generator(n_clusters=3, ydim=2)
true_model.logits = nn.Parameter(torch.Tensor([1, 2, 3]))

# %%
Y, z = true_model.rsample(n_samples=1000, ret_z=True)
Y = Y.detach()

# %%
plt.figure(figsize=(10, 10))
plt.scatter(Y[:, 0], Y[:, 1], c=z, s=5)
plt.show()

# %%
gmmG = GMM_Generator(ydim=2, n_clusters=3).cuda()
netD = Discriminator(ydim=2, hdim=1, n_hidden=1).cuda()

# %%
count_parameters(gmmG), count_parameters(netD)

# %%
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

#optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.99))
optD = optim.SGD(netD.parameters(), lr=0.0001)
optG = optim.Adam(gmmG.parameters(), lr=0.0002, betas=(0.5, 0.99))

# %%
Y = Y.cuda()

# %%
n_epochs = 5000
bs = 50

true_batch_labels = real_label * torch.ones(bs).cuda()
fake_batch_labels = fake_label * torch.ones(bs).cuda()

writer = SummaryWriter(f"/workspace/runs/{now_str()}")

for epoch in trange(n_epochs):
    batches = range((len(Y) - 1) // bs + 1)
    for i in batches:
        start_i = i * bs
        end_i = start_i + bs
        yb = Y[start_i:end_i]

        true_batch_labels = real_label * torch.ones(len(yb)).cuda().unsqueeze(1)
        fake_batch_labels = fake_label * torch.ones(len(yb)).cuda().unsqueeze(1)

        # optimize discriminator
        netD.zero_grad()

        true_discr = netD(yb)
        discr_true_avg = true_discr.detach().mean()
        lossD_real = criterion(true_discr, true_batch_labels)
        lossD_real.backward()

        fake = gmmG.sample(len(yb), temperature=0.2)

        fake_discr = netD(fake.detach())
        discr_fake_avg = fake_discr.detach().mean()
        lossD_fake = criterion(fake_discr, fake_batch_labels)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optD.step()

        # optimize generator
        gmmG.zero_grad()

        # for gen loss, fake is true
        lossG = criterion(netD(fake), true_batch_labels)
        lossG.backward()
        optG.step()

        writer.add_scalar("gans/discriminator", lossD, epoch*bs + i)
        writer.add_scalar("gans/generator", lossG, epoch*bs + i)
        writer.add_scalar("gans/discr_true_avg", discr_true_avg, epoch*bs + i)

# %%
with torch.no_grad():
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20/0.8, 10), sharey="row", sharex="row")
    p = ax1.scatter(Y[:,0].cpu(), Y[:,1].cpu(), c=netD(Y).cpu().numpy().squeeze(), s=5, norm=mcolors.Normalize(0, 1))

    fake = gmmG.rsample(n_samples=3000, temperature=0.01, ret_z=False)
    p = ax2.scatter(fake[:, 0].cpu(), fake[:, 1].cpu(), s=5, c=netD(fake).cpu().numpy().squeeze(), norm=mcolors.Normalize(0, 1))
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.02, 0.7])

    f.colorbar(p, cax=cbar_ax)

    plt.show()

# %%
with torch.no_grad():
    y1 = np.linspace(-10, 10, 200)
    y2 = np.linspace(-10, 10, 200)

    y1, y2 = np.meshgrid(y1, y2)

    Ygrid = np.dstack([y1, y2]).reshape(-1, 2)
    plt.figure(figsize=(13, 10))
    p = plt.scatter(Ygrid[:, 0], Ygrid[:, 1], c=netD(torch.Tensor(Ygrid).cuda()).cpu().numpy().squeeze(), s=5, norm=mcolors.Normalize(0, 1))
    plt.colorbar(p)
    plt.show()

# %%
with torch.no_grad():
    fake, labels = gmmG.rsample(n_samples=3000, ret_z=True, temperature=0.01)

    plt.figure(figsize=(10, 10))
    plt.scatter(fake[:, 0].cpu(), fake[:, 1].cpu(), s=5, c=labels.cpu().numpy())
    plt.show()

# %%

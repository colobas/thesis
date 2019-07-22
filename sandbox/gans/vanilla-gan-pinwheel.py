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

from tensorboardX import SummaryWriter
from tqdm import trange

use_cuda = torch.cuda.is_available()
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")


# %%
def one_hot_sample(n_cats, n_samples, device):
    return (
        torch.zeros(n_samples, n_cats, device=device)
             .scatter_(1, torch.randint(0, n_cats, (n_samples, 1), device=device), 1)
    )


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate, seed=1):
    # code from Johnson et. al. (2016)
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    np.random.seed(seed)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))

    data[:, 0:2] = data[:, 0:2]/20

    return data[:, 0:2], data[:, 2].astype(np.int)


# %%
data, labels = make_pinwheel_data(0.3, 0.05, 3, 1000, 0.25)

if ipython is not None:
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:,0], data[:,1], c=labels, s=5)
    plt.show()


# %%
class Generator(nn.Module):
    def __init__(self, ydim, xdim, hdim, n_hidden=1):
        super().__init__()

        net_modules = (
          [nn.Linear(xdim, hdim), nn.LeakyReLU()] + 
          sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, ydim), nn.Tanh()]
        )

        self.net = nn.Sequential(*net_modules)

    def forward(self, y):
        return self.net(y)

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
class CatGenerator(nn.Module):
    def __init__(self, ydim, xdim, hdim, n_hidden=1, n_cats=3):
        super().__init__()
        
        self.xdim = xdim
               
        nets = []
        for i in range(n_cats):
            net_modules = (
              [nn.Linear(xdim, hdim), nn.LeakyReLU()] + 
              sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], []) +
              [nn.Linear(hdim, ydim), nn.Tanh()]
            )
            nets.append(
                nn.Sequential(*net_modules)
            )
        
        self.nets = nn.ModuleList(nets)

    def forward(self, x):       
        outputs = torch.cat([net(x[:,0:self.xdim]).unsqueeze(1) for net in self.nets], dim=1)
        
        return (x[:, self.xdim:].unsqueeze(-1) * outputs).sum(dim=1)

class CatDiscriminator(nn.Module):
    def __init__(self, ydim, hdim, n_hidden=1, n_cats=3):
        super().__init__()

        net_modules1 = (
          [nn.Linear(ydim, hdim), nn.LeakyReLU()] + 
          sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], [])
        )
        
        net_modules2 = (
          [nn.Linear(ydim, hdim), nn.LeakyReLU()] + 
          sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, n_cats), nn.Softmax(dim=1)]
        )
        
        self.net1 = nn.Sequential(*net_modules1)
        self.net2 = nn.Sequential(*net_modules2)
        
        self.final = nn.Linear(hdim + n_cats, 1)

    def forward(self, y):        
        z = torch.cat([self.net1(y), self.net2(y)], dim=1)
        z = self.final(z)
        return F.sigmoid(z)


# %%
Y = torch.Tensor(data).cuda()

# %%
#netG = Generator(ydim=2, xdim=3, hdim=25, n_hidden=3).cuda()
netD = Discriminator(ydim=2, hdim=100, n_hidden=3).cuda()

netG = CatGenerator(ydim=2, xdim=2, hdim=60, n_hidden=3, n_cats=3).cuda()
#netD = CatDiscriminator(ydim=2, hdim=25, n_hidden=3, n_cats=3).cuda()

# %%
count_parameters(netG), count_parameters(netD)

# %%
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.99))
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.99))

# %%
n_epochs = 5000
bs = 128

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
        
        latent = torch.cat([
            torch.randn(len(yb), 2, device="cuda:0"),
            one_hot_sample(3, len(yb), "cuda:0")],
            dim=1
        )
        fake = netG(latent)
        
        fake_discr = netD(fake.detach())
        discr_fake_avg = fake_discr.detach().mean()
        lossD_fake = criterion(fake_discr, fake_batch_labels)
        lossD_fake.backward()
        
        lossD = lossD_real + lossD_fake
        optD.step()
        
        # optimize generator
        netG.zero_grad()
        
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

    latent = torch.cat([
        torch.randn(3000, 2, device="cuda:0"),
        one_hot_sample(3, 3000, "cuda:0")],
        dim=1
    )
    
    fake = netG(latent)
    p = ax2.scatter(fake[:, 0].cpu(), fake[:, 1].cpu(), s=5, c=netD(fake).cpu().numpy().squeeze(), norm=mcolors.Normalize(0, 1))
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.02, 0.7])
    
    f.colorbar(p, cax=cbar_ax)
    
    plt.show()

# %%
with torch.no_grad():
    y1 = np.linspace(-1, 1, 200)
    y2 = np.linspace(-1, 1, 200)

    y1, y2 = np.meshgrid(y1, y2)

    Ygrid = np.dstack([y1, y2]).reshape(-1, 2)
    plt.figure(figsize=(13, 10))
    p = plt.scatter(Ygrid[:, 0], Ygrid[:, 1], c=netD(torch.Tensor(Ygrid).cuda()).cpu().numpy().squeeze(), s=5, norm=mcolors.Normalize(0, 1))
    plt.colorbar(p)
    plt.show()

# %%
with torch.no_grad():
    latent = torch.cat([
        torch.randn(3000, 2, device="cuda:0"),
        one_hot_sample(3, 3000, "cuda:0")],
        dim=1
    )
    
    fake = netG(latent)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(fake[:, 0].cpu(), fake[:, 1].cpu(), s=5, c=latent[:, 2:].argmax(dim=1).cpu().numpy())
    plt.show()

# %%

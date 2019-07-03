# -*- coding: utf-8 -*-
# %% [markdown]
# # Unfair GAN
#
# Say we have a dataset whose distribution we know has multiple modes. For example:
#
# `INSERT PINWHEEL IMAGE`
#
# One way to try to model this dataset would be with a GMM. However, a Gaussian is clearly not expressive enough to accurately model each of the components, so that isn't a very attractive approach.
#
# On the other hand, we know there are very expressive generative models, like GANs, that are way more expressive than a simple Gaussian. These however are harder to reason about and interpret. For instance, if we fit a "vanilla" GAN on this dataset, how would it deal with the multiple modes? How could we endow it with the notion of different components?
#
# I decided to play around with this, and came up with this idea:
#
# - What if we have a GAN with more than one Generator, each one "specializing" in one of the components?
#
# It sounded worth trying. So that's what I did:

# %% [markdown]
# ### Imports
# [Move on to the next section](#Definitions)

# %%
from IPython.display import HTML
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

# %%
import itertools
import pickle

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

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)


# %% [markdown]
# ### Definitions
# [Move on to next section](#Generator-and-Discriminator)

# %%
def one_hot_sample(n_cats, n_samples, device):
    """
    helper function that samples a categorical variable via `torch.randint`
    and converts it into a one-hot vector
    """
    return (
        torch.zeros(n_samples, n_cats, device=device)
             .scatter_(1, torch.randint(0, n_cats, (n_samples, 1), device=device), 1)
    )


# %%
def count_parameters(model):
    """
    helper to count parameters in a pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate, seed=1):
    """
    code adapted from https://github.com/mattjj/svae
    
    generates a pinwheel dataset, with given number of components
    """
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


# %% [markdown]
# #### Generator and Discriminator

# %% [markdown]
# Lets define the Generator and Discriminator.
#
# Since I'll be working on a simple problem, I'll use simple building blocks.
#
# The `Discriminator` is a simple MLP, with `LeakyReLU` activations. We can pick the width and depth of its layers.
#
# The `Generator` is comprised of K "sub-generators". (I'm calling this "Unfair GAN" because it's K against 1. Lame, I know). Each "sub-generator" is a simple NN, with `LeakyReLU` activations, and it gets a vector in some input-space and outputs a vector in the desired observation space.
#
# To generate one sample from the full `Generator`, we sample two vectors. The first is a one-hot vector, which will select the "sub-generator" we'll be sampling from. The second is an isotropic Gaussian vector, which will be fed to the selected "sub-generator".

# %%
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

class Generator(nn.Module):
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


# %% [markdown]
# ### Experiment

# %% [markdown]
# I'll start by generating the dataset. Each component will have 1000 samples.

# %%
data, labels = make_pinwheel_data(0.3, 0.05, 3, 1000, 0.25)

if ipython is not None:
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:,0], data[:,1], c=labels, s=5)
    plt.show()

# %%
Y = torch.Tensor(data).cuda()

# %% [markdown]
# Now let's instantiate the `Generator` and `Discriminator`. Given the simplicity of the problem, I don't expect to need lots of parameters. I do expect that the balance between the discriminator's capacity and the generator's capacity will play an important role. To see this, I'll experiment a couple different parameter configurations to illustrate different ratios of complexity between generator and discriminator.
#
# Let me define the following acronyms, to refer to the configurations:
#
# - "VS" : "Very simple",
# - "S" : "Simple",
# - "C" : "Complex",
# - "VC" : "Very complex",
# - "G" : "Generator",
# - "D" : "Discriminator"

# %%
D_configs = {
    "VS": dict(ydim=2, hdim=3 , n_hidden=2),
    "S":  dict(ydim=2, hdim=15, n_hidden=3),
    "C":  dict(ydim=2, hdim=15, n_hidden=5),
    "VC": dict(ydim=2, hdim=30, n_hidden=5)
}

G_configs = {
    "VS": dict(ydim=2, xdim=2, hdim=3 , n_hidden=2),
    "S":  dict(ydim=2, xdim=2, hdim=4 , n_hidden=3),
    "C":  dict(ydim=2, xdim=3, hdim=10, n_hidden=4),
    "VC": dict(ydim=2, xdim=4, hdim=15, n_hidden=4)
}

# %%
experiments = dict()

# %%
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

n_epochs = 5000
bs = 128

true_batch_labels = real_label * torch.ones(bs).cuda()
fake_batch_labels = fake_label * torch.ones(bs).cuda()

def train(experiment_key):
    writer = SummaryWriter(f"/workspace/runs/{experiment_key}")

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
                torch.randn(len(yb), gen_xdim, device="cuda:0"),
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
for d_conf_key, d_conf in D_configs.items():
    for g_conf_key, g_conf in G_configs.items():
        netD = Discriminator(**d_conf).cuda()
        netG = Generator(**g_conf, n_cats=3).cuda()
        
        experiment_key = f"D_{d_conf_key} / G_{g_conf_key}"
        
        print(f"Running experiment '{experiment_key}'")
        print(f"Generator params: {count_parameters(netG)}")
        print(f"Discriminator params: {count_parameters(netD)}")
        
        optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.99))
        optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.99))
        
        if experiment_key in experiments:
            continue
        
        gen_xdim = g_conf["xdim"]
        train(experiment_key)
        
        experiments[experiment_key] = {
            "netG": netG.state_dict(),
            "netD": netD.state_dict()
        }
        
    with torch.no_grad():
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20/0.8, 10), sharey=True, sharex=True)
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
        plt.title(experiment_key)
        plt.show()
        plt.close()

# %%
with open("experiment_results.pickle", "wb") as f:
    pickle.dump(experiments, f)

# %%
with torch.no_grad():
    x = np.linspace(-1, 1, 200)
    
    xx, yy = np.meshgrid(x, x)

    zz = netD(torch.Tensor(np.dstack([xx, yy]).reshape(-1, 2)).cuda()).cpu().numpy().reshape([200, 200])
    
    
    
    plt.figure(figsize=(13, 10))
    #p = plt.scatter(Ygrid[:, 0], Ygrid[:, 1], c=netD(torch.Tensor(Ygrid).cuda()).cpu().numpy().squeeze(), s=5, norm=mcolors.Normalize(0, 1))
    p = plt.contourf(xx, yy, zz, 50, cmap="rainbow")
    #p.set_clim(0, 1)
    plt.colorbar(p)
    plt.show()

# %%
f, axs = plt.subplots(4, 4, figsize=(20, 20))


for i, (d_conf_key, d_conf) in enumerate(D_configs.items()):
    for j, (g_conf_key, g_conf) in enumerate(G_configs.items()):
        netD = Discriminator(**d_conf).cuda()
        netG = Generator(**g_conf, n_cats=3).cuda()
        
        experiment_key = f"D_{d_conf_key} / G_{g_conf_key}"
        gen_xdim = g_conf["xdim"]
        
        netD.load_state_dict(experiments[experiment_key]["netD"])
        netG.load_state_dict(experiments[experiment_key]["netG"])
        
        with torch.no_grad():
            latent = torch.cat([
                torch.randn(3000, gen_xdim, device="cuda:0"),
                one_hot_sample(3, 3000, "cuda:0")],
                dim=1
            )

            fake = netG(latent)

            axs[i, j].set_title(f"D: {count_parameters(netD)} // G: {count_parameters(netG)}")
            axs[i, j].scatter(fake[:, 0].cpu(),
                       fake[:, 1].cpu(),
                       s=5,
                       c=latent[:, gen_xdim:].argmax(dim=1).cpu().numpy())

plt.show()

# %% [markdown]
# So the GAN is generally to use each of the subgenerators to specialize on a specific component! Neat!
#
# We see the Generator doesn't need to be extremely complex to be able to generate something remotely close to the dataset. We also see the importance of the balance of complexities between the two nets: basically anytime the Generator has more complexity than the Discriminator we get garbage.
#
# Note that the Discriminator has no clue about the random categorical variable, so the Generator has no explicit incentive to "assign" a component of the dataset to each of the "sub-generators". In fact, we see that in some cases the components overlapped, precisely because the GAN didn't learn to completely segregate them.

# %%

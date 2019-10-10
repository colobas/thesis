# %% [markdown]
# # Imports

# %% [markdown]
# **Import external libraries, classes and methods**

# %%
import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from tqdm import trange
from tensorboardX import SummaryWriter

# %% [markdown]
# **Import custom libraries, classes and methods**

# %%
from normalizing_flows import NormalizingFlow
from normalizing_flows.models import RealNVP
from vmonf import VariationalMixture

from thesis_utils import (
    now_str, 
    count_parameters, 
    figure2tensor, torch_onehot, 
    make_pinwheel_data, 
    make_circles_data
)

# %%
colors = sns.color_palette("bright", 8)
sns.palplot(colors)

# %%
DATASET = "CIRCLES"

# %%
if DATASET == "PINWHEEL":
    X, C = make_pinwheel_data(0.3, 0.05, 5, 8192, 0.4)
    X = torch.Tensor(X)
    C = torch.Tensor(C)

    X_pt, C_pt = make_pinwheel_data(0.3, 0.05, 5, 32, 0.4)
    X_pt = torch.Tensor(X_pt)
    C_pt = torch.Tensor(C_pt)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=C.numpy(), s=5)
    plt.show()
elif DATASET == "CIRCLES":
    X, C = make_circles_data(0.8, 0.4, n_samples=4096)
    X = torch.Tensor(X)
    C = torch.Tensor(C)
    
    X_pt, C_pt = make_circles_data(0.8, 0.4, n_samples=32)
    X_pt = torch.Tensor(X_pt)
    C_pt = torch.Tensor(C_pt)

    plt.figure(figsize=(10, 10))
    plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=C.numpy(), s=5)
    plt.show()
else:
    raise ValueError("Invalid dataset.")

# %%
base_dist = distrib.Normal(loc=torch.zeros(2), scale=torch.ones(2))

xdim = 2
hdim = 16
n_hidden = 2
n_classes = 2
n_realnvp_blocks = 4
hdim_realnvp = 8
n_hidden_realnvp = 2

mixture = VariationalMixture(
    xdim=xdim,
    hdim=hdim,
    n_hidden=n_hidden,
    n_classes=n_classes,
    components=[
        RealNVP(
            n_blocks=n_realnvp_blocks,
            xdim=xdim,
            hdim=hdim_realnvp,
            n_hidden=n_hidden_realnvp,
            base_dist=base_dist
        )
        for _ in range(n_classes)
    ],
)

# %%
count_parameters(mixture)

# %%
x = np.linspace(-1, 1, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

def _pre_backward_callback(mixture, n_iter, log_probs, prior_crossent, q_entropy, temperature, writer):
    if n_iter % 5 == 0:
        writer.add_scalar('train/log_probs', log_probs, n_iter)
        writer.add_scalar('train/prior_crossent', prior_crossent, n_iter)
        writer.add_scalar('train/q_entropy', q_entropy, n_iter)
        writer.add_scalar('train/temperature', temperature, n_iter)

def _post_backward_callback(mixture, n_iter, log_probs, prior_crossent, q_entropy, temperature, writer):
    if n_iter % 100 == 0:
        densities = mixture.forward(torch.Tensor(z), temperature).numpy()
        f = plt.figure(figsize=(10, 10))
        zz = np.argmax(densities, axis=1).reshape([1000, 1000])

        plt.contourf(xx, yy, zz, 50, cmap="rainbow")

        with torch.no_grad():
            for i, component in enumerate(mixture.components):
                X_k = component.sample(512)

                plt.scatter(X_k[:, 0].numpy(), X_k[:, 1].numpy(),
                            c=[colors[i]],
                            s=5)

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        
        writer.add_image("distributions", figure2tensor(f), n_iter)
        plt.close(f)        


# %%
#n_epochs = 50
#bs = 16
#lr = 1e-3

#opt = optim.Adam(mixture.parameters(), lr=lr)
#opt.zero_grad()

#mixture.pretrain(
#    DataLoader(TensorDataset(X_pt, torch_onehot(C_pt, 2)), batch_size=bs, shuffle=True, num_workers=0),
#    n_epochs,
#    opt,
#    verbose=True
#)

# %%
n_epochs = 500
bs = 128
lr = 1e-3


writer = SummaryWriter(
    f"./tensorboard_logs/{DATASET}_{n_epochs}epochs_{bs}bs_{lr}lr"
    f"_{hdim}hdim_{hdim_realnvp}hdim_realnvp"
    f"_SS_"
    f"_{now_str()[:-7]}"
)


opt = optim.Adam(mixture.parameters(), lr=lr)

opt.zero_grad()

best_loss, best_params = mixture.fit(
        dataloader=DataLoader(X, batch_size=bs, shuffle=True, num_workers=0),
        n_epochs=n_epochs,
        opt=opt,
        temperature_schedule=lambda t: np.exp(-1e-3 * t),
        verbose=True,
        pre_backward_callback=lambda m, n, l, p, q, t: _pre_backward_callback(m, n, l, p, q, t, writer),
        post_backward_callback=lambda m, n, l, p, q, t: _post_backward_callback(m, n, l, p, q, t, writer),
        sup_dataloader=DataLoader(TensorDataset(X_pt, torch_onehot(C_pt, 2)), batch_size=bs, shuffle=True, num_workers=0)
)

# %%
mixture.load_state_dict(best_params)

# %%
x = np.linspace(-1, 1, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = mixture.forward(torch.Tensor(z)).numpy()

mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

f, axs = plt.subplots(1, len(mixture.components), figsize=(30, 10))

for i, ax in enumerate(axs):
    zz = densities[:,i].reshape([1000, 1000])
    ax.set_title(f"$q(z={i} | x)$")
    cb = ax.contourf(xx, yy, zz, 50, cmap="rainbow")


plt.colorbar(cb)
plt.tight_layout(h_pad=1)
plt.show()

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 10))

zz = np.argmax(densities, axis=1).reshape([1000, 1000])

ax.contourf(xx, yy, zz, 50, cmap="rainbow")

with torch.no_grad():
    c = [colors[i] for i in np.argmax(mixture.forward(X).numpy(), axis=1)]
    
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=c, s=5)

plt.show()

# %%
q = torch_onehot(C, 3)

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 10))

zz = np.argmax(densities, axis=1).reshape([1000, 1000])

ax.contourf(xx, yy, zz, 50, cmap="rainbow")

colors = ["yellow", "green", "black"]
with torch.no_grad():
    for i, component in enumerate(mixture.components):
        X_k = component.sample(2048)

        ax.scatter(X_k[:, 0].numpy(), X_k[:, 1].numpy(), c=colors[i],
            s=5)

plt.show()

# %%
x = np.linspace(-20, 20, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = mixture.forward(torch.Tensor(z)).numpy()

mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

f, axs = plt.subplots(1, 3, figsize=(30, 10))

for i, ax in enumerate(axs):
    zz = densities[:,i].reshape([1000, 1000])
    ax.set_title(f"$q(z={i} | x)$")
    cb = ax.contourf(xx, yy, zz, 50, cmap="rainbow")


plt.colorbar(cb)
plt.tight_layout(h_pad=1)
plt.show()

# %%

# %%

# %%

# %%

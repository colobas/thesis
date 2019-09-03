# %%
import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.optim as optim

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
import matplotlib
import matplotlib.pyplot as plt
import datetime

from copy import deepcopy

from tqdm import trange
from tensorboardX import SummaryWriter

# %%
from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import PReLUFlow, AffineLUFlow, BatchNormFlow, StructuredAffineFlow

# %%
from thesis_utils import now_str, count_parameters, figure2tensor


# %%
def gen_samples(batch_size=512):
    x2_dist = distrib.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample((batch_size,))

    x1 = distrib.Normal(loc=.25 * (x2_samples.pow(2)),
                  scale=torch.ones((batch_size,)))

    x1_samples = x1.sample()
    return torch.stack([x1_samples, x2_samples]).t()

X = gen_samples(512)

# %%
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], s=5)
plt.show()

# %%
base_dist = distrib.Normal(loc=torch.zeros(2), scale=torch.ones(2))

# %%
X0 = base_dist.sample((1000,)).numpy()

# %%
colors = np.zeros(len(X0))

idx_0 = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
colors[idx_0] = 0
idx_1 = np.logical_and(X0[:, 0] >= 0, X0[:, 1] < 0)
colors[idx_1] = 1
idx_2 = np.logical_and(X0[:, 0] >= 0, X0[:, 1] >= 0)
colors[idx_2] = 2
idx_3 = np.logical_and(X0[:, 0] < 0, X0[:, 1] >= 0)
colors[idx_3] = 3

# %%
plt.scatter(X0[:, 0], X0[:, 1], s=5, c=colors)


# %%
def get_density(cur_z, prev_density, flow):
    density = prev_density.squeeze() / np.exp(flow.log_abs_det_jacobian(torch.Tensor(cur_z), None).detach().squeeze())
    return torch.Tensor(density)


# %%
def get_meshes(cur_z, density, grid_side=1000, dim=2):
    mesh = cur_z.reshape([grid_side, grid_side, dim]).transpose(2, 0, 1)
    xx = mesh[0]
    yy = mesh[1]
    zz = density.numpy().reshape([grid_side, grid_side])
    
    return xx, yy, zz


# %%
blocks = [PReLUFlow(2)]

flow = NormalizingFlow( 
    *blocks,
    base_dist=base_dist,
)

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
x = np.linspace(-5, 5, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = flow.base_dist.log_prob(torch.Tensor(z)).sum(dim=1).exp().numpy()

mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

f, ax = plt.subplots(figsize=(10, 10))

zz = densities.reshape([1000, 1000])
cb = ax.contourf(xx, yy, zz, 50, cmap="rainbow")
ax.set_aspect("equal")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cb, cax=cax)
plt.show()

# %%
x = np.linspace(-4, 4, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = flow.log_prob(torch.Tensor(z)).exp().numpy()
    #densities = flow.log_prob(torch.Tensor(z)).numpy()
    
mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

f, ax = plt.subplots(figsize=(10, 10))

zz = densities.reshape([1000, 1000])
cb = ax.contourf(xx, yy, zz, 50, cmap="rainbow")
ax.set_aspect("equal")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(cb, cax=cax)
plt.show()

# %%
flow[0].alpha.data = torch.tensor([5.])

# %%
with torch.no_grad():
    y, _ = flow(torch.tensor(X0))

# %%
COLORS = ["red", "green", "blue", "orange"]

# %%
plt.figure(figsize=(10, 10))
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.scatter(X0[:, 0], X0[:, 1], s=10, c=[COLORS[int(i)] for i in colors])

# %%
plt.figure(figsize=(10, 10))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter(y[:, 0], y[:, 1], s=10)#, c=[COLORS[int(i)] for i in colors])

# %%

# %%

# %%

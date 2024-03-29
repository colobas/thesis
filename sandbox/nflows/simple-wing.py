# %%
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
.container { 
    width:80% !important;
}
</style>
""")

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
import matplotlib.pyplot as plt
import datetime

from copy import deepcopy

from tqdm import trange
from tensorboardX import SummaryWriter

# %%
from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import AffineFlow, PReLUFlow, StructuredAffineFlow, AffineLUFlow

# %%
from thesis_utils import now_str, count_parameters, figure2tensor


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
X, c = make_pinwheel_data(0.3, 0.05, 1, 512, 0.25)
X = torch.Tensor(X)
c = torch.Tensor(c)

plt.figure(figsize=(10, 10))
plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=c.numpy(), s=5)
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
flow = NormalizingFlow(
    dim=2, 
    blocks=([StructuredAffineFlow, PReLUFlow]*5 + [StructuredAffineFlow]),
    base_density=base_dist,
    flow_length=1
)

opt = optim.Adam(flow.parameters(), lr=1e-3)

# %%
count_parameters(flow)

# %%
n_epochs = 100000
bs = 512

# %%
writer = SummaryWriter(f"/workspace/sandbox/tensorboard_logs/{now_str()}")

best_loss = torch.Tensor([float("+inf")])

attempts = 0

for epoch in trange(n_epochs):
    batches = range((len(X) - 1) // bs + 1)
    for i in batches:
        start_i = i * bs
        end_i = start_i + bs
        xb = X[start_i:end_i]
        it = epoch*len(batches) + i + 1

        opt.zero_grad()
        loss = -flow.final_density.log_prob(xb).mean()

        if loss <= 0:
            if attempts < 100:
                attempts += 1
                continue
            else:
                print("Loss has diverged, halting train and not backpropagating")
                break

        if loss <= best_loss:
            best_loss = loss
            best_flow = deepcopy(flow)

        loss.backward()
        opt.step()
        
    if epoch % 5 == 0:
        writer.add_scalar("loss", loss, it)
    if epoch % 200 == 0:
        with torch.no_grad():
            Xhat = flow.final_density.sample((1000, ))
            f = plt.figure(figsize=(10, 10))
            plt.xlim(-30, 30)
            plt.ylim(-20, 20)
            plt.title(f"{it} iterations")
            plt.scatter(Xhat[:, 0], Xhat[:, 1], s=5, c="red", alpha=0.5)
            plt.scatter(X[:, 0], X[:, 1], s=5, c="blue", alpha=0.5)
            writer.add_image("distributions", figure2tensor(f), it)
            plt.close(f)

# %%
flow = best_flow

# %%
Xhat = flow.final_density.sample((1000, ))
plt.scatter(X[:, 0], X[:, 1], s=5, c="blue", alpha=0.5)
plt.scatter(Xhat[:, 0], Xhat[:, 1], s=5, c="red", alpha=0.5)
plt.show()

# %%
n_flows = len(flow.bijectors)

f, axs = plt.subplots(
    n_flows + 1,
    1,
    figsize=(10, n_flows*12),
    #sharex=True,
    #sharey=True
)

axs[0].scatter(X0[:, 0], X0[:, 1], s=5, c=colors)

cur_x = X0

for ax, bij in zip(axs[1:], flow.bijectors[:n_flows+1]):
    cur_x = bij(torch.Tensor(cur_x)).detach().numpy()
    ax.scatter(cur_x[:, 0], cur_x[:, 1], s=5, c=colors)
    
plt.show()

# %%
x = np.linspace(-4, 4, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])


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
f, axs = plt.subplots(
    n_flows + 1,
    1,
    figsize=(10, n_flows*12),
    #sharex=True,
    #sharey=True
)

cur_z = z
prev_density = (flow.base_density
                    .log_prob(torch.Tensor(cur_z))
                    .sum(dim=1)
                    .exp().detach())

xx, yy, zz = get_meshes(cur_z, prev_density)

axs[0].contourf(xx, yy, zz, 50, cmap='rainbow')

for ax, bij in zip(axs[1:], flow.bijectors[:n_flows+1]):
    cur_z = bij(torch.Tensor(cur_z)).detach().numpy()
    
    prev_density = get_density(cur_z, prev_density, bij)
    xx, yy, zz = get_meshes(cur_z, prev_density)
    ax.contourf(xx, yy, zz, 50, cmap='rainbow')
    
plt.show()

# %%
list(flow.named_parameters())

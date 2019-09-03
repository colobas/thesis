# %%
import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.optim as optim
from torch.utils.data import DataLoader

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
from normalizing_flows.models import RealNVP

from thesis_utils import now_str, count_parameters, figure2tensor


# %%
def gen_samples(batch_size=512):
    x2_dist = distrib.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample((batch_size,))

    x1 = distrib.Normal(loc=.25 * (x2_samples.pow(2)),
                  scale=torch.ones((batch_size,)))

    x1_samples = x1.sample()
    return torch.stack([x1_samples, x2_samples]).t()
    
    
X = gen_samples(2048)

# %%
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
flow = RealNVP(
    n_blocks=4,
    input_size=2,
    hidden_size=3,
    n_hidden=1,
    base_dist=base_dist
)

opt = optim.Adam(flow.parameters(), lr=1e-2)

# %%
count_parameters(flow)

# %%
n_epochs = 200
bs = 128

# %%
writer = SummaryWriter(f"../tensorboard_logs/{now_str()}")
dataloader = DataLoader(X, batch_size=bs, shuffle=True, num_workers=0)
best_loss = torch.Tensor([float("+inf")])

#attempts = 0

for epoch in trange(n_epochs):
    batches = range((len(X) - 1) // bs + 1)
    for i, xb in enumerate(dataloader):
        start_i = i * bs
        end_i = start_i + bs
        xb = X[start_i:end_i]
        it = epoch*len(batches) + i + 1

        opt.zero_grad()
        loss = -flow.log_prob(xb).mean()
        
        #z, log_abs_det_jacobian = flow.inverse(xb)
        #base_log_probs = flow.base_dist.log_prob(z)
        
        #base_log_probs_ = -base_log_probs.mean()
        #log_abs_det_jacob_ = -log_abs_det_jacobian.mean()
        
        #loss = (base_log_probs_ + log_abs_det_jacob_)

        #if loss <= 0:
        #    if attempts < 100:
        #        attempts += 1
        #        continue
        #    else:
        #        print("Loss has diverged, halting train and not backpropagating")
        #        break

        if loss <= best_loss:
            best_loss = loss
            best_flow_params = flow.state_dict()

        loss.backward()
        opt.step()
        
        if it % 2 == 0:
            writer.add_scalar("debug/loss", loss, it)
            #writer.add_scalar("debug/log_probs", base_log_probs_, it)
            #writer.add_scalar("debug/log_abs_det_jacobian", log_abs_det_jacob_, it)
        if it % 5 == 0:
            with torch.no_grad():
                Xhat = flow.sample(1000)
                f = plt.figure(figsize=(10, 10))
                plt.xlim(-30, 30)
                plt.ylim(-20, 20)
                plt.title(f"{it} iterations")
                plt.scatter(X[:, 0], X[:, 1], s=5, c="blue", alpha=0.5)
                plt.scatter(Xhat[:, 0], Xhat[:, 1], s=5, c="red", alpha=0.5)
                writer.add_image("debug/samples", figure2tensor(f), it)
                plt.close(f)

                f = plt.figure(figsize=(10, 10))
                plt.title(f"{it} iterations")
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                z = flow.base_dist.sample((1000,))
                zhat, _ = flow.inverse(X)
                plt.scatter(z[:, 0], z[:, 1], s=5, c="blue", alpha=0.5)
                plt.scatter(zhat[:, 0], zhat[:, 1], s=5, c="red", alpha=0.5)
                writer.add_image("debug/base_dist", figure2tensor(f), it)
                plt.close(f)

# %%
loss

# %%
flow.eval()
xhat_samples = flow.sample(1000)
plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
plt.scatter(X[:, 0], X[:, 1], s=5, c="blue")
#plt.xlim(0, 60)
#plt.ylim(-15, 15)
plt.show()

# %%
x2 = flow.net[-2].x2_mask(X)

# %%
with torch.no_grad():
    zhat, _ = flow.inverse(X)

plt.scatter(zhat[:, 0], zhat[:, 1], s=5, c="red")
#plt.xlim(0, 60)
#plt.ylim(-15, 15)
plt.show()

# %%
n_flows = len(flow.net)

f, axs = plt.subplots(
    n_flows + 1,
    1,
    figsize=(10, n_flows*12),
    #sharex=True,
    #sharey=True
)

axs[0].scatter(X0[:, 0], X0[:, 1], s=5, c=colors)

cur_x = X0

for ax, bij in zip(axs[1:], [flow.net[i] for i in range(n_flows)]):
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

# %%
z, log_abs_det_jacobian = flow.inverse(X)


# %%
def inverse(self, x):
    log_abs_dets = []
    z_cur = x
    for flow in reversed(self):
        z_prev = flow.inverse(z_cur)
        log_abs_dets.append(
            -flow.log_abs_det_jacobian(z_prev, z_cur)
        )
        z_cur = z_prev

    return z_prev, log_abs_dets

with torch.no_grad():
    z, log_abs_dets = inverse(flow.net, X)

# %%
plt.scatter(z[:, 0].numpy(), z[:, 1].numpy(), s=5)
plt.show()

# %%
sum(log_abs_dets)

# %%
with torch.no_grad():
    lp = flow.base_dist.log_prob(z)

# %%
lp.sum(dim=1)

# %%
(lp.sum(dim=1) + log_abs_det_jacobian).mean()

# %%
flow.log_prob(X).mean()

# %%
from sklearn import datasets

# %%
train_data = datasets.make_moons(n_samples=50000, noise=.05)[0].astype(np.float32)

# %%
train_data.shape

# %%

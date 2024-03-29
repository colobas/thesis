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
f1 = AffineLUFlow(2)

# %%
f1.forward(X)

# %%
#blocks = sum(
#    [[AffineLUFlow(2), PReLUFlow(2), BatchNormFlow(2)] for _ in range(5)] + 
#    [[AffineLUFlow(2)]],
#[])

blocks = sum(
    [[StructuredAffineFlow(2), PReLUFlow(2), BatchNormFlow(2)] for _ in range(5)] + 
    [[StructuredAffineFlow(2)]],
[])

flow = NormalizingFlow( 
    *blocks,
    base_dist=base_dist,
)

opt = optim.Adam(flow.parameters(), lr=1e-2)

# %%
count_parameters(flow)

# %%
n_epochs = 2000
bs = 512
clip_grad = 1e6

# %%
writer = SummaryWriter(f"../tensorboard_logs/{now_str()}")

best_loss = torch.Tensor([float("+inf")])

attempts = 0

flow.train()
for epoch in trange(n_epochs):
    batches = range((len(X) - 1) // bs + 1)
    for i in batches:
        start_i = i * bs
        end_i = start_i + bs
        xb = X[start_i:end_i]
        it = epoch*len(batches) + i + 1

        opt.zero_grad()    
        loss = -flow.log_prob(xb).mean()

        #if loss <= 0:
        #    if attempts < 100:
        #        attempts += 1
        #        continue
        #    else:
        #        print("Loss has diverged, halting train and not backpropagating")
        #        break

        if loss <= best_loss:
            best_loss = loss
            best_params = flow.state_dict()

        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), clip_grad)
        
        opt.step()
        
    if it % 5 == 0:
        writer.add_scalar("loss", loss, it)
    if it % 200 == 0:
        with torch.no_grad():
            Xhat = flow.sample(1000)
            f = plt.figure(figsize=(10, 10))
            plt.xlim(-30, 30)
            plt.ylim(-20, 20)
            plt.title(f"{it} iterations")
            plt.scatter(Xhat[:, 0], Xhat[:, 1], s=5, c="red", alpha=0.5)
            plt.scatter(X[:, 0], X[:, 1], s=5, c="blue", alpha=0.5)
            writer.add_image("distributions", figure2tensor(f), it)
            plt.close(f)
            
flow.eval()

# %%
for f in flow:
    if hasattr(f, "log_abs_s"):
        print(f.log_abs_s)

# %%
flow.load_state_dict(best_params)

# %%
Xhat = flow.sample(1000)
plt.scatter(X[:, 0], X[:, 1], s=5, c="blue", alpha=0.5)
plt.scatter(Xhat[:, 0], Xhat[:, 1], s=5, c="red", alpha=0.5)
plt.show()

# %%
n_flows = len(flow)

f, axs = plt.subplots(
    n_flows + 1,
    1,
    figsize=(10, n_flows*12),
    #sharex=True,
    #sharey=True
)

axs[0].scatter(X0[:, 0], X0[:, 1], s=5, c=colors)

cur_x = X0

for ax, bij in zip(axs[1:], [bij for bij in flow]):
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
x = np.linspace(-20, 20, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = flow.log_prob(torch.Tensor(z)).exp().numpy()

mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

plt.figure(figsize=(10, 10))

zz = densities.reshape([1000, 1000])
cb = plt.contourf(xx, yy, zz, 50, cmap="rainbow")

plt.colorbar(cb)
plt.tight_layout(h_pad=1)
plt.show()

# %%
flow2 = NormalizingFlow( 
    flow[3],
    base_dist=base_dist,
)

# %%
x = np.linspace(-5, 5, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = flow2.log_prob(torch.Tensor(z)).exp().numpy()

mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]

plt.figure(figsize=(10, 10))

zz = densities.reshape([1000, 1000])
cb = plt.contourf(xx, yy, zz, 50, cmap="rainbow")

plt.colorbar(cb)
plt.tight_layout(h_pad=1)
plt.show()

# %%

# %%
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

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
from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import AffineFlow, PReLUFlow, StructuredAffineFlow, AffineLUFlow, CouplingLayerFlow

# %%
import matplotlib.pyplot as plt
import datetime

from copy import deepcopy

from tqdm import trange
from tensorboardX import SummaryWriter

# %%
now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def gen_samples(batch_size=512):
    x2_dist = distrib.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample((batch_size,))

    x1 = distrib.Normal(loc=.25 * (x2_samples.pow(2)),
                  scale=torch.ones((batch_size,)))

    x1_samples = x1.sample()
    return torch.stack([x1_samples, x2_samples]).t()

x_samples = gen_samples(512)

# %%
plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5)
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
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# %%
flow = NormalizingFlow(
    dim=2, 
    blocks=[CouplingLayerFlow, CouplingLayerFlow],
    base_density=base_dist,
    flow_length=1,
    block_args=[
        (MLP(1, 3, 1), MLP(1, 3, 1), (0,)),
        (MLP(1, 3, 1), MLP(1, 3, 1), (1,))
    ]
)

opt = optim.Adam(flow.parameters(), lr=1e-3)

# %%
count_parameters(flow)

# %%
x_samples = distrib.Normal(loc=torch.zeros(2), scale=torch.ones(2)*0.3).sample((1024, ))
plt.scatter(x_samples[:, 0].numpy(), x_samples[:, 1].numpy(), s=5)

# %%
writer = SummaryWriter(f"/workspace/sandbox/tensorboard_logs/{now_str()}")

best_loss = torch.Tensor([float("+inf")])

attempts = 0

for it in trange(int(1e5)):
    opt.zero_grad()
    loss = -flow.final_density.log_prob(x_samples).mean()
    
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
    if it % 50 == 0:
        writer.add_scalar("loss", loss, it)
    
    if it % 5000 == 0:
        with torch.no_grad():
            xhat_samples = flow.final_density.sample((1000, ))
            plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
            plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5, c="blue")
            #plt.xlim(0, 60)
            #plt.ylim(-15, 15)
            plt.show()

#    if it % 100 == 0:
#        f = plt.figure(figsize=(20, 20))
#        xhat_samples = flow.final_density.sample((1000, ))
#        plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
#        plt.xlim(-5, 40)
#        plt.ylim(-15, 15)
#        plt.title(f"{loss.detach().numpy()}")
#        plt.savefig(f"to_gif/it_{it}.png")
#        plt.close()

    opt.step()

# %%
flow = best_flow

# %%
xhat_samples = flow.final_density.sample((1000, ))
plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5, c="blue")
#plt.xlim(0, 60)
#plt.ylim(-15, 15)
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

# %%

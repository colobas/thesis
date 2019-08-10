# %%
import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt
import datetime

from tqdm import trange
from tensorboardX import SummaryWriter

from copy import deepcopy

# %%
from thesis_utils import now_str, count_parameters, figure2tensor, torch_onehot

from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import PReLUFlow, StructuredAffineFlow


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
class MLP(nn.Module):
    def __init__(self, xdim, hdim, n_hidden, n_classes):
        super().__init__()

        self.xdim = xdim

        net_modules = (
          [nn.Linear(xdim, hdim), nn.LeakyReLU()] +
          sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, n_classes)]
        )

        self.encoder = nn.Sequential(*net_modules)

    def forward(self, x, T=1):
        x = self.encoder(x)
        return F.softmax(x/T, dim=1)



# %%
X, c = make_pinwheel_data(0.3, 0.05, 3, 1000, 0.25)
X = torch.Tensor(X)
c = torch.Tensor(c)

plt.figure(figsize=(10, 10))
plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=c.numpy(), s=5)
plt.show()

# %%
xdim = 2
hdim = 2
n_hidden = 3
n_classes = 3

mlp = MLP(
    xdim=xdim,
    hdim=hdim,
    n_hidden=n_hidden,
    n_classes=n_classes
)

# %%
count_parameters(mlp)

# %%
n_epochs = 10000
bs = 512
opt = optim.Adam(mlp.parameters(), lr=0.001)

# %%
c_onehot = torch_onehot(c, 3)

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
        cb = c_onehot[start_i:end_i]
        it = epoch*len(batches) + i + 1

        opt.zero_grad()
        loss = F.binary_cross_entropy(mlp(xb), cb) 

        if loss <= 0:
            if attempts < 100:
                attempts += 1
                continue
            else:
                print("Loss has diverged, halting train and not backpropagating")
                break

        if loss <= best_loss:
            best_loss = loss
            best_params = mlp.state_dict()

        loss.backward()
        opt.step()
        
    if epoch % 5 == 0:
        writer.add_scalar("loss", loss, it)

# %%
mlp.load_state_dict(best_params)

# %%
x = np.linspace(-3, 3, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])

with torch.no_grad():
    densities = mlp.forward(torch.Tensor(z)).numpy()

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
f, ax = plt.subplots(1, 1, figsize=(10, 10))

zz = np.argmax(densities, axis=1).reshape([1000, 1000])

ax.contourf(xx, yy, zz, 50, cmap="rainbow")

colors = ["yellow", "white", "black"]

with torch.no_grad():
    cl = [colors[i] for i in np.argmax(mlp.forward(X).numpy(), axis=1)]
    
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=cl, s=5)

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

# %%

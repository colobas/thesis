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

from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import PReLUFlow, StructuredAffineFlow

# %%
from thesis_utils import now_str, count_parameters, figure2tensor

# %%
import io
from bad_grad_viz import register_hooks


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
#debug = False
#debug_str = io.StringIO("")

class VariationalMixture(nn.Module):
    def __init__(self, xdim, hdim, n_hidden, n_classes, components):
        super().__init__()

        self.xdim = xdim

        net_modules = (
          [nn.Linear(xdim, hdim), nn.LeakyReLU()] +
          sum([[nn.Linear(hdim, hdim), nn.LeakyReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, n_classes)]
        )

        self.encoder = nn.Sequential(*net_modules)

        self.components = nn.ModuleList(components)
        self.n_components = len(components)

        self.log_prior = torch.Tensor([1/len(components)]*len(components)).log()

    def forward(self, x, T=1):
        x = self.encoder(x)
        return F.softmax(x/T, dim=1)

    def update_mixture_weights(self, x):
        num = torch.zeros(len(x), len(components))
        for k in range(self.n_components):
            num[:, k] = (self.n_components[k].log_prob(x) + self.log_prior[k]).exp()

        num = num / num.sum(dim=1)

        self.log_prior = (num.mean(dim=0) + 1e-6).log()

        return self.log_prior
    
    def _centroid_distances(self, X, q):
        Xbar = (X.unsqueeze(1) * q.unsqueeze(-1)).sum(dim=0) / q.sum(dim=0).unsqueeze(-1)

        diffs = (X.unsqueeze(1) - Xbar)

        covs = ((diffs.unsqueeze(3) @ diffs.unsqueeze(2)) * 
          q.unsqueeze(-1).unsqueeze(-1)).sum(dim=0) / (
          q.sum(dim=0).unsqueeze(-1).unsqueeze(-1))
                
        return (((diffs ** 2 / (covs.diagonal(dim1=1, dim2=2) ** 2).unsqueeze(0))
                  * q.unsqueeze(-1))
                     .sum(dim=1)
                     .sum(dim=1)
                     .sqrt())
        
    
    def pretrain_loss(self, X):
        q = self.forward(X)
        
        distances = self._centroid_distances(X, q)
        
        qmean = q.mean(dim=0)
        
        return (0
                + distances.mean()
                - (q * (q + 1e-6).log()).mean() 
                - (qmean * (qmean + 1e-6).log()).mean()
        )

    def elbo(self, x, T=1):
        q = self.forward(x, T)

        log_probs = 0
        for k in range(self.n_components):
            log_probs += q[:, k] * self.components[k].log_prob(x)

        log_probs = log_probs.sum()
        prior_crossent = (q * self.log_prior).sum(dim=1).sum()
        q_entropy = - (q * (q + 1e-6).log()).sum(dim=1).sum()

        return log_probs + prior_crossent + q_entropy

    def fit(self, X, n_epochs=1, bs=100, opt=None, temperature_schedule=None,
            clip_grad=None, verbose=False, writer=None, is_pretraining=False):

        best_loss = float("inf")
        best_params = dict()

        if opt is None:
            if is_pretraining:
                opt = optim.Adam(self.encoder.parameters(), lr=0.01)
            else:
                opt = optim.Adam(self.parameters(), lr=0.001)

        if temperature_schedule is None:
            temperature_schedule = lambda t: max(0.01, np.exp(-1e-4 * t))

        if verbose:
            epochs = trange(n_epochs, desc="epoch")
        else:
            epochs = range(n_epochs)

        for epoch in epochs:
            batches = range((len(X) - 1) // bs + 1)
            for i in batches:
                start_i = i * bs
                end_i = start_i + bs
                xb = X[start_i:end_i]
                n_iter = epoch*len(X) + start_i
                
                if is_pretraining:
                    loss = self.pretrain_loss(xb)
                else:
                    loss = - self.elbo(xb, temperature_schedule(n_iter))
                if loss <= best_loss:
                    best_loss = loss.item()
                    if is_pretraining:
                        best_params = self.encoder.state_dict()
                    else:
                        best_params = self.state_dict()

                # if we're writing to tensorboard
                if writer is not None:
                    if n_iter % 10 == 0:
                        if is_pretraining:
                            writer.add_scalar('losses/pretraining_loss', loss, n_iter)
                        else:
                            writer.add_scalar('losses/loss', loss, n_iter)
                            writer.add_scalar('misc/temperature', temperature_schedule(n_iter), n_iter)

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

                opt.step()
                opt.zero_grad()

        return best_loss, best_params

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

n_mlp_flows = 5

mixture = VariationalMixture(
    xdim=xdim,
    hdim=hdim,
    n_hidden=n_hidden,
    n_classes=n_classes,
    components=[NormalizingFlow(
        dim=2, 
        blocks=([StructuredAffineFlow, PReLUFlow]*n_mlp_flows + [StructuredAffineFlow]),
        base_density=distrib.Normal(loc=torch.zeros(2), scale=torch.ones(2)),
        flow_length=1
    ) for _ in range(n_classes)],
)

# %%
count_parameters(mixture)

# %%
n_epochs = 100000
bs = 512
opt = optim.Adam(mixture.parameters(), lr=0.0005)
writer = SummaryWriter(f"/workspace/runs/{now_str()}")

best_loss, best_params = mixture.fit(X,
    n_epochs=n_epochs,
    bs=bs,
    opt=opt,
    temperature_schedule=None,
    clip_grad=1e5,
    verbose=True,
    writer=writer,
    is_pretraining=False)

# %%
mixture.load_state_dict(best_params)

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
f, ax = plt.subplots(1, 1, figsize=(10, 10))

zz = np.argmax(densities, axis=1).reshape([1000, 1000])

ax.contourf(xx, yy, zz, 50, cmap="rainbow")

colors = ["yellow", "white", "black"]

with torch.no_grad():
    c = [colors[i] for i in np.argmax(mixture.forward(X).numpy(), axis=1)]
    
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=c, s=5)

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()


# %%
def torch_onehot(y, n_cat):
    return (
        torch.zeros(len(y), n_cat)
            .scatter_(1, y.type(torch.LongTensor).unsqueeze(-1), 1)
    )


# %%
q = torch_onehot(c, 3)

# %%
with torch.no_grad():
    d = mixture._centroid_distances(X, q)

# %%
colors = torch_onehot(c, 3).numpy()


# %%

# %%
colors = np.hstack((torch_onehot(c, 3).numpy(), (d/d.max()).numpy().reshape(-1, 1)))

plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), color=colors)

# %%

# %%

# %%
n_epochs = 200
bs = 64
opt = optim.Adam(mixture.parameters(), lr=0.01)
writer = SummaryWriter(f"/workspace/runs/{now_str()}")

best_loss, best_params = mixture.fit(X,
    n_epochs=n_epochs,
    bs=bs,
    opt=opt,
    temperature_schedule=lambda t: max(0.01, np.exp(-1e-5 * t)),
    clip_grad=1e3,
    verbose=True,
    writer=writer)

# %%
mixture.load_state_dict(best_params)

# %%
colors = ["red", "green", "blue"]

with torch.no_grad():
    for i, component in enumerate(mixture.components):
        X_k = distrib.LowRankMultivariateNormal(component.loc,
                                                component.sqrt_cov_factor ** 2 + 0.1,
                                                component.sqrt_cov_diag ** 2 + 0.1).sample((500,))

        plt.scatter(X_k[:, 0].numpy(), X_k[:, 1].numpy(), c=colors[i],
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
f, ax = plt.subplots(1, 1, figsize=(10, 10))

zz = np.argmax(densities, axis=1).reshape([1000, 1000])

ax.contourf(xx, yy, zz, 50, cmap="rainbow")

colors = ["yellow", "white", "black"]

with torch.no_grad():
    for i, component in enumerate(mixture.components):
        X_k = distrib.LowRankMultivariateNormal(component.loc,
                                                component.sqrt_cov_factor ** 2 + 0.1,
                                                component.sqrt_cov_diag ** 2 + 0.1).sample((500,))

        ax.scatter(X_k[:, 0].numpy(), X_k[:, 1].numpy(), c=colors[i],
                    s=5)

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()

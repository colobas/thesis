# %%
import numpy as np

np.random.seed(0)

import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt
import datetime

from tqdm import trange
from tensorboardX import SummaryWriter

from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import PReLUFlow, StructuredAffineFlow, AffineLUFlow, BatchNormFlow

# %%
from thesis_utils import now_str, count_parameters, figure2tensor, torch_onehot
from thesis_utils import RAdam, Lookahead, Ranger

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
def module_grad_health(module):
    flat = torch.cat([param.grad.abs().flatten()
                      for param in module.parameters()])
    return (
        flat.median(),
        flat.max(),
        flat.min()
    )


# %%
x = np.linspace(-1, 1, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])


mesh = z.reshape([1000, 1000, 2]).transpose(2, 0, 1)
xx = mesh[0]
yy = mesh[1]


# %%
#debug = False
#debug_str = io.StringIO("")

class VariationalMixture(nn.Module):
    def __init__(self, xdim, hdim, n_hidden, n_classes, components):
        super().__init__()

        self.xdim = xdim

        net_modules = (
          [nn.Linear(xdim, hdim), nn.ReLU(), nn.BatchNorm1d(hdim)] +
          sum([[nn.Linear(hdim, hdim), nn.ReLU(), nn.BatchNorm1d(hdim)] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, n_classes)]
        )

        self.encoder = nn.Sequential(*net_modules)

        self.components = nn.ModuleList(components)
        self.n_components = len(components)

        self.log_prior = torch.Tensor([1/len(components)]*len(components)).log()
    
    def to(self, device="cuda:0"):
        super(VariationalMixture, self).to(device)
        for component in self.components:
            component.to(device)
        
        self.log_prior = self.log_prior.to(device)
        
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

    def elbo(self, x, T=1):
        q = self.forward(x, T)

        log_probs = 0
        for k in range(self.n_components):
            log_probs = log_probs + q[:, k] * self.components[k].log_prob(x)

        log_probs = log_probs.sum()
        prior_crossent = (q * self.log_prior).sum(dim=1).sum()
        q_entropy = - (q * (q + 1e-6).log()).sum(dim=1).sum()

        return log_probs, prior_crossent, q_entropy


    def fit(self, X, dataloader, n_epochs=1, opt=None,
            temperature_schedule=None, clip_grad=None,
            verbose=False, writer=None):

        best_loss = float("inf")
        best_params = dict()

        if temperature_schedule is None:
            temperature_schedule = lambda t: 1

        if verbose:
            epochs = trange(n_epochs, desc="epoch")
        else:
            epochs = range(n_epochs)

        for epoch in epochs:
            for i, xb in enumerate(dataloader):
                opt.zero_grad()
                n_iter = epoch*((len(X) - 1) // dataloader.batch_size + 1) + i
            
                log_probs,prior_crossent,q_entropy = self.elbo(xb, temperature_schedule(n_iter))
                loss = -(log_probs + prior_crossent + q_entropy)
                
                if loss != loss:
                    continue
                
                if loss <= best_loss:
                    best_loss = loss.item()
                    best_params = self.state_dict()

                # if we're writing to tensorboard
                if writer is not None:
                    if n_iter % 20 == 0:
                        writer.add_scalar('losses/log_probs', log_probs, n_iter)
                        #writer.add_scalar('losses/prior_crossent', prior_crossent, n_iter)
                        writer.add_scalar('losses/q_entropy', q_entropy, n_iter)

                loss.backward()
                
                if n_iter % 100 == 0:
                    with torch.no_grad():
                        densities = mixture.forward(torch.Tensor(z)).numpy()
                        f = plt.figure(figsize=(10, 10))
                        zz = np.argmax(densities, axis=1).reshape([1000, 1000])

                        plt.contourf(xx, yy, zz, 50, cmap="rainbow")

                        colors = ["yellow", "green", "black", "cyan"]
                        with torch.no_grad():
                            for i, component in enumerate(mixture.components):
                                X_k = component.sample(500)

                                plt.scatter(X_k[:, 0].numpy(), X_k[:, 1].numpy(), c=colors[i],
                                    s=5)

                        plt.xlim(-1.1, 1.1)
                        plt.ylim(-1.1, 1.1)
                        writer.add_image("distributions", figure2tensor(f), n_iter)
                        plt.close(f)


                #if clip_grad is not None:
                #    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

                opt.step()
                #if n_iter == 2500:
                    #print("changing learning rates")
                    #for param_group in opt.param_groups:
                        #if param_group["label"] == "remaining":
                         #   param_group["lr"] = 6e-2
                        
                        #if param_group["label"] == "encoder":
                        #    param_group["lr"] = 1e-3

        return best_loss, best_params

# %%
X, C = make_pinwheel_data(0.3, 0.05, 3, 512, 0.25)
X = torch.Tensor(X)
C = torch.Tensor(C)

plt.figure(figsize=(10, 10))
plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=C.numpy(), s=5)
plt.show()


# %%
def make_nf(shared_blocks, n_blocks, base_dist):
    blocks = [] + shared_blocks
    for _ in range(n_blocks):
        blocks += [StructuredAffineFlow(2), PReLUFlow(2), BatchNormFlow(2)]
    blocks += [StructuredAffineFlow(2)]
  #      blocks += [AffineLUFlow(2), PReLUFlow(2), BatchNormFlow(2)]
  #  blocks += [AffineLUFlow(2)]

    return NormalizingFlow( 
        *blocks,
        base_dist=base_dist,
    )


# %%
xdim = 2
hdim = 3
n_hidden = 3
n_classes = 3
shared_blocks = sum([
    [StructuredAffineFlow(2), PReLUFlow(2), BatchNormFlow(2)]
    for _ in range(3)
], [])
#shared_blocks = []
n_flow_blocks = 2


mixture = VariationalMixture(
    xdim=xdim,
    hdim=hdim,
    n_hidden=n_hidden,
    n_classes=n_classes,
    components=[make_nf(shared_blocks, n_flow_blocks, distrib.Normal(loc=torch.zeros(2), scale=torch.ones(2)))
                for _ in range(n_classes)],
)

# %%
count_parameters(mixture)

# %%
#mixture.to("cuda:0")
#X = X.to("cuda:0")

# %%
opt = optim.Adam([
    {   #encoder params
        "label": "encoder",
        "params": (v for k,v in mixture.named_parameters() if "encoder" in k),
        "lr": 1e-2
    },
    {   #remaining params
        "label": "remaining",
        "params": (v for k,v in mixture.named_parameters() if "encoder" not in k),
        "lr": 1e-2
    },
])

opt.zero_grad()


# %%
def train(n_epochs, bs, mixture):
    writer = SummaryWriter(f"./tensorboard_logs/{now_str()}")

    best_loss, best_params = mixture.fit(X,
        dataloader=DataLoader(X, batch_size=bs, shuffle=True, num_workers=0),
        n_epochs=n_epochs,
        #bs=bs,
        opt=opt,
        #temperature_schedule=lambda t: np.exp(-5e-4 * t),
        temperature_schedule=lambda t: 1,
        clip_grad=1e6,
        verbose=True,
        writer=writer)

    return best_loss, best_params


# %%
best_loss, best_params = train(
    n_epochs=1000,
    bs=512,
    mixture=mixture
)

# %%
mixture.load_state_dict(best_params)

# %%
mixture.to("cpu")
X = X.to("cpu")

# %%
x = np.linspace(-1, 1, 1000)
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
        X_k = component.sample(500)

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

# %%
import os
import random
import pickle
import itertools
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

from tqdm import trange, tqdm
from tensorboardX import SummaryWriter

from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import PReLUFlow, StructuredAffineFlow, AffineLUFlow, BatchNormFlow

# %%
from thesis_utils import now_str, count_parameters, figure2tensor, torch_onehot
from thesis_utils.extract_images import save_images_from_event

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

        return best_loss, best_params

# %%
X, C = make_pinwheel_data(0.3, 0.05, 3, 512, 0.25)
X = torch.Tensor(X)
C = torch.Tensor(C)

plt.figure(figsize=(10, 10))
plt.scatter(X[:,0].numpy(), X[:,1].numpy(), c=C.numpy(), s=5)
plt.show()


# %%
def make_nf(n_blocks, affine_class, base_dist):
    blocks = []
    for _ in range(n_blocks):
        blocks += [affine_class(2), PReLUFlow(2), BatchNormFlow(2)]
    blocks += [affine_class(2)]

    return NormalizingFlow( 
        *blocks,
        base_dist=base_dist,
    )


def sample_one_of_each(*iterables):
    combo = []
    for iterab in iterables:
        combo.append(
            random.choice(iterab)
        )

    return tuple(combo)


# %%
xdim = 2
n_classes = 3

#epochs = [100, 500, 1000, 5000]
#batch_sizes = [64, 128, 256, 512]
#n_hidden_encoder = [2, 3, 4, 5]
#hdim_encoder = [2, 3, 4, 5]
#lr_encoder = [1e-4, 1e-3, 1e-2]
#lr_remaining = [1e-4, 1e-3, 1e-2]
#n_flow_blocks = [5, 6, 7, 8]

epochs = [500]
batch_sizes = [512]
n_hidden_encoder = [3]
hdim_encoder = [3]
lr_encoder = [1e-2]
lr_remaining = [1e-2]
n_flow_blocks = [5]


#affine_classes = [StructuredAffineFlow, AffineLUFlow]
affine_class = StructuredAffineFlow

seen = set()

combo_str = lambda combo: "__"+"_".join(str(c) for c in combo)+"__"

for i in trange(1):
    combo = sample_one_of_each(
        epochs, batch_sizes, n_hidden_encoder,
        hdim_encoder, lr_encoder, lr_remaining,
        n_flow_blocks
    )

    while combo in seen:
        combo = sample_one_of_each(
            epochs, batch_sizes, n_hidden_encoder,
            hdim_encoder, lr_encoder, lr_remaining,
            n_flow_blocks
        )
    seen.add(combo)

    print(f"Current combo: {combo}")

    (n_epochs, bs, n_hidden, hdim, lr_enc, lr_rem, n_blocks) = combo

    mixture = VariationalMixture(
        xdim=xdim,
        hdim=hdim,
        n_hidden=n_hidden,
        n_classes=n_classes,
        components=[
            make_nf(n_blocks,
                    affine_class,
                    distrib.Normal(loc=torch.zeros(2), scale=torch.ones(2))
            ) for _ in range(n_classes)
        ]
    )

    opt = optim.Adam([
        {   #encoder params
            "label": "encoder",
            "params": (v for k,v in mixture.named_parameters() if "encoder" in k),
            "lr": lr_enc
        },
        {   #remaining params
            "label": "remaining",
            "params": (v for k,v in mixture.named_parameters() if "encoder" not in k),
            "lr": lr_rem
        },
    ])

    opt.zero_grad()
    writer = SummaryWriter(f"./pinwheel_tests/tensorboard_logs/{combo_str(combo)}")

    best_loss, best_params = mixture.fit(X,
        dataloader=DataLoader(X, batch_size=bs, shuffle=True, num_workers=0),
        n_epochs=n_epochs,
        opt=opt,
        temperature_schedule=lambda t: 1,
        clip_grad=1e6,
        verbose=True,
        writer=writer)

    fn = writer.file_writer.event_writer._ev_writer._file_name
    writer.close()

    with open(f"./pinwheel_tests/params/{combo_str(combo)}.pickle", "wb") as f:
        pickle.dump((best_params, combo), f)

    outdir = f"./pinwheel_tests/images/{combo_str(combo)}/"
    os.mkdir(outdir)
    save_images_from_event("./"+fn, "distributions", outdir)



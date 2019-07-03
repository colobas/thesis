# -*- coding: utf-8 -*-
# %%
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from vade import VaDE
from vade.utils import gaussianMLP

from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")

# %%
def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    # code from Johnson et. al. (2016)
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    np.random.seed(1)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    feats = 10 * np.einsum('ti,tij->tj', features, rotations)

    data = np.random.permutation(np.hstack([feats, labels[:, None]]))
    data[:, 0:2] = (data[:, 0:2] + np.array([20, 20]))/40

    return data[:, 0:2], data[:, 2].astype(np.int)


# %%
data, labels = make_pinwheel_data(0.3, 0.05, 3, 10000, 0.25)


if ipython is not None:
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:,0], data[:,1], c=labels, s=5)
    plt.show()

# %%
x_dim = 1
y_dim = 2
h_dim_enc = 50
h_dim_dec = 50
n_clusters = 3
n_hidden_enc = 1
n_hidden_dec = 1


if use_cuda:
    model = VaDE(
        n_clusters=n_clusters,
        y_dim=y_dim,
        x_dim=x_dim,
        encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=n_hidden_enc).cuda(),
        decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=n_hidden_dec).cuda(),
        gmm_pis=[0.3333, 0.3333, 0.3333],
        train_pis=False
    ).cuda()
    data = torch.Tensor(data).cuda()
    print("using cuda")
else:
    model = VaDE(
        n_clusters=n_clusters,
        y_dim=y_dim,
        x_dim=x_dim,
        encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=n_hidden_enc),
        decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=n_hidden_dec),
        gmm_pis=torch.Tensor([0.3333, 0.3333, 0.3333]),
        train_pis=False
    )
    data = torch.Tensor(data)

# %%
best_enc, best_dec = (
    model.pretrain(data,
                   bs=128,
                   n_epochs=200,
                   writer=SummaryWriter(f"/workspace/runs/{now_str()}"),
                   verbose=True,
                   opt=optim.Adam(model.decoder.get_pretrain_params()+
                                  model.encoder.get_pretrain_params(), lr=0.001)))

# %%
enc, _ = model.encoder(data)
Yhat, _ = model.decoder(enc)

Yhat = Yhat.detach().cpu()
enc = enc.detach().cpu()

# %%
if ipython is not None:
    plt.figure(figsize=(10, 10))
    plt.scatter(Yhat[:,0], Yhat[:,1], c=labels, s=5)
    plt.show()

# %% [markdown]
# with best_params

# %%
model.encoder.load_state_dict(best_enc)
model.decoder.load_state_dict(best_dec)

# %%
enc, _ = model.encoder(data)
Yhat, _ = model.decoder(enc)

Yhat = Yhat.detach().cpu()
enc = enc.detach().cpu()

# %%
if ipython is not None:
    plt.figure(figsize=(10, 10))
    plt.scatter(Yhat[:,0], Yhat[:,1], c=labels, s=5)
    plt.show()

# %%

# %%

# %%

# %%

# %%

# %%
best_losses, best_params = model.fit(
    data,
    n_epochs=500,
    bs=128,
    opt=optim.Adam(model.parameters(), lr=0.01),
    L=100,
    verbose=True,
    clip_grad=1e3,
    writer=SummaryWriter(f"/workspace/runs/{now_str()}")
)

final_params = model.state_dict()

# %%
for i, state_dict in enumerate(best_params):
    torch.save(state_dict, f"state_dict_{i}")

# %%
model.load_state_dict(best_params[0])

# %%
X, Z = model.predict(data.cuda())

# %%
X, Z = model.predict(data.cuda())

if use_cuda:
    data = data.cpu()
    Z = Z.cpu()

x_min = np.min(data.numpy()[:, 0])
x_max = np.max(data.numpy()[:, 0])
y_min = np.min(data.numpy()[:, 1])
y_max = np.max(data.numpy()[:, 1])

xx = np.linspace(x_min, x_max, 100)
yy = np.linspace(y_min, y_max, 100)

grid = np.array(list(itertools.product(xx, yy)))

if use_cuda:
    proba = model.predict_proba(torch.Tensor(grid).cuda()).cpu()
else:
    proba = model.predict_proba(torch.Tensor(grid))

label = proba.argmax(dim=1).detach().numpy()

plt.hexbin(grid[:, 0], grid[:, 1], C=proba.detach())

#plt.scatter(data.numpy()[:,0], data.numpy()[:,1], c=Z.numpy(), s=5)
plt.show()

# %%
import pandas as pd

# %%
z_samples = torch.zeros(1000)
x_samples = torch.zeros(1000, x_dim)
for i in range(1000):
    k = Categorical(probs=model.gmm_πs).sample().item()
    z_samples[i] = k
    x_samples[i] = (
            Normal(model.gmm_μs[k],
                   model.gmm_log_σs_sqr.exp().sqrt()[k])
    ).sample()

#pd.Series(x_samples).hist(bins=20)

# %%
X = model.predict_X(data.cuda()).detach().cpu()

# %%
pd.Series(X).hist(bins=20)

# %%
z_samples = torch.zeros(1000)
x_samples = torch.zeros(1000, x_dim)
for i in range(1000):
    k = Categorical(probs=model.gmm_πs).sample().item()
    z_samples[i] = k
    x_samples[i] = (
            Normal(model.gmm_μs[k],
                   model.gmm_log_σs_sqr.exp().sqrt()[k])
    ).sample()


plt.scatter(x_samples.numpy()[:, 0], x_samples.numpy()[:, 1], s=5, c=z_samples.numpy())
plt.show()

# %%
y_samples = model.decoder(x_samples.cuda())[0].detach().cpu()
plt.scatter(y_samples.numpy()[:, 0], y_samples.numpy()[:, 1], s=5, c=z_samples.numpy())
plt.show()

# %%

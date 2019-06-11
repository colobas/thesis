# -*- coding: utf-8 -*-
# %%
# %load_ext autoreload
# %autoreload 2

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
x_dim = 2
y_dim = 2
h_dim_enc = 2
h_dim_dec = 2
n_clusters = 3

# %%
true_model = VaDE(
    n_clusters=n_clusters,
    y_dim=y_dim,
    x_dim=x_dim,
    encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=1).cuda(),
    decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=1).cuda(),
)


# %%
def generate_samples(model, gen_y_samples=False):
    z_samples = torch.zeros(1000)
    x_samples = torch.zeros(1000, x_dim)
    for i in range(1000):
        k = Categorical(probs=model.gmm_πs).sample().item()
        z_samples[i] = k
        x_samples[i] = (
                Normal(model.gmm_μs[k],
                       model.gmm_log_σs_sqr.exp().sqrt()[k])
        ).sample()


    if gen_y_samples:
        samples = model.decoder(x_samples.cuda())[0].detach().cpu()
    else:
        samples = x_samples

    plt.scatter(samples[:, 0], samples[:, 1], s=5, c=z_samples)
    plt.show()
    return samples

# %%
x_samples = generate_samples(true_model)

# %%
y_samples = generate_samples(true_model, gen_y_samples=True)

# %%
data = y_samples

# %%
x_dim = 2
y_dim = 2
h_dim_enc = 2
h_dim_dec = 2
n_clusters = 3


if use_cuda:
    model = VaDE(
        n_clusters=n_clusters,
        y_dim=y_dim,
        x_dim=x_dim,
        encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=1).cuda(),
        decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=1).cuda(),
    ).cuda()
    data = torch.Tensor(data).cuda()
    print("using cuda")
else:
    model = VaDE(
        n_clusters=n_clusters,
        y_dim=y_dim,
        x_dim=x_dim,
        encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=1),
        decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=1),
    )
    data = torch.Tensor(data)

# %%
model.fit(
    data,
    n_epochs=2000,
    bs=100,
    opt=optim.Adam(model.parameters(), lr=0.005),
    L=10,
    verbose=True,
    clip_grad=1e3,
    writer=SummaryWriter(f"/workspace/runs/{now_str()}")
)

# %%

X, Z = model.predict(data)

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

plt.scatter(data.numpy()[:,0], data.numpy()[:,1], c=Z.numpy(), s=5)
plt.scatter(grid[:, 0], grid[:, 1], c=label, s=proba.max(dim=1)[0].detach().numpy()*20, alpha=0.6)
plt.show()

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

z_samples = torch.zeros(1000)
x_samples = torch.zeros(1000, x_dim)
for i in range(1000):
    k = Categorical(probs=model.gmm_πs).sample().item()
    z_samples[i] = k
    x_samples[i] = (
            Normal(model.gmm_μs[k],
                   model.gmm_log_σs_sqr.exp().sqrt()[k])
    ).sample()


y_samples = model.decoder(x_samples.cuda())[0].detach().cpu()
plt.scatter(y_samples.numpy()[:, 0], y_samples.numpy()[:, 1], s=5, c=z_samples.numpy())
plt.show()

# %%

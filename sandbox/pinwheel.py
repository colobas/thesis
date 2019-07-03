# -*- coding: utf-8 -*-

# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

from ts_deep_gmm import DeepGMM
from ts_deep_gmm.utils import gaussianMLP, categMLP

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

    return data[:, 0:2], data[:, 2].astype(np.int)


# %%
data, labels = make_pinwheel_data(0.3, 0.05, 3, 10000, 0.25)

if False:
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:,0], data[:,1], c=labels, s=5)
    plt.show()

# %%
x_dim = 2
y_dim = 2
h_dim_enc = 2
h_dim_dec = 2
n_clusters = 3


if use_cuda:
    model = DeepGMM(
        n_clusters=n_clusters,
        y_dim=y_dim,
        x_dim=x_dim,
        gaussian_encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=4).cuda(),
        cat_encoder=categMLP(x_dim, n_clusters, x_dim, n_hidden=4).cuda(),
        decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=2).cuda(),
    ).cuda()
    data = torch.Tensor(data).cuda()
    print("using cuda")
else:
    model = DeepGMM(
        n_clusters=n_clusters,
        y_dim=y_dim,
        x_dim=x_dim,
        gaussian_encoder=gaussianMLP(y_dim, x_dim, h_dim_enc, n_hidden=4),
        cat_encoder=categMLP(x_dim, n_clusters, x_dim, n_hidden=4),
        decoder=gaussianMLP(x_dim, y_dim, h_dim_dec, n_hidden=2),
    )
    data = torch.Tensor(data)

# %%


model.fit(
    data,
    temperature_schedule=None, # use default
    n_epochs=100,
    bs=1000,
    #opt=optim.Adam(model.parameters(), lr=0.001, momentum=0.0),
    opt=optim.RMSprop(model.parameters(), lr=0.001),
    n_samples=200,
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

print(f"""

    model.theta_gmm_mus: {model.θ_gmm_μs}


    model.theta_gmm_sigmas: {model.θ_gmm_Σ_diags}


    model.theta_gmm_pis: {model.θ_gmm_πs}

""")


z_samples = torch.zeros(1000)
x_samples = torch.zeros(1000, x_dim)
for i in range(1000):
    k = Categorical(probs=model.θ_gmm_πs).sample().item()
    z_samples[i] = k
    x_samples[i] = (
            MultivariateNormal(loc=model.θ_gmm_μs[k],
                       covariance_matrix=torch.diag_embed(model.θ_gmm_Σ_diags[k]))
    ).sample()


plt.scatter(x_samples.numpy()[:, 0], x_samples.numpy()[:, 1], s=5, c=z_samples.numpy())
plt.show()

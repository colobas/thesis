# -*- coding: utf-8 -*-
# %%
from IPython.display import HTML
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

# %%
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from tensorboardX import SummaryWriter
from tqdm import trange

use_cuda = torch.cuda.is_available()
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")

# %%
from ignite.engine import Events, Engine
from ignite.metrics import MeanSquaredError, RunningAverage


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
    
    data[:, 0:2] = (data[:, 0:2] + np.array([20, 20]))/40

    return data[:, 0:2], data[:, 2].astype(np.int)


# %%
class Autoencoder(nn.Module):
    def __init__(self, ydim, xdim, hdim, n_hidden=1):
        super().__init__()

        encoder_modules = (
          [nn.Linear(ydim, hdim), nn.ReLU()] + 
          sum([[nn.Linear(hdim, hdim), nn.ReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, xdim)]
        )
        
        decoder_modules = (
          [nn.Linear(xdim, hdim), nn.ReLU()] + 
          sum([[nn.Linear(hdim, hdim), nn.ReLU()] for i in range(n_hidden)], []) +
          [nn.Linear(hdim, ydim)]
        )
        
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, y):
        x = self.encoder(y)
        yhat = self.decoder(x)    
        
        return yhat



# %%
data, labels = make_pinwheel_data(0.3, 0.05, 3, 1000, 0.25)


if ipython is not None:
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:,0], data[:,1], c=labels, s=5)
    plt.show()

# %%
val_data, val_labels = make_pinwheel_data(0.3, 0.05, 3, 1000, 0.25, seed=1337)

# %%
model = Autoencoder(ydim=2, xdim=1, hdim=50, n_hidden=1)

sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
Y = torch.Tensor(data).cuda()
model = model.cuda()

# %%
_ = model.train()

# %%
yhats = []
xhats = []
best_loss = float("+inf")

n_epochs = 1000
bs = 128
#opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
opt = optim.Adam(model.parameters())

writer = SummaryWriter(f"/workspace/runs/{now_str()}")

for epoch in trange(n_epochs):
    batches = range((len(Y) - 1) // bs + 1)
    for i in batches:
        start_i = i * bs
        end_i = start_i + bs
        yb = Y[start_i:end_i]
        
        loss = F.mse_loss(model(yb), yb)
        loss.backward()
        
        if loss <= best_loss:
            best_model = model.state_dict()
            best_loss = loss
        
        writer.add_scalar("AE-test/loss", loss, epoch*bs + i)
        
        opt.step()
        opt.zero_grad()
        
        if (epoch*bs + i) % 100 == 0:
            with torch.no_grad():
                yhats.append(model(Y).cpu())
                xhats.append(model.encoder(Y).cpu())

# %%
model.load_state_dict(best_model)

# %%
dat, lab = data, labels
#dat, lab = val_data, val_labels


model.eval()
with torch.no_grad():
    yhat = model(torch.Tensor(dat).cuda()).cpu()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(yhat[:, 0], yhat[:, 1], c=lab, s=5)
    plt.show()

# %%
fig = plt.figure(figsize=(20, 10))
plt.xlim(-20, 20)
plt.ylim(0, 400)

def update(i):
    plt.gca().cla()
    plt.xlim(-20, 20)
    plt.ylim(0, 400)
    plt.hist(xhats[i].numpy().squeeze(), bins=100)
    
ani = FuncAnimation(fig, update, interval=500, frames=len(xhats), repeat=True)
HTML(ani.to_jshtml())

# %%
dat, lab = data, labels
#dat, lab = val_data, val_labels

# %%
model.eval()
with torch.no_grad():
    xhat = model.encoder(torch.Tensor(dat).cuda()).cpu()
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    pd.Series(xhat.numpy().squeeze()).hist(bins=100, ax=ax)

# %%
fig = plt.figure(figsize=(10, 10))
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)

def update(i):
    plt.gca().cla()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.scatter(yhats[i].numpy()[:,0],
                yhats[i].numpy()[:,1],
                c=labels,
                s=5)
    
ani = FuncAnimation(fig, update, interval=500, frames=len(xhats), repeat=True)
HTML(ani.to_jshtml())

# %%

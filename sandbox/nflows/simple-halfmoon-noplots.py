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
from normalizing_flows.flows import PReLUFlow, StructuredAffineFlow

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
flow = NormalizingFlow(
    dim=2, 
    blocks=([StructuredAffineFlow, PReLUFlow]*5 + [StructuredAffineFlow]),
    base_density=base_dist,
    flow_length=1
)

opt = optim.Adam(flow.parameters(), lr=2e-3)

# %%
count_parameters(flow)

# %%
n_epochs = 100000
bs = 512

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
        it = epoch*len(batches) + i + 1

        opt.zero_grad()
        loss = -flow.final_density.log_prob(xb).mean()

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
        opt.step()

# %%

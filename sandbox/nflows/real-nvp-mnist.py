# %%
import tensorflow as tf
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
from normalizing_flows.models import RealNVP

from thesis_utils import now_str, count_parameters, figure2tensor

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.view(60000, 784)
x_test = x_test.view(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

X = torch.tensor(x_train[y_train == 1])

# %%
base_dist = distrib.Normal(loc=torch.zeros(X.shape[1]),
                           scale=torch.ones(X.shape[1]))

# %%
flow = RealNVP(
    n_blocks=10,
    input_size=X.shape[1],
    hidden_size=X.shape[1]//4,
    n_hidden=3,
    base_dist=base_dist
)

opt = optim.Adam(flow.parameters(), lr=1e-4)

# %%
count_parameters(flow)

# %%
n_epochs = 100000
bs = 256

# %%
writer = SummaryWriter(f"./tensorboard_logs/{now_str()}")

best_loss = torch.Tensor([float("+inf")])

#attempts = 0

for epoch in trange(n_epochs):
    batches = range((len(X) - 1) // bs + 1)
    for i in batches:
        start_i = i * bs
        end_i = start_i + bs
        xb = X[start_i:end_i]
        it = epoch*len(batches) + i + 1

        opt.zero_grad()
        loss = -flow.log_prob(xb).mean()

        #if loss <= 0:
        #    if attempts < 100:
        #        attempts += 1
        #        continue
        #    else:
        #        print("Loss has diverged, halting train and not backpropagating")
        #        break

        if loss <= best_loss:
            best_loss = loss
            best_flow_params = flow.state_dict()

        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            writer.add_scalar("loss", loss, it)
        if epoch % 200 == 0:
            with torch.no_grad():
                Xhat = flow.sample(1000)
                f = plt.figure(figsize=(10, 10))
                plt.xlim(-30, 30)
                plt.ylim(-20, 20)
                plt.title(f"{it} iterations")
                plt.scatter(Xhat[:, 0], Xhat[:, 1], s=5, c="red", alpha=0.5)
                plt.scatter(X[:, 0], X[:, 1], s=5, c="blue", alpha=0.5)
                writer.add_image("distributions", figure2tensor(f), it)
                plt.close(f)

# %%

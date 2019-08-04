# %%
from IPython import get_ipython
ipython = get_ipython()

if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

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
from normalizing_flows import NormalizingFlow
from normalizing_flows.flows import AffineFlow, PReLUFlow, StructuredAffineFlow, AffineLUFlow

# %%
import matplotlib.pyplot as plt
import datetime

from copy import deepcopy

from tqdm import trange
from tensorboardX import SummaryWriter

# %%
now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def gen_samples(batch_size=512):
    x2_dist = distrib.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample((batch_size,))

    x1 = distrib.Normal(loc=.25 * (x2_samples.pow(2)),
                  scale=torch.ones((batch_size,)))

    x1_samples = x1.sample()
    return torch.stack([x1_samples, x2_samples]).t()

x_samples = gen_samples(512)

# %%
plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5)
plt.show()

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
plt.scatter(X0[:, 0], X0[:, 1], s=5, c=colors)

# %%
flow = NormalizingFlow(
    dim=2, 
    blocks=([StructuredAffineFlow, PReLUFlow]*5 + [StructuredAffineFlow]),
    base_density=base_dist,
    flow_length=1
)

opt = optim.Adam(flow.parameters(), lr=1e-3)

# %%
count_parameters(flow)

# %%
n_epochs = 10000
bs = 512

# %%
writer = SummaryWriter(f"/workspace/sandbox/tensorboard_logs/{now_str()}")

best_loss = torch.Tensor([float("+inf")])

attempts = 0


#for epoch in trange(n_epochs):
#    batches = range((len(x_samples) - 1) // bs + 1)
#    for i in batches:
#        start_i = i * bs
#        end_i = start_i + bs
#        xb = x_samples[start_i:end_i]
#        it = epoch*len(x_samples) + start_i

#        opt.zero_grad()
#        loss = -flow.final_density.log_prob(xb).mean()

for it in trange(int(1e5)):
    opt.zero_grad()
    loss = -flow.final_density.log_prob(x_samples).mean()

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
    
    if it % 50 == 0:
        writer.add_scalar("loss", loss, it)

    if it % 5000 == 0:
        with torch.no_grad():
            xhat_samples = flow.final_density.sample((1000, ))
            plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
            plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5, c="blue")
            #plt.xlim(0, 60)
            #plt.ylim(-15, 15)
            plt.show()
            print(loss)

    #    if it % 100 == 0:
    #        f = plt.figure(figsize=(20, 20))
    #        xhat_samples = flow.final_density.sample((1000, ))
    #        plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
    #        plt.xlim(-5, 40)
    #        plt.ylim(-15, 15)
    #        plt.title(f"{loss.detach().numpy()}")
    #        plt.savefig(f"to_gif/it_{it}.png")
    #        plt.close()
    opt.step()

# %%
loss

# %%
flow = best_flow

# %%
xhat_samples = flow.final_density.sample((1000, ))
plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5, c="blue")
#plt.xlim(0, 60)
#plt.ylim(-15, 15)
plt.show()

# %%
n_flows = len(flow.bijectors)

f, axs = plt.subplots(
    n_flows + 1,
    1,
    figsize=(10, n_flows*12),
    #sharex=True,
    #sharey=True
)

axs[0].scatter(X0[:, 0], X0[:, 1], s=5, c=colors)

cur_x = X0

for ax, bij in zip(axs[1:], flow.bijectors[:n_flows+1]):
    cur_x = bij(torch.Tensor(cur_x)).detach().numpy()
    ax.scatter(cur_x[:, 0], cur_x[:, 1], s=5, c=colors)
    
plt.show()

# %%
x = np.linspace(-4, 4, 1000)
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])


# %%
def get_density(cur_z, prev_density, flow):
    density = prev_density.squeeze() / np.exp(flow.log_abs_det_jacobian(torch.Tensor(cur_z), None).detach().squeeze())
    return torch.Tensor(density)


# %%
def get_meshes(cur_z, density, grid_side=1000, dim=2):
    mesh = cur_z.reshape([grid_side, grid_side, dim]).transpose(2, 0, 1)
    xx = mesh[0]
    yy = mesh[1]
    zz = density.numpy().reshape([grid_side, grid_side])
    
    return xx, yy, zz


# %%
f, axs = plt.subplots(
    n_flows + 1,
    1,
    figsize=(10, n_flows*12),
    #sharex=True,
    #sharey=True
)

cur_z = z
prev_density = (flow.base_density
                    .log_prob(torch.Tensor(cur_z))
                    .sum(dim=1)
                    .exp().detach())

xx, yy, zz = get_meshes(cur_z, prev_density)

axs[0].contourf(xx, yy, zz, 50, cmap='rainbow')

for ax, bij in zip(axs[1:], flow.bijectors[:n_flows+1]):
    cur_z = bij(torch.Tensor(cur_z)).detach().numpy()
    
    prev_density = get_density(cur_z, prev_density, bij)
    xx, yy, zz = get_meshes(cur_z, prev_density)
    ax.contourf(xx, yy, zz, 50, cmap='rainbow')
    
plt.show()

# %%
list(flow.named_parameters())


# %% [markdown]
# ___

# %% [markdown]
# Trying to specify the params obtained in the tensorflow version, to see if the output corresponds

# %% [markdown]
# It's last to first, and prelu_0 is discarded

# %%
def _fill_triangular(x, upper=False):
    """Numpy implementation of `fill_triangular`."""
    x = np.asarray(x)
    # Formula derived by solving for n: m = n(n+1)/2.
    m = np.int32(x.shape[-1])
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
        raise ValueError("Invalid shape.")
    n = np.int32(n)
    # We can't do: `x[..., -(n**2-m):]` because this doesn't correctly handle
    # `m == n == 1`. Hence, we do absolute indexing.
    x_tail = x[..., (m - (n * n - m)):]
    y = np.concatenate(
        [x, x_tail[..., ::-1]] if upper else [x_tail, x[..., ::-1]],
        axis=-1)
    y = y.reshape(np.concatenate([
        np.int32(x.shape[:-1]),
        np.int32([n, n]),
    ], axis=0))
    return np.triu(y) if upper else np.tril(y)


# %%
Ls = [ 
    [ 0.19130541, -0.66285866,  0.635223  ],
    [-0.21493493, -1.1085328,  -0.09023035],
    [-0.97976303,  0.51778686, -0.58213425],
    [-0.5339833,  -0.22712007,  0.37868443],
    [ 0.9055861,  -0.32401618,  0.01458302],
    [-0.7340774,   0.17165975, -0.17290545],
]
Ls = list(map(torch.Tensor, Ls))


Vs = [ 
    [[-1.9192145,  -2.8893428 ],
     [ 0.61188304,  0.7053445 ]],

    [[-1.6655174e-36,  9.8027325e-01],
     [ 2.7752662e-36,  1.1918546e+00]],

    [[-0.2731575,   0.85080206],
     [ 0.91976583,  0.715398  ]],

    [[-0.78363204, -0.8567729 ],
     [ 0.5309808,  -0.87246627]],

    [[-0.7394123,   1.4964929 ],
     [ 1.9175456,   0.08542425]],

    [[2.0492299,  0.13704464],
     [0.01747171, 0.35763052]]
]
Vs = list(map(torch.Tensor, Vs))

# %%
shifts = [
 np.array([-2.7384284,  4.6790996]),
 np.array([1.4246637, 2.5997574]),
 np.array([-1.2736514 , -0.14639635]),
 np.array([-3.702082  ,  0.40250754]),
 np.array([-2.8402073, -3.1306276]),
 np.array([-4.177382,  7.369329])]

# %%
alphas = [0.8555014, 0.712254, 0.4994685, 0.37195796, 0.5392429]

# %%
flow = NormalizingFlow(
    dim=2, 
    blocks=([StructuredAffineFlow, PReLUFlow]*5 + [StructuredAffineFlow]),
    base_density=base_dist,
    flow_length=1
)

# %%
for i in range(len(flow.bijectors)):
    if i % 2 == 0:
        flow.bijectors[i].L.data = torch.Tensor(_fill_triangular(Ls[i//2]))
        flow.bijectors[i].V.data = Vs[i//2]
        flow.bijectors[i].shift.data = torch.Tensor(shifts[i//2]).squeeze()

# %%
for i, alpha in enumerate(alphas):
    flow.bijectors[i*2 + 1].alpha.data = torch.Tensor([alpha]).squeeze()

# %%
for i, bij in enumerate(flow.bijectors):
    if i % 2 == 0:
        print(bij.weights)

# %%
xhat_samples = flow.final_density.sample((1000, ))
plt.scatter(xhat_samples[:, 0], xhat_samples[:, 1], s=5, c="red")
plt.scatter(x_samples[:, 0], x_samples[:, 1], s=5, c="blue")
#plt.xlim(0, 60)
#plt.ylim(-15, 15)
plt.show()

# %%
if hasattr(torch, "triangular_solve"):
    triangular_solve = torch.triangular_solve
elif hasattr(torch, "trtrs"):
    triangular_solve = torch.trtrs

def log_abs_det_jacobian(self, z, z_next):
    """
    roughly following tensorflow's LinearOperatorLowRankUpdate logic and
    naming, but U and V are the same, and D is the identity
    """
    log_det_L = self.L.diag().prod().log()
    print(f"log_det_L {log_det_L}")

    linv_u = triangular_solve(self.V, self.L * self.tril_mask, upper=False)[0]
    vt_linv_u = self.V.t() @ linv_u
    capacitance = vt_linv_u + self.I

    return ((log_det_L + torch.slogdet(capacitance)[1]) # we would sum the log determinant of the diagonal component, but it is 0, because the diagnoal is I
                # the jacobian is constant, so we just repeat it for the
                # whole batch
                .repeat(z.size(0), 1)
                .squeeze())

def make_cap(self):
    linv_u = triangular_solve(self.V, self.L * self.tril_mask, upper=False)[0]
    vt_linv_u = self.V.t() @ linv_u
    capacitance = vt_linv_u + self.I
    return capacitance

def make_vtlinvu(self):
    linv_u = triangular_solve(self.V, self.L * self.tril_mask, upper=False)[0]
    vt_linv_u = self.V.t() @ linv_u
    
    return vt_linv_u


# %%
for i, bij in enumerate(flow.bijectors):
    if i % 2 == 0:
        print(f"\nvt_linvu_{i}")
        print(make_cap(bij).detach())

# %%
log_abs_det_jacobian(flow.bijectors[-1], torch.Tensor([[1, 1]]), None)

# %%
flow.bijectors[-1].log_abs_det_jacobian(torch.Tensor([[1, 1]]), None)

# %%
x = torch.Tensor(X0)

# %%
for i, bij in enumerate(flow.bijectors):
    if i % 2 == 0:
        print(bij.log_abs_det_jacobian(torch.Tensor([[1, 1]]), None))

# %%
-flow.final_density.log_prob(torch.Tensor([[1, 1]]))

# %%
flow.transforms.log_abs_det_jacobian(torch.Tensor([[1, 1]]), None)

# %%
flow.final_density.log_prob(x_samples).mean()

# %%

# %%

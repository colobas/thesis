# %% [markdown]
# This short notebook is just a sanity check, to confirm that using the `torch.stack`ed GMM distribution params allows me to compute log probs for the K distributions at once

# %%
import torch
import matplotlib.pyplot as plt

from torch.distributions import Normal

# %% [markdown]
# First checking I'm sure the distribution represents what it is supposed to (K independent Gaussians, 3 in this case)

# %%
mus = torch.stack([
    torch.Tensor([-10, -10]),
    torch.Tensor([0, 0]),
    torch.Tensor([10, 10]),
])

sigs = torch.stack([
    torch.Tensor([0.1, 0.1]),
    torch.Tensor([1, 1]),
    torch.Tensor([2, 2]),
])
# %%
gmm = Normal(mus, sigs)
gmm_samples = gmm.sample((1000, ))

plt.scatter(gmm_samples[:, 0, 0], gmm_samples[:, 0, 1], s=5)
plt.scatter(gmm_samples[:, 1, 0], gmm_samples[:, 1, 1], s=5)
plt.scatter(gmm_samples[:, 2, 0], gmm_samples[:, 2, 1], s=5)

plt.show()

# %% [markdown]
# Now I'll generate some samples in the shape of (batch_size, x_dim).

# %%
x_samples = Normal(torch.Tensor([0, 1]), torch.Tensor([1, 0.3])).sample((1000,))
print(x_samples.shape)

# %% [markdown]
# The shape of `gmm_samples` is (n_samples, K, x_dim). I suspect that to use `gmm.log_prob` on `x_samples` I will have to `.unsqueeze(1)`, so that I'm calling `log_prob` on a tensor with shape (n_samples, 1, x_dim)

# %%
log_probs = gmm.log_prob(x_samples.unsqueeze(1))

print(log_probs.shape)

# %% [markdown]
# Also, I'll need to call `.sum` on the result of `.log_probs` because it returns a per-dimension log prob:

# %%
log_probs = gmm.log_prob(x_samples.unsqueeze(1)).sum(dim=2)

print(log_probs.shape)

# %% [markdown]
# It computes. I'm getting a per-mixture component log-prob. Now I'll make sure the result is consistent to what I'd get if I did it manually:

# %%
for i, (mu, sig) in enumerate(zip(mus, sigs)):
    assert torch.all(log_probs[:, i] == Normal(mu, sig).log_prob(x_samples).sum(dim=1))

# %% [markdown]
# Nice.

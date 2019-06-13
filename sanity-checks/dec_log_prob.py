# %% [markdown]
# This short notebook is just a sanity check, to make sure I can correctly do a batch_gaussian_log_probs 

# %%
import torch
import matplotlib.pyplot as plt

from torch.distributions import Normal

# %%
dec_mus = torch.randn((1000, 5, 2))
dec_sigmas = torch.rand((1000, 5, 2))

# %%
y_observations = torch.randn((1000, 2))

# %% [markdown]
# Now I want to compute the 5 log_probs for each observation's corresponding group of 5 parameters

# %%
log_probs = Normal(dec_mus, dec_sigmas).log_prob(y_observations.unsqueeze(1))

print(log_probs.shape)

# %% [markdown]
# I need to sum, because this gives-me a per dimension log_prob:

# %%
log_probs = Normal(dec_mus, dec_sigmas).log_prob(y_observations.unsqueeze(1)).sum(dim=2)

print(log_probs.shape)

# %% [markdown]
# Now making sure the result is consistent with what we get manually:

# %%
manual_log_probs = torch.zeros(1000, 5)
for i, obs in enumerate(y_observations):
    for j in range(5):
        manual_log_probs[i, j] = Normal(dec_mus[i,j], dec_sigmas[i,j]).log_prob(obs).sum()

assert torch.all(log_probs == manual_log_probs)

# %%

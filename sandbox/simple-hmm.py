# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

# %%
startprob = np.array([0.6, 0.3, 0.1, 0.0])

# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.9, 0.07, 0.0, 0.03],
                     [0.066, 0.9, 0.034, 0.0],
                     [0.0, 0.066, 0.9, 0.034],
                     [0.05, 0.0, 0.05, 0.9]])

# The means of each component
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])

# The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1))

# Build an HMM instance and set parameters
model = hmm.GaussianHMM(n_components=4, covariance_type="full")

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

# %%
# Generate samples
X, Z = model.sample(5000)

# %%
f, axs = plt.subplots(X.shape[1], 1, figsize=(20, 10))

for i, ax in enumerate(axs):
    ax.scatter(list(range(len(X))), X[:,i], s=5, c=Z)

plt.show()

# %%

# %%

# %%

# %% [markdown]
# Now trying the PO-HMM learning approach

# %%
n_components=4

# %%
gmm = GaussianMixture(n_components=n_components)

# %%
gmm.fit(X)

# %%
Zhat = gmm.predict(X)

# %%
pd.Series(list(zip(Z.tolist(), Zhat.tolist()))).value_counts()

# %% [markdown]
# The GMM was perfectly fitted to the original gaussians. Can I now estimate the transition matrix?

# %% [markdown]
# The moment matrix:
#
# ![image.png](attachment:image.png)

# %%
Fs = []

for i in range(n_components):
    Fs.append(multivariate_normal(mean=gmm.means_[i], cov=gmm.covariances_[j]))
    
def m_ij(Y, i, j, Fs):

    res = np.sum(
        Fs[i].pdf(Y[0:-1]) * Fs[j].pdf(Y[1:])
        #gauss_log_prob(Y[0:-1], gmm.means_[i], gmm.precisions_cholesky_[i]) + 
        #gauss_log_prob(Y[1:], gmm.means_[j], gmm.precisions_cholesky_[j])
    )
    
    return (1/(len(Y) - 1))*res
    


# %%
Mhat = np.zeros((n_components, n_components))
for i in range(n_components):
    for j in range(n_components):
        Mhat[i, j] = m_ij(X, i, j, Fs)

# %%

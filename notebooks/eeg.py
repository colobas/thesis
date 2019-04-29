# -*- coding: utf-8 -*-
# %%
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import arff
from sklearn.mixture import GaussianMixture
from umap import UMAP

data = arff.loadarff("../data/EEG Eye State.arff")

# %%
df = pd.DataFrame(data[0])
df.head()
# %%
df["eyeDetection"] = df["eyeDetection"].astype(int)

# %%
X_cols = df.columns.tolist()[:-1]

df["eyeDetection"].plot()
plt.show()

# %%
emb = UMAP(n_components=2).fit_transform(df[X_cols])

# %%
f = plt.figure(figsize=(20, 10))

plt.scatter(emb[:,0], emb[:, 1], c=df["eyeDetection"], s=1, alpha=0.6)
plt.show()

# %%

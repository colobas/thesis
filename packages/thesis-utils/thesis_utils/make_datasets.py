import torch
import numpy as np

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

    data[:, 0:2] = data[:, 0:2]/20

    return data[:, 0:2], data[:, 2].astype(np.int)

def gen_circle(n_samples=512, r=0.8, eps=0.05):
    rs = np.random.randn(n_samples) * eps + r
    thetas = np.random.rand(n_samples) * 2 * np.pi

    x = rs * np.cos(thetas)
    y = rs * np.sin(thetas)

    X = np.vstack((x, y)).T

    return torch.tensor(X).float()

def make_circles_data(r1, r2, n_samples=512, eps=0.005):
    inner = gen_circle(n_samples=n_samples, r=r1, eps=eps)
    labels_inner = np.ones((n_samples, 1))

    outer = gen_circle(n_samples=n_samples, r=r2, eps=eps)
    labels_outer = np.zeros((n_samples, 1))

    labels = np.vstack([labels_inner, labels_outer])
    feats = np.vstack([inner, outer])

    data = np.random.permutation(np.hstack([feats, labels]))
    return data[:, 0:2], data[:, 2].astype(np.int)

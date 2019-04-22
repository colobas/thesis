import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from torch.autograd import gradcheck, Variable

#from pyro.distributions.relaxed_straight_through import RelaxedOneHotCategoricalStraightThrough
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_diag, _batch_mv
from torch.distributions import RelaxedOneHotCategorical, MultivariateNormal, Uniform

from tqdm import trange

from ts_deep_gmm.utils.bad_grad_viz import register_hooks

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def mv_gaussian_log_prob(value, μ, Σ_diags, event_shape):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    FOR DIAGONAL COV MATRICES ONLY
    """
    diff = value - μ
    M = ((diff**2) / Σ_diags).sum(-1)
    log_det = Σ_diags.log().sum(-1)

    return -0.5 * (event_shape * math.log(2 * math.pi) + M + log_det)


# auxiliary functions for the EM part
sel = lambda z_samples, k: z_samples.transpose(0, 1)[k].unsqueeze(-1)
norm = lambda z_samples, k: sel(z_samples, k).sum()

class DeepGMM(nn.Module):
    def __init__(self, n_clusters, y_dim, x_dim, gaussian_encoder, cat_encoder,
                 decoder, reg=0.01):
        """
        Note that Y are the observations (the data)
        and X are the latent representations
        """
        super(DeepGMM, self).__init__()

        self.n_clusters = n_clusters
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.reg = reg

        ################# Recognition parameters #################
        self.gaussian_encoder = gaussian_encoder # observ. -> gaussian latent (params)
        self.cat_encoder = cat_encoder # latent gauss -> latent categ. logits

        ##########################################################

        ################## Generative parameters #################
        self.decoder = decoder # latent var -> observ (params)

        self.θ_gmm_μs = torch.stack([
            torch.randn(x_dim)
            for _ in range(n_clusters)
        ])

        self.θ_gmm_Σ_diags = torch.stack([
            torch.randn(x_dim)
            for _ in range(n_clusters)
        ])

        # store in logits form because it's unconstrained,
        # then use log_softmax to convert to log probabilities
        self.θ_gmm_πs = torch.rand(n_clusters)
        ##########################################################

    def predict(self, Y):
        """
        Given data points Y, infer their latent X and Z
        """
        X, _ = self.gaussian_encoder(Y)
        z_log_probs = self.cat_encoder(X)

        Z = z_log_probs.argmax(dim=1)

        return X, Z

    def predict_proba(self, Y, log_probs=False):
        """
        Given data points Y, give the probability of belonging to each class
        """
        X, _ = self.gaussian_encoder(Y)
        z_log_probs = self.cat_encoder(X)

        if log_probs:
            return z_log_probs

        return z_log_probs.exp()

    def _fit_step(self, Y, temperature, n_samples=10):
        ϕ_enc_μ, ϕ_enc_Σ_diags = self.gaussian_encoder(Y)

        N = Y.shape[0]

        x_samples = (
            MultivariateNormal(loc=ϕ_enc_μ,
                               covariance_matrix=torch.diag_embed(ϕ_enc_Σ_diags))
                .rsample((n_samples,)).transpose(0, 1) #TODO: check this
        ).flatten(0, 1)

        z_log_probs = self.cat_encoder(x_samples)

        # pseudo e-step
        z_samples = RelaxedOneHotCategorical(temperature, probs=z_log_probs.exp()).rsample()

        # pseudo m-step
        for k in range(self.n_clusters):
            self.θ_gmm_μs[k] = ((sel(z_samples, k) * x_samples).sum(dim=0) / norm(z_samples, k)).detach()

            self.θ_gmm_Σ_diags[k] = (
                (sel(z_samples, k) * (x_samples ** 2)).sum(dim=0) / norm(z_samples, k) +
                -2 * (self.θ_gmm_μs[k] * (sel(z_samples, k) * x_samples)).sum(dim=0) / norm(z_samples, k) +
                self.θ_gmm_μs[k] ** 2 +
                self.reg
            ).sqrt().detach()

            self.θ_gmm_πs[k] = z_samples[k].mean().detach()

        _θ_gmm_μ = (z_samples.unsqueeze(-1) * self.θ_gmm_μs).sum(dim=1)
        _θ_gmm_Σ = (z_samples.unsqueeze(-1) * self.θ_gmm_Σ_diags).sum(dim=1)

        #_θ_gmm_μ = torch.einsum("bsk,kd->bsd", z_samples, self.θ_gmm_μs)
        #_θ_gmm_Σ = torch.einsum("bsk,kdf->bsd", z_samples, self.θ_gmm_Σs)

        μ_y, Σ_y = self.decoder(x_samples)

        # decoder NN part of the loss: log p(y | x, z)
        loss1 = mv_gaussian_log_prob(Y.unsqueeze(1).repeat(1, n_samples, 1).flatten(0, 1),
                                     μ_y,
                                     Σ_y,
                                     self.y_dim)

        # encoder gauss NN part of the loss: log q(x | y)
        loss2 = -mv_gaussian_log_prob(x_samples,
                                     ϕ_enc_μ.unsqueeze(1).repeat(1, n_samples, 1).flatten(0, 1),
                                     ϕ_enc_Σ_diags.unsqueeze(1).repeat(1, n_samples, 1).flatten(0, 1),
                                     self.x_dim)

        # encoder categ NN part of the loss: -log q (z | x)
        loss3 = -(z_samples * z_log_probs).sum(dim=1)

        # decoder GMM part of the loss: log p(x | z) + log p(z)
        loss4 = (mv_gaussian_log_prob(x_samples,
                                       _θ_gmm_μ,
                                       _θ_gmm_Σ,
                                       self.x_dim) +
                 (z_samples * self.θ_gmm_πs.log()).sum(dim=1))

        # what's inside parens is the ELBO, and we want to maximize it,
        # hence minimize its reciprocal
        res = -((loss1 + loss2 + loss3 + loss4).sum()/n_samples)

        return res



    def fit(self, Y, temperature_schedule=None, n_epochs=1, bs=100, opt=None,
            n_samples=10, clip_grad=None, verbose=False, callback=None):

        # TODO: implement default values

        if temperature_schedule is None:
            # default temperature schedule, adapted from https://arxiv.org/pdf/1611.01144.pdf
            # TODO: r and M should be hyperparams
            r = 1e-2
            M = bs//10
            temperature = 1
            def temperature_schedule(epoch, i, N):
                t = (epoch * N + i)
                if t % M == 0:
                    return max(0.5, math.exp(-r * t))
                else:
                    return temperature

        if opt is None:
            # TODO: better defaults
            opt = optim.SGD(self.parameters(), lr=0.01, momentum=0.0)

        losses = []

        if verbose:
            epochs = trange(n_epochs, desc="epoch")
        else:
            epochs = range(n_epochs)

        for epoch in epochs:
            if verbose:
                batches = trange((len(Y) - 1) // bs + 1, desc="batch")
            else:
                batches = range((len(Y) - 1) // bs + 1)
            for i in batches:
                start_i = i * bs
                end_i = start_i + bs
                yb = Y[start_i:end_i]

                temperature = temperature_schedule(epoch, start_i, len(Y))

                loss = self._fit_step(yb, temperature, n_samples)
                losses.append((epoch, start_i, loss.item()))
                loss.backward()

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

                opt.step()
                opt.zero_grad()
                if callback is not None:
                    callback

        return losses

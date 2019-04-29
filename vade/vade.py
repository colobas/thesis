# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from torch.autograd import gradcheck, Variable

from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_diag, _batch_mv
from torch.distributions import Normal, Uniform, OneHotCategorical
from tensorboardX import SummaryWriter

from tqdm import trange

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

class VaDE(nn.Module):
    def __init__(self, n_clusters, y_dim, x_dim, encoder,
                 decoder, reg=0.01):
        """
        Note that Y are the observations (the data)
        and X are the latent representations
        """
        super(VaDE, self).__init__()

        self.n_clusters = n_clusters
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.reg = reg

        ################# Recognition parameters #################
        self.encoder = encoder # observ. -> gaussian latent (params)


        ################## Generative parameters #################
        self.decoder = decoder # latent var -> observ (params)

        self.gmm_μs = torch.Parameter(torch.stack([
            torch.randn(x_dim, device=_DEVICE)
            for _ in range(n_clusters)
        ]))

        self.gmm_σs = torch.Parameter(torch.stack([
            torch.randn(x_dim, device=_DEVICE)
            for _ in range(n_clusters)
        ]))

        self.gmm_πs_logits = torch.Parameter(torch.rand(n_clusters, device=_DEVICE))

    def predict(self, Y):
        pass

    def _fit_step(self, Y, temperature, n_samples=10):
        enc_μs, enc_σs = self.gaussian_encoder(Y)

        N = Y.shape[0]
        x_samples = Normal(enc_μs, enc_σs).rsample((n_samples,)).flatten(0, 1)

        dec_μs, dec_σs = self.decoder(x_samples)

        aux = (F.log_softmax(self.gmm_πs_logits, dim=0) +
               Normal(self.gmm_μs, self.gmm_σs**2).log_prob(x_samples))

        λ_c = F.softmax(aux, dim=1).view(N, n_samples, self.x_dim)


        return 

    def fit(self, Y, temperature_schedule=None, n_epochs=1, bs=100, opt=None,
            n_samples=10, clip_grad=None, verbose=False, writer=None):

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

                loss, loss1, loss2, loss3, loss4 = self._fit_step(yb, temperature, n_samples)

                # if we're writing to tensorboard
                if writer is not None:
                    n_iter = epoch*len(Y) + start_i
                    if n_iter % 10 == 0:
                        writer.add_scalar('losses/loss1', loss1, n_iter)
                        writer.add_scalar('losses/loss2', loss2, n_iter)
                        writer.add_scalar('losses/loss3', loss3, n_iter)
                        writer.add_scalar('losses/loss4', loss4, n_iter)
                        writer.add_scalar('losses/total_loss', loss, n_iter)

                loss.backward()

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

                opt.step()
                opt.zero_grad()

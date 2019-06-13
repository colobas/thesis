# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from torch.autograd import gradcheck, Variable

from torch.distributions import Normal
from tensorboardX import SummaryWriter

from tqdm import trange

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class VaDE(nn.Module):
    def __init__(self, n_clusters, y_dim, x_dim, encoder,
                 decoder, gmm_pis=None, train_pis=True):
        """
        Note that Y are the observations (the data)
        and X are the latent representations
        """
        super(VaDE, self).__init__()

        self.n_clusters = n_clusters
        self.y_dim = y_dim
        self.x_dim = x_dim

        ################# Recognition parameters #################
        self.encoder = encoder # observ. -> gaussian latent (params)


        ################## Generative parameters #################
        self.decoder = decoder # latent var -> observ (params)

        self.gmm_μs = nn.Parameter(torch.stack([
            torch.randn(x_dim, device=_DEVICE)*(torch.Tensor(1).uniform_(-100, 100).to(_DEVICE))
            for _ in range(n_clusters)
        ]))

        self.gmm_log_σs_sqr = nn.Parameter(torch.stack([
            torch.randn(x_dim, device=_DEVICE)
            for _ in range(n_clusters)
        ]))

        if train_pis:
            if gmm_pis is None:
                self.gmm_πs = nn.Parameter(F.softmax(torch.randn(n_clusters, device=_DEVICE), dim=0))
            else:
                self.gmm_πs = nn.Parameter(F.softmax(torch.Tensor(gmm_pis).to(_DEVICE), dim=0))
        else:
            if gmm_pis is None:
                self.gmm_πs = F.softmax(torch.ones(n_clusters, device=_DEVICE), dim=0)
            else:
                self.gmm_πs = F.softmax(torch.Tensor(gmm_pis).to(_DEVICE), dim=0)

    def predict_X(self, Y, return_log_sigma_sqr=False):
        enc_μs, enc_log_σs_sqr = self.encoder(Y)

        if return_log_sigma_sqr:
            return enc_μs, enc_log_σs_sqr

        return enc_μs

    def _estimate_lambda_z(self, Y, enc_μs, enc_log_σs_sqr, n_samples=10):
        # estimate using samples of x
        N = len(Y)

        x_samples = Normal(enc_μs, enc_log_σs_sqr.exp().sqrt()).sample((n_samples,)).flatten(0, 1)
        aux = (
          (self.gmm_πs + 1e-6).log() +
          Normal(self.gmm_μs, self.gmm_log_σs_sqr.exp().sqrt())
            .log_prob(x_samples.unsqueeze(1))
            .sum(dim=2)
        )

        λ_z = (
          # exp-norm the log probs, to get probs -> softmax
          F.softmax(aux, dim=1)
          # reshape in L-samples-per-N-observations shape
          .view(N, n_samples, self.n_clusters)
          # compute the expectation per Y observation
          .mean(dim=1)
        )

        return λ_z

    def predict(self, Y, n_samples=10):
        enc_μs, enc_log_σs_sqr = self.encoder(Y)

        if n_samples > 0:
            λ_z = self._estimate_lambda_z(Y, enc_μs, enc_log_σs_sqr, n_samples)
        else:
            raise NotImplemented
            # estimate using enc_μs
            # TODO: check this
            #aux = (F.log_softmax(self.gmm_πs_logits, dim=0) +
            #       Normal(self.gmm_μs, self.gmm_log_σs_sqr.exp().sqrt()).log_prob(enc_μs))
            #λ_z = F.softmax(aux, dim=1)

        return enc_μs, λ_z.max(dim=1)[1]

    def predict_proba(self, Y, n_samples=100):
        enc_μs, enc_log_σs_sqr = self.encoder(Y)

        if n_samples > 0:
            return self._estimate_lambda_z(Y, enc_μs, enc_log_σs_sqr, n_samples)
        else:
            raise NotImplemented
            # estimate using enc_μs
            # TODO: check this
            #aux = (F.log_softmax(self.gmm_πs_logits, dim=0) +
            #       Normal(self.gmm_μs, self.gmm_log_σs_sqr.exp().sqrt()).log_prob(enc_μs))
            #return F.softmax(aux, dim=1)

    def pretrain(self, Y, bs, n_epochs, opt=None, writer=None, verbose=False):
        best_loss = float("+inf")

        if opt is None:
            opt = optim.Adam(list(self.decoder.parameters())+list(self.encoder.parameters()), lr=0.001)

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

                enc_mu_x, _ = self.encoder(yb)
                dec_mu_y, _ = self.decoder(enc_mu_x)

                loss = F.mse_loss(yb, dec_mu_y)

                loss.backward()
                if loss <= best_loss:
                    best_dec = self.decoder.state_dict()
                    best_enc = self.encoder.state_dict()
                    best_loss = loss

                if writer is not None:
                    n_iter = epoch*len(Y) + start_i
                    if n_iter % 10 == 0:
                        writer.add_scalar("losses/ae_pre_train", loss, n_iter)

                opt.step()
                opt.zero_grad()

        #self.encoder.load_state_dict(best_enc)
        #self.decoder.load_state_dict(best_dec)

        return best_enc, best_dec

    def _fit_step(self, Y, L=10, return_all_terms=False):
        """
        L: integer, number of (x) samples per observation to estimate expectations.
        """
        enc_μs, enc_log_σs_sqr = self.encoder(Y)

        N = Y.shape[0]
        x_samples = Normal(enc_μs, enc_log_σs_sqr.exp().sqrt()).rsample((L,)).flatten(0, 1)

        dec_μs, dec_log_σs_sqr = self.decoder(x_samples)

        # reshape to L-params-per-Y-observation form
        dec_μs = dec_μs.view(N, L, self.y_dim)
        dec_log_σs_sqr = dec_log_σs_sqr.view(N, L, self.y_dim)

        # log (p(z')p(x(l)|z')) = logp(z') + logp(x(l)|z')
        # (explanation of why I'm calling `.unsqueeze` on x_samples can be
        #  found in tests/gmm_log_prob.py)
        aux = (
          (self.gmm_πs + 1e-6).log() +
          Normal(self.gmm_μs, self.gmm_log_σs_sqr.exp().sqrt())
            .log_prob(x_samples.unsqueeze(1))
            .sum(dim=2)
        )

        # assert aux.shape == torch.Size([N*L, self.n_clusters])

        λ_z = (
          # exp-norm the log probs, to get probs -> softmax
          F.softmax(aux, dim=1)
          # reshape in L-samples-per-N-observations shape
          .view(N, L, self.n_clusters)
          # compute the expectation per Y observation
          .mean(dim=1)
        )

        # expected gaussian log prob of observation (1 y obs -> L x samples)
        # (explanation of why I'm calling `.unsqueeze` on Y can be
        #  found in tests/dec_log_prob.py)
        term1 = (
          Normal(dec_μs, dec_log_σs_sqr.exp().sqrt())
            # expand Y to calculate log prob w.r.t the different samples' parameters
            .log_prob(Y.unsqueeze(1))
            # sum the per-dimension log probs, for each sample
            .sum(dim=2)
            # compute the expectation per Y observation
            .mean(dim=1)
        )

        # expected gaussian log prob of continuous latent given cluster label
        # (given by close form -> doesn't use the x samples)
        # (it does indirectly, via q(z|y))
        term2 = -(
          λ_z * (
            self.x_dim/2 * math.log(2*math.pi) +
            0.5 * (
              # sum of log(σ)^2 for fixed z, across dimensions.
              # gmm_log_σs_sqr shape is (n_clusters, n_dimensions)
              self.gmm_log_σs_sqr.sum(dim=1) + # this is shaped (n_clusters,)
              # sum of ratio enc_σs^2 / gmm_σs^2 for fixed z, across dimensions
              (enc_log_σs_sqr.unsqueeze(1) - self.gmm_log_σs_sqr)
                .exp()
                .sum(dim=2) + # this is shaped (batch_size, n_clusters)
              # sqr diff of μs, over gmm_σs^2, for fixed z, summed across dimensions
              (((enc_μs.unsqueeze(1) - self.gmm_μs)**2)/(self.gmm_log_σs_sqr.exp()))
                .sum(dim=2) # this is shaped (batch_size, n_clusters)
            )
          )
        ).sum(dim=1)

        # expected categorical logp(z)
        term3 = (λ_z * (self.gmm_πs + 1e-6).log()).sum(dim=1)

        # expected gaussian log q(x|y)
        term4 = -0.5 * (
                    self.x_dim * math.log(2*math.pi) +
                    (1 + enc_log_σs_sqr).sum(dim=1) # sum across x dimensions
                )

        # expected categorical logq(z|y)
        term5 = (λ_z * (λ_z + 1e-6).log()).sum()

        elbo = (term1 + term2 + term3 - term4 - term5).mean()

        # we want to maximize elbo, so the loss is -elbo
        if return_all_terms:
            return -elbo, -term1.mean(), -term2.mean(), -term3.mean(), term4.mean(), term5.mean()
        else:
            return -elbo

    def fit(self, Y, temperature_schedule=None, n_epochs=1, bs=100, opt=None,
            L=10, clip_grad=None, verbose=False, writer=None):

        best_losses = [float("inf") for i in range(6)]
        best_params = [dict() for i in range(6)]

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

                if writer is not None:
                    losses = self._fit_step(yb, L, return_all_terms=True)
                    loss, loss1, loss2, loss3, loss4, loss5 = losses

                    for i, _loss in enumerate(losses):
                        if _loss <= best_losses[i]:
                            best_losses[i] = _loss.item()
                            best_params[i] = self.state_dict()
                else:
                    loss = self._fit_step(yb, L)
                    if loss <= best_losses[0]:
                        best_losses[0] = loss.item()
                        best_params[0] = self.state_dict()

                # if we're writing to tensorboard
                if writer is not None:
                    n_iter = epoch*len(Y) + start_i
                    if n_iter % 10 == 0:
                        writer.add_scalar('losses/loss1', loss1, n_iter)
                        writer.add_scalar('losses/loss2', loss2, n_iter)
                        writer.add_scalar('losses/loss3', loss3, n_iter)
                        writer.add_scalar('losses/loss4', loss4, n_iter)
                        writer.add_scalar('losses/loss5', loss5, n_iter)
                        writer.add_scalar('losses/-elbo', loss, n_iter)

                loss.backward()

                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

                opt.step()
                opt.zero_grad()

        return best_losses, best_params

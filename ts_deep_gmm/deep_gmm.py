import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

#from pyro.distributions.relaxed_straight_through import RelaxedOneHotCategoricalStraightThrough
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_diag, _batch_mv
from torch.distributions import RelaxedOneHotCategorical, MultivariateNormal, Uniform

def mv_gaussian_log_prob(value, mu, sigma, event_shape):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    """
    scale_tril = torch.cholesky(sigma)
    diff = value - mu
    M = _batch_mahalanobis(scale_tril, diff)
    half_log_det = _batch_diag(scale_tril).log().sum(-1)
    return -0.5 * (event_shape * math.log(2 * math.pi) + M) - half_log_det


# TODO: should DeepGMM be subclass of nn.Module?
#       - Pros: has .paramers() method
#       - Cons: can cause confusion? (no forward nor backward method)
class DeepGMM(nn.Module):
    def __init__(self, n_clusters, y_dim, x_dim, encoder, decoder, reg=1e-6):
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
        self.encoder = encoder # observ. -> latent (params)

        self.phi_gmm_mus = nn.Parameter(torch.stack([
            torch.randn(x_dim)
            for _ in range(n_clusters)
        ]))

        # sigma = sigma_factor @ sigma_factor.t() + sigma_diag
        self.phi_gmm_sigma_diags = nn.Parameter(torch.stack([
            torch.rand(x_dim)
            for _ in range(n_clusters)
        ]))

        self.phi_gmm_sigma_factors = nn.Parameter(torch.stack([
            torch.randn(x_dim)
            for _ in range(n_clusters)
        ]))


        self.phi_gmm_pis = nn.Parameter(F.softmax(torch.randn(n_clusters), dim=0))
        ##########################################################

        ################## Generative parameters #################
        self.decoder = decoder # latent var -> observ (params)

        self.theta_gmm_mus = nn.Parameter(torch.stack([
            torch.randn(y_dim)
            for _ in range(n_clusters)
        ]))

        # sigma = sigma_factor @ sigma_factor.t() + sigma_diag
        self.theta_gmm_sigma_diags = nn.Parameter(torch.stack([
            torch.rand(y_dim)
            for _ in range(n_clusters)
        ]))

        self.theta_gmm_sigma_factors = nn.Parameter(torch.stack([
            torch.randn(y_dim)
            for _ in range(n_clusters)
        ]))


        self.theta_gmm_pis = nn.Parameter(F.softmax(torch.randn(n_clusters), dim=0))
        ##########################################################

    def _sigma(self, factor, diag, diag_size):
        return factor @ factor.t() + torch.eye(diag_size)*(diag + self.reg)

    def phi_gmm_sigma_k(self, k):
        sigma_factor = self.phi_gmm_sigma_factors[k].unsqueeze(-1)
        sigma_diag = self.phi_gmm_sigma_diags[k]

        return self._sigma(sigma_factor, sigma_diag, self.x_dim)

    def theta_gmm_sigma(self, k):
        sigma_factor = self.theta_gmm_sigma_factors[k].unsqueeze(-1)
        sigma_diag = self.theta_gmm_sigma_diags[k]

        return self._sigma(sigma_factor, sigma_diag, self.y_dim)

    def _sigmas(self, factors, diags):
        factors = factors.unsqueeze(-1)
        return (factors @ torch.transpose(factors, 1, 2) +
                torch.diag_embed(diags + self.reg))

    @property
    def phi_gmm_sigmas(self):
        return self._sigmas(self.phi_gmm_sigma_factors, self.phi_gmm_sigma_diags)

    @property
    def theta_gmm_sigmas(self):
        return self._sigmas(self.theta_gmm_sigma_factors, self.theta_gmm_sigma_diags)



    def _fit_step(self, Y, temperature, n_samples=10):
        phi_enc_mu, phi_enc_sigma = self.encoder(Y)
        N = Y.shape[0]
        z_logits = torch.zeros((N, self.n_clusters))
        mu_tilde = torch.zeros((N, self.n_clusters, self.x_dim))
        sigma_tilde = torch.zeros((N, self.n_clusters, self.x_dim, self.x_dim))

        inv_phi_enc_sigma = torch.inverse(phi_enc_sigma)

        # TODO: do this without the loop? I think the question is how
        # to deal with mv_gaussian_log_prob per cluster
        # can I somehow map the loop iteration?
        # EDIT: I think this is doable with flatten/view !!!
        for k in range(self.n_clusters):
            phi_gmm_sigma_k = self.phi_gmm_sigma_k(k)

            z_logits[:,k] = (self.phi_gmm_pis[k].log() + 
                             mv_gaussian_log_prob(value=phi_enc_mu, mu=self.phi_gmm_mus[k],
                                                  sigma=phi_enc_sigma+phi_gmm_sigma_k,
                                                  event_shape=self.x_dim))

            # TODO: check if I'm being the smartest possible about these inverses
            # Repeated inverses? Also confirm the batch operations make sense
            inv_phi_gmm_sigma = torch.inverse(phi_gmm_sigma_k)
            sigma_tilde[:,k,:,:] = torch.inverse(
                inv_phi_enc_sigma + # one matrix per batch element (N, x_dim, x_dim)
                inv_phi_gmm_sigma # one matrix per cluster (x_dim, x_dim)
            )

            mu_tilde[:,k] = _batch_mv(sigma_tilde[:,k], (
                (inv_phi_enc_sigma.unsqueeze(0) @ phi_enc_mu.unsqueeze(-1)).squeeze() +
                (inv_phi_gmm_sigma @ self.phi_gmm_mus[k])
            ))

        # for each batch element, sample n_samples
        # this results in a (N, n_samples, n_clusters) tensor: N*n_samples "pseudo one-hot" vectors
        #z_samples = RelaxedOneHotCategoricalStraightThrough(temperature, logits=z_logits).rsample(n_samples)

        z_samples = RelaxedOneHotCategorical(temperature, logits=z_logits).rsample((n_samples,)).transpose(0, 1)

        # now use each "pseudo one-hot" vector to select which phi_tilde to use
        _mu_tilde = torch.einsum("bsk,bkd->bsd", z_samples, mu_tilde)
        _sigma_tilde = torch.einsum("bsk,bkdf->bsdf", z_samples, sigma_tilde)

        _theta_gmm_mu = torch.einsum("bsk,kd->bsd", z_samples, self.theta_gmm_mus)
        _theta_gmm_sigma = torch.einsum("bsk,kdf->bsdf", z_samples, self.theta_gmm_sigmas)

        _phi_gmm_mu = torch.einsum("bsk,kd->bsd", z_samples, self.phi_gmm_mus)
        _phi_gmm_sigma = torch.einsum("bsk,kdf->bsdf", z_samples, self.phi_gmm_sigmas)

        x_samples = (
            MultivariateNormal(loc=_mu_tilde.flatten(0, 1),
                               covariance_matrix=_sigma_tilde.flatten(0, 1))
                .rsample()
                #.view(100, 5, 2)
        )

        mu_y, sigma_y = self.decoder(x_samples)

        #mu_y = mu_y.view(100, 5, 2)
        #sigma_y = sigma_y.view(100, 5, 2, 2)
        #x_samples = x_samples.view(100, 5, 2)

        unorm_obj = (
            # decoder NN part of the loss
            mv_gaussian_log_prob(Y, mu_y, sigma_y, self.y_dim) +

            # encoder NN part of the loss
            -mv_gaussian_log_prob(x_samples, phi_enc_mu, phi_enc_sigma, self.x_dim) +

            # encoder GMM part of the loss
            (mv_gaussian_log_prob(x_samples, _theta_gmm_mu, _theta_gmm_sigma, self.x_dim) +
             (self.theta_gmm_pis * zamples.flatten(0, 1)).sum(dim=1)
            )

            # decoder GMM part of the loss
            -(mv_gaussian_log_prob(x_samples, _phi_gmm_mu, _phi_gmm_sigma, self.x_dim)
              (z_logits.flatten(0, 1) * z_samples.flatten(0, 1)).sum(dim=1))
        ).sum()/n_samples

        return -(unorm_obj + z_logits.exp().sum(dim=1).log())


    def fit(self, Y, temperature_schedule=None, n_epochs=1, bs=100, opt=None,
            n_samples=10):

        # TODO: implement default values

        if temperature_schedule is None:
            # default temperature schedule, adapted from https://arxiv.org/pdf/1611.01144.pdf
            # TODO: r and M should be hyperparams
            r = Uniform(1e-5, 1e-5).sample()
            M = torch.randint(500, 1001, (1,)).item()
            temperature = 1
            def temperature_schedule(epoch, i, bs):
                t = (epoch * bs + i)
                if t % M == 0:
                    return max(0.5, math.exp(-r * t))
                else:
                    return temperature

        if opt is None:
            # TODO: better defaults
            opt = optim.SGD(self.parameters(), lr=0.01, momentum=0.0)

        for epoch in range(n_epochs):
            for i in range((len(Y) - 1) // bs + 1):
                start_i = i * bs
                end_i = start_i + bs
                yb = Y[start_i:end_i]

                temperature = temperature_schedule(epoch, i, bs)

                loss = self._fit_step(yb, temperature, n_samples)

            loss.backward()
            opt.step()
            opt.zero_grad()

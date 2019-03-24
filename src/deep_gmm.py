import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.autograd as ag

from pyro.distributions.relaxed_straight_through import RelaxedOneHotCategoricalStraightThrough
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_diag

def mv_gaussian_log_prob(value, mu, sigma):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    """
    scale_tril = torch.cholesky(sigma)
    diff = value - mu
    M = _batch_mahalanobis(scale_tril, diff)
    half_log_det = _batch_diag(scale_tril).log().sum(-1)
    return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det


class DeepGMM(nn.Module):
    def __init__(self, n_clusters, y_dim, x_dim, encoder, decoder):
        """
            Note that Y are the observations (the data)
            and X are the latent representations
        """
        super(DeepGMM, self).__init__()

        self.n_clusters = n_clusters
        self.y_dim = y_dim
        self.x_dim = x_dim

        ################# Recognition parameters #################
        self.encoder = encoder # observ. -> latent (params)

        self.phi_gmm_mus = nn.Parameter(torch.stack([
            torch.randn(x_dim)
            for _ in range(n_regimes)
        ]))

        # TODO: low rank sigma instead https://pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal
        self.phi_gmm_sigmas = nn.Parameter(torch.stack([
            torch.randn((x_dim, x_dim))
            for _ in range(n_regimes)
        ]))

        self.phi_gmm_pis = nn.Parameter(F.softmax(torch.randn(n_clusters), dim=0))
        ##########################################################

        ################## Generative parameters #################
        self.decoder = decoder # latent var -> observ (params)

        self.theta_gmm_mus = nn.Parameter(torch.stack([
            torch.randn(y_dim)
            for _ in range(n_regimes)
        ]))

        # TODO: low rank sigma instead
        self.theta_gmm_sigmas = nn.Parameter(torch.stack([
            torch.randn((y_dim, y_dim))
            for _ in range(n_regimes)
        ]))

        self.theta_gmm_pis = nn.Parameter(F.softmax(torch.randn(n_clusters), dim=0))
        ##########################################################

    def _fit_step(self, Y, temperature, n_samples=10):
        phi_enc_mu, phi_enc_sigma = self.encoder(Y)
        N = Y.shape[0]
        z_logits = torch.zeros((N, n_clusters))
        mu_tilde = torch.zeros((N, n_clusters, x_dim))
        sigma_tilde = torch.zeros((N, n_clusters, x_dim, x_dim))

        inv_phi_enc_sigma = torch.inverse(phi_enc_sigma)

        # TODO: do this without the loop?
        for k in range(n_clusters):
            z_logits[k] = phi_gmm_pis[k].log() + mv_gaussian_log_prob(value=phi_enc_mu,
                                                                      loc=phi_gmm_mu,
                                                                      sigma=phi_enc_sigma+phi_gmm_sigmas[k])

            # TODO: check if I'm being the smartest possible about these inverses
            # Repeated inverses? Also confirm the batch operations make sense
            inv_phi_gmm_sigma = torch.inverse(phi_gmm_sigmas[k])
            sigma_tilde[:,k,:,:] = torch.inverse(
                inv_phi_enc_sigma + # one matrix per batch element (N, x_dim, x_dim)
                inv_phi_gmm_sigma # one matrix per cluster (x_dim, x_dim)
            )

            mu_tilde[k] = sigma_tilde[k] @ (inv_phi_enc_sigma @ phi_enc_mu +
                                            inv_phi_gmm_sigma @ phi_gmm_mu[k])

        # for each batch element, sample n_samples
        # this results in a (N, n_samples, n_clusters) tensor: N*n_samples "pseudo one-hot" vectors
        z_samples = RelaxedOneHotCategoricalStraightThrough(temperature, logits=z_logits).rsample(n_samples)

        # now use each "pseudo one-hot" vector to select which phi_tilde to use
        _mu_tilde = torch.einsum("bsk,bkd->bsd", z_samples, mu_tilde)
        _sigma_tilde = torch.einsum("bsk,bkdf->bsdf", z_samples, sigma_tilde)

        # 
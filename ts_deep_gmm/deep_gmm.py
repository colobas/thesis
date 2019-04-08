import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from torch.autograd import gradcheck

#from pyro.distributions.relaxed_straight_through import RelaxedOneHotCategoricalStraightThrough
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_diag, _batch_mv
from torch.distributions import RelaxedOneHotCategorical, MultivariateNormal, Uniform

from tqdm import trange

from ts_deep_gmm.utils.bad_grad_viz import register_hooks

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def mv_gaussian_log_prob(value, μ, Σ_diags, event_shape, debug=False):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    FOR DIAGONAL COV MATRICES ONLY
    """
    diff = value - μ
    M = ((diff**2) / Σ_diags**2).sum(-1)
    half_log_det = Σ_diags.log().sum(-1)

    if debug:
        print("###################################################################################")
        print(f"M.mean(): {M.mean()}")
        print(f"value: {value}")
        print(f"mu: {μ}")
        print(f"half_log_det: {half_log_det}")
        print(f"diff**2: {diff**2}")
        print(f"diags**2: {diags**2}")

    return -0.5 * (event_shape * math.log(2 * math.pi) + M) - half_log_det


# TODO: should DeepGMM be subclass of nn.Module?
#       - Pros: has .paramers() method
#       - Cons: can cause confusion? (no forward nor backward method)
class DeepGMM(nn.Module):
    def __init__(self, n_clusters, y_dim, x_dim, encoder, decoder, reg=0.01,
                 diag_mode=False):
        """
            Note that Y are the observations (the data)
            and X are the latent representations
        """
        super(DeepGMM, self).__init__()

        self.n_clusters = n_clusters
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.reg = reg
        self.diag_mode = diag_mode

        ################# Recognition parameters #################
        self.encoder = encoder # observ. -> latent (params)

        self.ϕ_gmm_μs = torch.stack([
            torch.randn(x_dim)
            for _ in range(n_clusters)
        ])

        self.ϕ_gmm_Σ_diags = torch.stack([
            torch.randn(x_dim)
            for _ in range(n_clusters)
        ])

        # store in logits form because it's unconstrained,
        # then use log_softmax to convert to log probabilities
        self.ϕ_gmm_logits_πs = torch.randn(n_clusters)
        ##########################################################

        ################## Generative parameters #################
        self.decoder = decoder # latent var -> observ (params)

        self.θ_gmm_μs = torch.stack([
            torch.randn(y_dim)
            for _ in range(n_clusters)
        ])

        self.θ_gmm_Σ_diags = torch.stack([
            torch.randn(y_dim)
            for _ in range(n_clusters)
        ])

        # store in logits form because it's unconstrained,
        # then use log_softmax to convert to log probabilities
        self.θ_gmm_logits_πs = torch.randn(n_clusters)
        ##########################################################

    def forward(Y):
        """
        Given data points Y, infer their latent X and Z
        """
        raise NotImplemented

    def _fit_step(self, Y, temperature, n_samples=10):
        ϕ_enc_μ, ϕ_enc_Σ = self.encoder(Y)
        N = Y.shape[0]
        z_logits = torch.zeros((N, self.n_clusters), device=_DEVICE)
        μ_tilde = torch.zeros((N, self.n_clusters, self.x_dim), device=_DEVICE)
        Σ_tilde = torch.zeros((N, self.n_clusters, self.x_dim, self.x_dim), device=_DEVICE)

        #inv_ϕ_enc_Σ = torch.inverse(ϕ_enc_Σ)
        inv_ϕ_enc_Σ = 1/ϕ_enc_Σ

        # TODO: do this without the loop? I think the question is how
        # to deal with mv_gaussian_log_prob per cluster
        # can I somehow map the loop iteration?
        # EDIT: I think this is doable with flatten/view !!!

        ϕ_gmm_log_πs = F.log_softmax(self.ϕ_gmm_logits_πs, dim=0)

        for k in range(self.n_clusters):
            aux = mv_gaussian_log_prob(value=ϕ_enc_μ, μ=self.ϕ_gmm_μs[k],
                                       Σ_diags=ϕ_enc_Σ+ϕ_gmm_Σ[k],
                                       event_shape=self.x_dim)
            z_logits[:,k] = (ϕ_gmm_log_πs[k] + aux)

            # TODO: check if I'm being the smartest possible about these inverses
            # Repeated inverses? Also confirm the batch operations make sense
            #inv_ϕ_gmm_Σ = torch.inverse(ϕ_gmm_Σ[k])
            inv_ϕ_gmm_Σ = 1/torch.inverse(ϕ_gmm_Σ[k])

            #Σ_tilde[:,k,:,:] = torch.inverse(
            Σ_tilde[:,k,:] = 1/(
                inv_ϕ_enc_Σ + # one matrix per batch element (N, x_dim, x_dim)
                inv_ϕ_gmm_Σ # one matrix per cluster (x_dim, x_dim)
            )

            #μ_tilde[:,k] = _batch_mv(Σ_tilde[:,k], (
            μ_tilde[:,k] = (Σ_tilde[:,k] * (
                #(inv_ϕ_enc_Σ.unsqueeze(0) @ ϕ_enc_μ.unsqueeze(-1)).squeeze() +
                #(inv_ϕ_gmm_Σ @ self.ϕ_gmm_μs[k])
                (inv_ϕ_enc_Σ.unsqueeze(0) * ϕ_enc_μ).squeeze() +
                (inv_ϕ_gmm_Σ * self.ϕ_gmm_μs[k])
            ))

        # for each batch element, sample n_samples
        # this results in a (N, n_samples, n_clusters) tensor: N*n_samples "pseudo one-hot" vectors

        z_log_probs = F.log_softmax(z_logits, dim=-1)
        z_samples = RelaxedOneHotCategorical(temperature, probs=z_log_probs.exp()).rsample((n_samples,)).transpose(0, 1)

        # now use each "pseudo one-hot" vector to select which ϕ_tilde to use
        _μ_tilde = torch.einsum("bsk,bkd->bsd", z_samples, μ_tilde)
        _Σ_tilde = torch.einsum("bsk,bkdf->bsd", z_samples, Σ_tilde)

        _θ_gmm_μ = torch.einsum("bsk,kd->bsd", z_samples, self.θ_gmm_μs)
        _θ_gmm_Σ = torch.einsum("bsk,kdf->bsd", z_samples, self.θ_gmm_Σs)

        _ϕ_gmm_μ = torch.einsum("bsk,kd->bsd", z_samples, self.ϕ_gmm_μs)
        _ϕ_gmm_Σ = torch.einsum("bsk,kdf->bsd", z_samples, self.ϕ_gmm_Σs)

        x_samples = (
            MultivariateNormal(loc=_μ_tilde.flatten(0, 1),
                               covariance_matrix=torch.diag_embed(_Σ_tilde.flatten(0, 1)))
                .rsample()
                #.view(100, 5, 2)
        )

        μ_y, Σ_y = self.decoder(x_samples)

        #μ_y = μ_y.view(100, 5, 2)
        #Σ_y = Σ_y.view(100, 5, 2, 2)
        #x_samples = x_samples.view(100, 5, 2)

        # decoder NN part of the loss
        loss1 = mv_gaussian_log_prob(Y.unsqueeze(1).repeat(1, n_samples, 1).flatten(0, 1),
                                   μ_y,
                                   Σ_y,
                                   self.y_dim,
                                   self.diag_mode)

        # encoder NN part of the loss
        loss2 = - mv_gaussian_log_prob(x_samples,
                                         ϕ_enc_μ.unsqueeze(1).repeat(1, n_samples, 1).flatten(0, 1),
                                         ϕ_enc_Σ.unsqueeze(1).repeat(1, n_samples, 1).flatten(0, 1),
                                         self.x_dim,
                                         self.diag_mode)

        # encoder GMM part of the loss
        loss3 = (mv_gaussian_log_prob(x_samples,
                                         _θ_gmm_μ.flatten(0, 1),
                                         _θ_gmm_Σ.flatten(0, 1),
                                         self.x_dim,
                                         self.diag_mode) +
                 (F.log_softmax(self.θ_gmm_logits_πs, dim=0) *
                     z_samples.flatten(0, 1)).sum(dim=1))

        # decoder GMM part of the loss
        loss4 = -(mv_gaussian_log_prob(x_samples,
                                         _ϕ_gmm_μ.flatten(0, 1),
                                         _ϕ_gmm_Σ.flatten(0, 1),
                                         self.x_dim,
                                         self.diag_mode) +
                  (z_log_probs.unsqueeze(1) * z_samples).flatten(0, 1).sum(dim=1))

        loss5 = z_log_probs.exp().sum(dim=1).log().sum()


        #for i, loss in enumerate([loss1, loss2, loss3, loss4, loss5]):
        #    print(f"loss_{i+1}: {loss}")

        # what's inside parens is the ELBO, and we want to maximize it,
        # hence minimize its reciprocal
        res =  -((loss1 + loss2 + loss3 + loss4).sum()/n_samples + loss5)

        return res


    def fit(self, Y, temperature_schedule=None, n_epochs=1, bs=100, opt=None,
            n_samples=10, verbose=False, debug=False):

        # TODO: implement default values

        if temperature_schedule is None:
            # default temperature schedule, adapted from https://arxiv.org/pdf/1611.01144.pdf
            # TODO: r and M should be hyperparams
            r = Uniform(1e-5, 1e-4).sample()
            M = torch.randint(bs//2, bs, (1,)).item()
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

        losses = []

        if verbose:
            epochs = trange(n_epochs, desc="epoch")
            batches = trange((len(Y) - 1) // bs + 1, desc="batch")
        else:
            epochs = range(n_epochs)
            batches = range((len(Y) - 1) // bs + 1)

        for epoch in epochs:
            for i in batches:
                start_i = i * bs
                end_i = start_i + bs
                yb = Y[start_i:end_i]

                temperature = temperature_schedule(epoch, i, bs)

                loss = self._fit_step(yb, temperature, n_samples)
                losses.append((epoch, i, loss.item()))
                if debug != 0:
                    if loss != loss:
                        get_dot = register_hooks(loss)

                loss.backward()
                if debug != 0:
                    debug_str = ""
                    print_me = False

                    print(self.θ_gmm_Σs)
                    print(self.ϕ_gmm_Σs)

                    if (epoch * bs + i) < debug:
                        debug_str += (f"################### debug iter {(epoch * bs + i)} #####################\n")
                        for n, p in self.named_parameters():
                            grad_mean = p.grad.mean()
                            debug_str += (f"{n}: {p.grad.mean()}\n")
                            if grad_mean != grad_mean:
                                print_me = True
                        debug_str += "\n"
                        if loss != loss:
                            dot = get_dot() 
                            dot.render(f"iter_{i}", format="pdf")
                            raise Exception("NAN LOSS")
                        if print_me:
                            print(debug_str)
                    else:
                        assert False


                opt.step()
                opt.zero_grad()

        return losses

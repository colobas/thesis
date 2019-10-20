import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

class VariationalMixture(nn.Module):
    def __init__(self, xdim, hdim, n_hidden, n_classes, components, log_prior=None, encoder=None):
        super().__init__()

        self.xdim = xdim

        if encoder is None:
            net_modules = (
              [nn.Linear(xdim, hdim), nn.ReLU(), nn.BatchNorm1d(hdim)] +
              sum([[nn.Linear(hdim, hdim), nn.ReLU(), nn.BatchNorm1d(hdim)] for i in range(n_hidden)], []) +
              [nn.Linear(hdim, n_classes)]
            )
            self.encoder = nn.Sequential(*net_modules)
        else:
            self.encoder = encoder

        self.components = nn.ModuleList(components)
        self.n_components = len(components)

        if log_prior is None:
            log_prior = torch.Tensor([1/len(components)]*len(components)).log()

        self.register_buffer(
            'log_prior',
            log_prior
        )

    def to(self, device):
        super(VariationalMixture, self).to(device)
        for component in self.components:
            component.to(device)

        self.log_prior = self.log_prior.to(device)

    def forward(self, x, T=1):
        x = self.encoder(x)
        return F.softmax(x/T, dim=1)

    def update_mixture_weights(self, x):
        num = torch.zeros(len(x), len(components))
        for k in range(self.n_components):
            num[:, k] = (self.n_components[k].log_prob(x) + self.log_prior[k]).exp()

        num = num / num.sum(dim=1)

        self.log_prior = (num.mean(dim=0) + 1e-6).log()

        return self.log_prior

    def loss_terms(self, x, T=1):
        q = self.forward(x, T)

        log_probs = 0
        for k in range(self.n_components):
            log_probs = log_probs + q[:, k] * self.components[k].log_prob(x)

        log_probs = log_probs.mean()
        prior_crossent = (q * self.log_prior).sum(dim=1).mean()
        q_entropy = - (q * (q + 1e-6).log()).sum(dim=1).mean()

        return log_probs, prior_crossent, q_entropy

    def supervised_loss_terms(self, x, y, T):
        log_probs = 0
        for k in range(self.n_components):
            log_probs = log_probs + y[:, k] * self.components[k].log_prob(x)

        q = self.forward(x, T)

        log_probs = log_probs.mean()

        bce = F.binary_cross_entropy(q, y)

        return log_probs, bce

    def fit(self, dataloader, n_epochs=1, opt=None, temperature_schedule=None,
            clip_grad=None, verbose=False, pre_backward_callback=None,
            post_backward_callback=None, sup_dataloader=None):

        best_loss = float("inf")
        best_params = dict()
        len_X = len(dataloader.dataset)

        if temperature_schedule is None:
            temperature_schedule = lambda t: 1

        if verbose:
            epochs = trange(n_epochs, desc="epoch")
        else:
            epochs = range(n_epochs)

        if sup_dataloader is not None:
            opt_sup = optim.Adam(self.parameters(), lr=1e-3)
            opt_sup.zero_grad()

        temperature = 1
        for epoch in epochs:
            if sup_dataloader is not None:
                for i, (xb, yb) in enumerate(sup_dataloader):
                    opt_sup.zero_grad()
                    log_probs, bce = self.supervised_loss_terms(xb, yb, temperature)
                    loss = -log_probs - bce
                    loss.backward()
                    opt_sup.step()

            for i, xb in enumerate(dataloader):
                opt.zero_grad()
                n_iter = epoch*((len_X - 1) // dataloader.batch_size + 1) + i
                temperature = temperature_schedule(n_iter)

                log_probs, prior_crossent, q_entropy = self.loss_terms(xb, temperature)
                loss = -(log_probs + prior_crossent + q_entropy)

                if loss != loss:
                    #if loss is nan don't backprop
                    continue

                if loss <= best_loss:
                    best_loss = loss.item()
                    best_params = self.state_dict()

                if pre_backward_callback is not None:
                    with torch.no_grad():
                        pre_backward_callback(self, n_iter, log_probs, prior_crossent, q_entropy, temperature)

                loss.backward()

                if post_backward_callback is not None:
                    with torch.no_grad():
                        post_backward_callback(self, n_iter, temperature, best_params, best_loss)

                opt.step()

        return best_loss, best_params, temperature

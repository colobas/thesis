import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.autograd as ag

from st_gumbel import gumbel_softmax, gumbel_softmax_sample

PI = torch.Tensor([np.pi]).cuda()

def categorical_kld(logp, logq):
    p = torch.exp(logp)
    q = torch.exp(logq)

    t = p * (logp - logq)

    return t.sum().sum()

def stud_t_log_prob(value, df, loc, scale):
    # adapted from: https://pytorch.org/docs/stable/_modules/torch/distributions/studentT.html
    y = (value - loc) / scale
    Z = (scale.log() +
         0.5 * df.log() +
         0.5 * PI.log() +
         torch.lgamma(0.5 * df) -
         torch.lgamma(0.5 * (df + 1.)))
    return -0.5 * (df + 1.) * torch.log1p(y**2. / df) - Z

class Model(nn.Module):
    def __init__(self, n_dims, bottlenecks, n_regimes, dilations,
                 kernel_size, intrinsic_probs, capacity, p):
        super(Model, self).__init__()

        self.n_dims = n_dims
        self.bottlenecks = bottlenecks
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.n_regimes = n_regimes
        self.intrinsic_probs = intrinsic_probs
        self.sum_intrinsic = sum(intrinsic_probs)

        self.v_0 = v_0
        self.a_0 = a_0
        self.b_0 = b_0
        self.m_0 = m_0

        assert len(bottlenecks) == len(dilations)

        n_layers = len(bottlenecks)
        self.n_layers = n_layers

        self.convs = [
            nn.Conv1d(
                n_dims,
                bottlenecks[0],
                kernel_size,
                dilation=dilations[0],
            ).cuda()
        ]
        self.convs += [
            nn.Conv1d(
                bottlenecks[l-1],
                bottlenecks[l],
                kernel_size,
                dilation=dilations[l],
            ).cuda() for l in range(1,n_layers - 1)
        ]
        self.convs.append(
            nn.Conv1d(
                bottlenecks[-1],
                n_regimes,
                kernel_size,
                dilation=dilations[-1],
            ).cuda()
        )

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x, temp, T, capacity, p):
        x = F.pad(x, (sum(self.dilations), 0, 0, 0))
        z = [torch.zeros_like(x[:,:,0]).cuda() for _ in range(sum(self.dilations))]
        logq = []

        for t in range(capacity, T):
            _x = x[:,:,t-capacity:t]
            _z = torch.stack(z[t-capacity:t])

            inp = torch.cat(_x, _z, dim=1)

            for conv in self.convs:
                inp = conv(inp)
                inp = F.elu(inp)
                _logq = F.log_softmax(inp)

            logq.append(_logq)
            z.append(gumbel_softmax(_logp.permute(0,2,1), temp).permute(0,2,1))

        return torch.stack(z, dim=2), torch.stack(logq, dim=2)

    def reweighting(self, capacity):
        _log_loss = 0
        _log_wtk = []
        _n = []

        _n.append(z[:,:,0:capacity].sum(dim=2))
        for t in range(capacity+1,T):
            _n.append(_n[t-1] - z[:,:,t-capacity-1] + z[:,:,t])
            log_wtk = torch.zeros_like(z[:,:,0])

            _n_tk0 = torch.einsum("bit->bi",[z[:,:,:t-0-1]]
            _sum_x_0tk = torch.einsum("bit,bjt->bij",[x[:,:,:t-1]],z[:,:,0,t-1]]
            _sum_x_sqr_0tk = torch.einsum("bit,bjt->bij",[x[:,:,:t-1]].pow(2),z[:,:,0,t-1]])

            v_tk0 = 1/(1/self.v_0[0] + _n_tk0)
            m_tk0 = v_tk0 * (1/self.v_0[0] + _sum_x_0tk)
            a_tk0 = self.a_0[0] + _n_tk0/2
            b_tk0 = self.b_0 + 0.5 * (self.m_0[0].pow(2)/self.v_0[0] + _sum_x_sqr_0tk - m_tk0.pow(2)/v_tk0)

            _log_loss = _log_loss + sum([stud_t_log_prob(x[:,d,t], a_tk0, m_tk0, b_tk0*(v_tk0 + 1)/a_tk0) for d in range(n_dimensions)])

            for i in range(1,p):
                _n_tki = torch.einsum("bit->bi",[z[:,:,:t-i-1]]
                _sum_x_itk = torch.einsum("bit,bjt->bij",[x[:,:,:t-i-1]],z[:,:,i,t-1]]
                _sum_x_sqr_itk = torch.einsum("bit,bjt->bij",[x[:,:,:t-i-1]].pow(2),z[:,:,i,t-1]])

                v_tki = 1/(1/self.v_0[i] + _n_tki)
                m_tki = v_tki * (1/self.v_0[i] + _sum_x_itk)
                a_tki = self.a_0[i] + _n_tki/2
                b_tki = self.b_0 + 0.5 * (self.m_0[i].pow(2)/self.v_0[i] + _sum_x_sqr_itk - m_tki.pow(2)/v_tki)

                log_wtk = log_wtk + sum([stud_t_log_prob(x[:,d,t-i], a_tki, m_tki, b_tki*(v_tki + 1)/a_tki) for d in range(n_dimensions)])
            _log_wtk.append(log_wtk)

        _log_wtk = torch.stack(_log_wtk, dim=2)
        _sum_log_wtk = _log_wtk.sum(dim=1)

        return _log_wtk, _sum_log_wtk, torch.stack(_n, dim=2), _log_loss

    def kl_regimes(self, capacity, _n_tk, log_wtk, sum_log_wtk):
        for k in range(n_regimes):
            _n_tk[:,k,:] = (_n_tk[:,k,:].clone() + self.intrinsic_probs[k])

        return torch.log(_n_tk + 1e-6) - torch.log(capacity + self.sum_intrinsic + 1e-6) + _log_wtk - _sum_log_wtk

    def compute_loss(self, x, z):
        _log_wtk, _sum_log_wtk, _n_tk, log_loss = self.reweighting(self.capacity)
        kl_loss = self.kl_regimes(self.capacity, _n_tk, _log_wtk, _sum_log_wtk)

        return log_loss, kl_loss


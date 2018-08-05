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

    y = (value.unsqueeze(1) - loc) / scale
    Z = (scale.log() +
         0.5 * df.log() +
         0.5 * PI.log() +
         torch.lgamma(0.5 * df) -
         torch.lgamma(0.5 * (df + 1.)))

    return -0.5 * (df + 1.) * torch.log1p(y**2. / df) - Z

def student_t_sample(loc, scale, df):
    X = df.new(df.shape).normal_()
    Z = Chi2(df).rsample(df.shape)
    Y = X * torch.rsqrt(Z / df)
    return loc + scale * Y

class Model(nn.Module):
    def __init__(self, n_dims, bottlenecks, n_regimes, dilations,
                 kernel_size, intrinsic_probs, capacity, p, v_0, a_0,
                 m_0, b_0):
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

        self.capacity = capacity
        self.p = p

        assert len(bottlenecks) == len(dilations)

        n_layers = len(bottlenecks)
        self.n_layers = n_layers

        self.convs = [
            nn.Conv1d(
                n_dims+n_regimes,
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

    def forward(self, x, temp, T):
        x = F.pad(x, (sum(self.dilations), 0, 0, 0))
        #z = [torch.zeros_like(x[:,:,0].unsqueeze(2)).cuda() for _ in range(sum(self.dilations))]
        z = [torch.zeros((x.shape[0],self.n_regimes,1)).cuda() for _ in range(self.capacity)]
        logq = []

        for t in range(self.capacity, T):
            _x = x[:,:,t-self.capacity:t]
            _z = torch.cat(z[t-self.capacity:t], dim=2)

            inp = torch.cat((_x, _z), dim=1)

            for conv in self.convs:
                inp = conv(inp)
                inp = F.elu(inp)

            _logq = F.log_softmax(inp, dim=1)[:,:,-1].unsqueeze(2)
            logq.append(_logq)

            z.append(gumbel_softmax(_logq.permute(0,2,1), temp).permute(0,2,1))

        return torch.cat(z, dim=2), torch.stack(logq, dim=2)

    def reweighting(self, capacity, x, z, T):
        _log_loss = 0
        _log_wtk = []
        _n = []

        for t in range(capacity,T):
            _n.append(z[:,:,t-capacity:t].sum(dim=2))
            #_n.append(_n[t-capacity-1] - z[:,:,t-capacity-1] + z[:,:,t])
            log_wtk = torch.zeros_like(z[:,:,0])

            _n_tk0 = torch.einsum("bit->bi",[z[:,:,:t-1]])
            _sum_x_0tk = torch.einsum("bit,bjt->bji",[x[:,:,:t-1],z[:,:,:t-1]])
            _sum_x_sqr_0tk = torch.einsum("bit,bjt->bji",[x[:,:,:t-1].pow(2),z[:,:,:t-1]])

            v_tk0 = 1/((1/self.v_0[0]).unsqueeze(0) + _n_tk0.unsqueeze(2))
            m_tk0 = v_tk0 * ((1/self.v_0[0]) + _sum_x_0tk)
            a_tk0 = self.a_0[0].unsqueeze(0).unsqueeze(0) + (_n_tk0/2).unsqueeze(2)
            b_tk0 = self.b_0 + 0.5 * (self.m_0[0].pow(2)/self.v_0[0] + _sum_x_sqr_0tk - m_tk0.pow(2)/v_tk0)

            _log_loss = _log_loss + sum([torch.einsum("bj,bj->b", [z[:,:,t], stud_t_log_prob(x[:,d,t], a_tk0[:,:,d],
                m_tk0[:,:,d], b_tk0[:,:,d]*(v_tk0[:,:,d] + 1)/a_tk0[:,:,d])]) for d in range(self.n_dims)])

            for i in range(1,self.p):
                _n_tki = torch.einsum("bit->bi",[z[:,:,:t-i-1]])
                _sum_x_itk = torch.einsum("bit,bjt->bji",[x[:,:,:t-i-1],z[:,:,i:t-1]])
                _sum_x_sqr_itk = torch.einsum("bit,bjt->bji",[x[:,:,:t-i-1].pow(2),z[:,:,i:t-1]])

                v_tki = 1/((1/self.v_0[i]).unsqueeze(0) + _n_tki.unsqueeze(2))
                m_tki = v_tki * ((1/self.v_0[i]).unsqueeze(0) + _sum_x_itk)
                a_tki = self.a_0[i].unsqueeze(0).unsqueeze(0) + (_n_tki/2).unsqueeze(2)
                b_tki = self.b_0 + 0.5 * (self.m_0[i].pow(2)/self.v_0[i] + _sum_x_sqr_itk - m_tki.pow(2)/v_tki)

                log_wtk = log_wtk + sum([stud_t_log_prob(x[:,d,t-i], a_tki[:,:,d], 
                    m_tki[:,:,d], b_tki[:,:,d]*(v_tki[:,:,d] + 1)/a_tki[:,:,d]) for d in range(self.n_dims)])
            _log_wtk.append(log_wtk)

        _log_wtk = torch.stack(_log_wtk, dim=2)
        _sum_log_wtk = _log_wtk.sum(dim=1)

        return _log_wtk, _sum_log_wtk, torch.stack(_n, dim=2), _log_loss

    def kl_regimes(self, z, capacity, _n_tk, log_wtk, sum_log_wtk):
        for k in range(self.n_regimes):
            _n_tk[:,k,:] = (_n_tk[:,k,:].clone() + self.intrinsic_probs[k])

        return torch.einsum("bkt,bkt->b", [z[:,:,capacity:], torch.log(_n_tk + 1e-6)\
             - torch.log(torch.Tensor([capacity + self.sum_intrinsic + 1e-6]).cuda())\
             + log_wtk\
             - sum_log_wtk.unsqueeze(1)]).sum()

    def compute_loss(self, x, z, T):
        _log_wtk, _sum_log_wtk, _n_tk, log_loss = self.reweighting(self.capacity, x, z, T)
        kl_loss = self.kl_regimes(z, self.capacity, _n_tk, _log_wtk, _sum_log_wtk)

        return log_loss.sum(), kl_loss


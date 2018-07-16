import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.autograd as ag

import math

from st_gumbel import gumbel_softmax

def categorical_kld(p, q):
    return torch.sum(torch.sum(torch.mul(p, torch.log(p/q)), 1))

def loglikelihood(mus, sigmas, preds):
    return torch.sum(-0.5 * torch.log(2 * np.pi * sigmas**2) - (1/(2*(sigmas**2)))*(mus-preds)**2 , 2)

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class Model(nn.Module):
    def __init__(self, n_layers, n_dims, bottlenecks, n_regimes,
                 dilations, kernel_size, window):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.n_dims = n_dims
        self.bottlenecks = bottlenecks
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.n_regimes = n_regimes
        self.window = window

        assert n_layers == len(dilations)
        assert n_layers == len(bottlenecks)

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

        dilations = list(reversed(dilations))

        self.inference_convs = [
            nn.Conv1d(
                n_dims,
                bottlenecks[0],
                kernel_size,
                dilation=dilations[0],
            ).cuda()
        ]
        self.inference_convs += [
            nn.Conv1d(
                bottlenecks[l-1],
                bottlenecks[l],
                kernel_size,
                dilation=dilations[l],
            ).cuda() for l in range(1,n_layers - 1)
        ]
        self.inference_convs.append(
            nn.Conv1d(
                bottlenecks[-1],
                n_regimes,
                kernel_size,
                dilation=dilations[-1],
            ).cuda()
        )
        self.inference_convs = nn.ModuleList(self.inference_convs)

        self.final_infer_convs = [
            nn.Conv1d(n_regimes*2, n_regimes*2, 1).cuda(),
            nn.Conv1d(n_regimes*2, n_regimes*2, 1).cuda(),
            nn.Conv1d(n_regimes*2, n_regimes, 1).cuda(),
        ]
        self.final_infer_convs = nn.ModuleList(self.final_infer_convs)

        self.params_convs = [
            nn.Conv1d(n_regimes*2, n_regimes*2, 1).cuda(),
            nn.Conv1d(n_regimes*2, n_regimes*2, 1).cuda(),
            nn.Conv1d(n_regimes*2, n_dims, 1).cuda(),
        ]
        self.params_convs = nn.ModuleList(self.params_convs)

        self.sqrt_cov_tt = nn.Parameter(torch.stack([
            torch.rand(1).cuda()\
            for _ in range(n_regimes)
        ]))
        self.fish_jt = nn.Parameter(torch.stack([
            torch.zeros(window).cuda()\
            for _ in range(n_regimes)
        ]))


    def inference_pass(self, y):
        y = F.pad(y, (sum(self.dilations), 0, 0, 0))
        for conv in self.inference_convs:
            y = conv(y)
            y = F.rrelu(y)
        y = flip(y,2)
        return y

    def forward_pass(self, x):
        x = F.pad(x, (sum(self.dilations), 0, 0, 0))
        for conv in self.convs:
            x = conv(x)
            x = F.rrelu(x)
        return x


    def infer(self, y):
        h_vecs = self.forward_pass(y[:,:,:-1])

        b_vecs = self.inference_pass(y[:,:,1:])

        h_inf = torch.cat((h_vecs, b_vecs), 1)
        for conv in self.final_infer_convs:
            h_inf = conv(h_inf)
            h_inf = F.rrelu(h_inf)

        q = F.softmax(h_inf, 1)

        return q

    def gen(self, x):
        h_vecs = self.forward_pass(x)

        gen_z = gumbel_softmax(h_vecs.permute(0,2,1), 0.1).permute(0,2,1)
        mus = self.params_convs[0](torch.cat((gen_z, h_vecs), 1))
        for conv in self.params_convs[1:]:
            mus = F.rrelu(mus)
            mus = conv(mus)

        window = self.window

        cov_tt = (torch.einsum('ijk,jl->ilk', [gen_z, self.sqrt_cov_tt])**2)[:,:,window:gen_z.shape[-1]]
        fish_jt = torch.einsum('ijk,jl->ilk', [gen_z, self.fish_jt])[:,:,window:gen_z.shape[-1]]

        x_window = []
        mus_window = []

        for i in range(window, x.shape[-1]):
            x_window.append(x[:,:,i-window:i].squeeze(1))
            mus_window.append(mus[:,:,i-window:i].squeeze(1))

        x_window = torch.stack(x_window, 2)
        mus_window = torch.stack(mus_window, 2)

        _mus = mus[:,:,window:gen_z.shape[-1]].squeeze(1) - torch.einsum(
            'ijt,ijt->it', [(cov_tt)*(fish_jt), (x_window - mus_window)]
        )


        #mu = _mus[:,-1]

#        normals = dist.Normal(mu, cov_tt)
#        return normals.sample(), mu, gen_z

        return _mus, gen_z

    def compute_loss(self, inputs, temp):
        x, y = inputs

        b_vecs = self.inference_pass(y)
        h_vecs = self.forward_pass(x)

        h_inf = torch.cat((h_vecs, b_vecs), 1)
        for conv in self.final_infer_convs:
            h_inf = conv(h_inf)
            h_inf = F.rrelu(h_inf)

        p = F.softmax(h_vecs, 1)
        q = F.softmax(h_inf, 1)

        post_z = gumbel_softmax(h_inf.permute(0,2,1), temp).permute(0,2,1)
        mus = self.params_convs[0](torch.cat((post_z, h_vecs), 1))
        for conv in self.params_convs[1:]:
            mus = F.rrelu(mus)
            mus = conv(mus)

        window = self.window

        cov_tt = (torch.einsum('ijk,jl->ilk', [post_z, self.sqrt_cov_tt])**2)[:,:,window:post_z.shape[-1]]
        fish_jt = torch.einsum('ijk,jl->ilk', [post_z, self.fish_jt])[:,:,window:post_z.shape[-1]]
        #cov_tt = (torch.einsum('ijk,jl->ilk', [F.softmax(h_inf), self.sqrt_cov_tt])**2)[:,:,window:post_z.shape[-1]]
        #fish_jt = torch.einsum('ijk,jl->ilk', [F.softmax(h_inf), self.fish_jt])[:,:,window:post_z.shape[-1]]



        y_window = []
        mus_window = []

        for i in range(window, y.shape[-1]):
            y_window.append(y[:,:,i-window:i].squeeze(1))
            mus_window.append(mus[:,:,i-window:i].squeeze(1))

        y_window = torch.stack(y_window, 2)
        mus_window = torch.stack(mus_window, 2)

        _mus = mus[:,:,window:post_z.shape[-1]].squeeze(1) - torch.einsum(
            'ijt,ijt->it', [(cov_tt)*(fish_jt), (y_window - mus_window)]
        )

        logl = loglikelihood(_mus, cov_tt, y[:,:,window:post_z.shape[-1]])

        return torch.mean(logl), -categorical_kld(q, p), torch.mean(
                                        F.cosine_similarity(q, 
                                            torch.ones(post_z.shape).cuda(), dim=1))
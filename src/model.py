import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.autograd as ag

import math

from st_gumbel import gumbel_softmax, gumbel_softmax_sample


PI = torch.Tensor([np.pi]).cuda()

def categorical_kld(p, q):
    return torch.sum(torch.sum(torch.mul(p, torch.log(p+1e-6)-torch.log(q+1e-6)), 1))

#def categorical_kld(p, q, p_logits, q_logits):
#    t = p * (p_logits - q_logits)
#    t[q == 0] = float('inf')
#    t[p == 0] = 0
#    return t.sum().sum()

def loglikelihood(mus, logvars, preds):
    var = torch.exp(logvars)
    diff = mus-preds

    res = torch.sum(-0.5 * torch.log(2 * PI) - var, 2)
    res = res + torch.sum(-(0.5 * (1/(var**2))) * diff**2, 2)

    return res

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def regime_dynamics(p, inert, temp):
    #z = torch.zeros_like(p).cuda()
    z = []
    #new_p = torch.zeros_like(p).cuda()
    new_p = []
    inert = (0.5 + 0.5 * torch.tanh(inert).unsqueeze(1)).clamp(min=0.01, max=0.99)

    z.append(gumbel_softmax(torch.log(p[:,:,0] + 1e-6).unsqueeze(2).permute(0,2,1), temp).permute(0,2,1))
    new_p.append(p[:,:,0].clone().unsqueeze(2))
    for t in range(1,p.shape[-1]):
        new_p.append(((1-inert[:,:,t])*p[:,:,t] + inert[:,:,t]*z[t-1].squeeze()).unsqueeze(2))
        z.append(gumbel_softmax(torch.log(new_p[t] + 1e-6).permute(0,2,1), temp).permute(0,2,1))

    new_p = torch.cat(new_p, dim=2)
    z = torch.cat(z, dim=2)

    return new_p, z

class Model(nn.Module):
    def __init__(self, n_dims, bottlenecks, n_regimes, dilations,
                 kernel_size, window, hidden_temp, params_net_layers,
                 inertia):
        super(Model, self).__init__()

        self.n_dims = n_dims
        self.bottlenecks = bottlenecks
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.n_regimes = n_regimes
        self.window = window
        self.hidden_temp = hidden_temp
        self.params_net_layers = params_net_layers
        self.inertia = inertia

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
                n_regimes + 1,
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
                n_regimes + 1,
                kernel_size,
                dilation=dilations[-1],
            ).cuda()
        )
        self.inference_convs = nn.ModuleList(self.inference_convs)

        self.params_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                torch.rand(params_net_layers[i+1], params_net_layers[i]).cuda()\
                for _ in range(n_regimes)
            ])) for i in range(len(params_net_layers)-1)
        ])
        self.params_bias = nn.ParameterList([
            nn.Parameter(torch.stack([
                torch.rand(params_net_layers[i+1]).cuda()\
                for _ in range(n_regimes)
            ])) for i in range(len(params_net_layers)-1)
        ])

    def inference_pass(self, y):
        y = flip(y,2)
        y = F.pad(y, (sum(self.dilations), 0, 0, 0))
        for conv in self.inference_convs:
            y = conv(y)
            y = F.elu(y)
        y = flip(y,2)
        return y

    def forward_pass(self, x):
        x = F.pad(x, (sum(self.dilations), 0, 0, 0))
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.elu(x)

        return F.elu(self.convs[-1](x)), x


    def infer(self, y):
        h_vecs, _ = self.forward_pass(y[:,:,:-1])

        b_vecs = self.inference_pass(y[:,:,1:])

        #h_inf = torch.cat((h_vecs, b_vecs), 1)
        #h_inf = h_vecs+b_vecs
        h_inf = b_vecs

        q = F.softmax(h_inf[:,:-1,:]/self.hidden_temp, 1)
        if self.inertia:
            _q, post_z = regime_dynamics(q, h_inf[:,-1,:], temp)
        else:
            _q = q
            #post_z = gumbel_softmax_sample(torch.log(q).permute(0,2,1), temp).permute(0,2,1)
            post_z = gumbel_softmax(torch.log(q + 1e-6).permute(0,2,1), 0.2).permute(0,2,1)

        return _q

    def gen(self, x):
        h_vecs, pre_h_vecs = self.forward_pass(x)
        pre_h_vecs = pre_h_vecs[:,:,self.dilations[-1]:]

        p = F.softmax(h_vecs[:,:-1,:]/self.hidden_temp + 1e-6, 1)
        if self.inertia:
            _, gen_z = regime_dynamics(p, h_vecs[:,-1,:], 0.2)
        else:
            gen_z = gumbel_softmax(torch.log(p + 1e-6).permute(0,2,1), 0.2).permute(0,2,1)

        weight = torch.einsum('ijk,jel->ielk',[gen_z, self.params_weights[0]])
        bias = torch.einsum('ijk,jl->ilk',[gen_z, self.params_bias[0]])
        params = torch.einsum('idjt,ikjt->idt', [weight, pre_h_vecs.unsqueeze(1)]) + bias
        F.elu(params)
        for weight, bias in zip(self.params_weights[1:], self.params_bias[1:]):
            _weight = torch.einsum('ijk,jel->ielk',[gen_z, weight])
            _bias = torch.einsum('ijk,jl->ilk',[gen_z, bias])
            params = torch.einsum('idjt,ikjt->idt', [_weight, params.unsqueeze(1)]) + _bias
            F.elu(params)

        mus, logvars = torch.chunk(params, 2, dim=1)
        logvars = logvars.clamp(min=-3,max=1)

#        normals = dist.Normal(mu, cov_tt)
#        return normals.sample(), mu, gen_z

        return mus, logvars, gen_z

    def compute_loss(self, inputs, temp):
        x, y = inputs

        b_vecs = self.inference_pass(y)
        h_vecs, pre_h_vecs = self.forward_pass(x)
        pre_h_vecs = pre_h_vecs[:,:,self.dilations[-1]:]

        p = F.softmax(h_vecs[:,:-1,:]/self.hidden_temp + 1e-6, 1)
        if self.inertia:
            _p, pre_z = regime_dynamics(p, h_vecs[:,-1,:], temp)
        else:
            _p = p

        #h_inf = torch.cat((h_vecs, b_vecs), 1)
        #h_inf = h_vecs+b_vecs
        h_inf = b_vecs

        q = F.softmax(h_inf[:,:-1,:]/self.hidden_temp + 1e-6, 1)
        if self.inertia:
            _q, post_z = regime_dynamics(q, h_inf[:,-1,:], temp)
        else:
            _q = q
            #post_z = gumbel_softmax_sample(torch.log(q).permute(0,2,1), temp).permute(0,2,1)
            post_z = gumbel_softmax(torch.log(q + 1e-6).permute(0,2,1), temp).permute(0,2,1)


        weight = torch.einsum('ijk,jel->ielk',[post_z, self.params_weights[0]])
        bias = torch.einsum('ijk,jl->ilk',[post_z, self.params_bias[0]])
        params = torch.einsum('idjt,ikjt->idt', [weight, pre_h_vecs.unsqueeze(1)]) + bias
        F.elu(params)
        for weight, bias in zip(self.params_weights[1:], self.params_bias[1:]):
            _weight = torch.einsum('ijk,jel->ielk',[post_z, weight])
            _bias = torch.einsum('ijk,jl->ilk',[post_z, bias])
            params = torch.einsum('idjt,ikjt->idt', [_weight, params.unsqueeze(1)]) + _bias
            F.elu(params)

        mus, logvars = torch.chunk(params, 2, dim=1)
        logvars = logvars.clamp(min=-3,max=1)

        logl = 0
        for dim in range(self.n_dims):
            logl = logl + loglikelihood(mus[:,dim,:].unsqueeze(1),
                                        logvars[:,dim,:].unsqueeze(1),
                                        y[:,dim,:].unsqueeze(1))

        return (
                torch.mean(logl),
                -categorical_kld(_q, _p),
               )

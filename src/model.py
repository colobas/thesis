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
    t = p * (logp - logq)
    return t.sum().sum()

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


def regime_dynamics(logp, inert, temp):
    #z = torch.zeros_like(p).cuda()
    z = []
    #new_p = torch.zeros_like(p).cuda()
    new_logp = []

    p = torch.exp(logp)

    if type(inert) != float:
        inert = (0.5 + 0.5 * torch.tanh(inert).unsqueeze(1)).clamp(min=0.01, max=0.99)
    else:
        inert = torch.Tensor([inert]).cuda()

    z.append(gumbel_softmax(logp[:,:,0].unsqueeze(2).permute(0,2,1), temp).permute(0,2,1))
    new_logp.append(logp[:,:,0].clone().unsqueeze(2))
    for t in range(1,p.shape[-1]):
        if inert.shape == torch.Size([1]):
            new_logp.append(
                (torch.log(1-inert) + logp[:,:,t] + torch.log(1 + (inert*(new_logp[t-1].exp()).squeeze())/((1-inert))))\
                        .unsqueeze(2)
            )
        else:
            new_logp.append(
                    (torch.log(1-inert[:,:,t]) + logp[:,:,t] + torch.log(1 + (inert[:,:,t]*(new_logp[t-1].exp()).squeeze())/((1-inert[:,:,t]))))\
                        .unsqueeze(2)
            )

        #z.append(gumbel_softmax(new_logp[t].permute(0,2,1), temp).permute(0,2,1))

    new_logp = torch.cat(new_logp, dim=2)
    #z = torch.cat(z, dim=2)
    z = gumbel_softmax(new_logp.permute(0,2,1), temp).permute(0,2,1)

    return new_logp, z

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
        self.inertia = inertia

        if self.inertia == True:
            convs_final_size = n_regimes + 1
        else:
            convs_final_size = n_regimes

        self.params_net_layers = [convs_final_size]+ params_net_layers

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
        self.convs = nn.ModuleList(self.convs)

        self.fw_prob_conv = nn.Conv1d(
                bottlenecks[-1],
                convs_final_size,
                kernel_size,
                dilation=dilations[-1],
        ).cuda()
        self.fw_pred_conv = nn.Conv1d(
                bottlenecks[-1],
                convs_final_size,
                kernel_size,
                dilation=dilations[-1],
        ).cuda()

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
                convs_final_size,
                kernel_size,
                dilation=dilations[-1],
            ).cuda()
        )
        self.inference_convs = nn.ModuleList(self.inference_convs)

        self.params_weights = nn.ParameterList([
            nn.Parameter(torch.stack([
                torch.rand(self.params_net_layers[i+1], self.params_net_layers[i]).cuda()\
                for _ in range(n_regimes)
            ])) for i in range(len(self.params_net_layers)-1)
        ])
        self.params_bias = nn.ParameterList([
            nn.Parameter(torch.stack([
                torch.rand(self.params_net_layers[i+1]).cuda()\
                for _ in range(n_regimes)
            ])) for i in range(len(self.params_net_layers)-1)
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
        for conv in self.convs:
            x = conv(x)
            x = F.elu(x)

        return F.elu(self.fw_prob_conv(x)), F.elu(self.fw_pred_conv(x))


    def infer(self, y):
        h_vecs, _ = self.forward_pass(y[:,:,:-1])

        b_vecs = self.inference_pass(y[:,:,1:])

        #h_inf = torch.cat((h_vecs, b_vecs), 1)
        #h_inf = h_vecs+b_vecs
        h_inf = b_vecs

        if self.inertia is not None:
            if type(self.inertia) == float:
                logq = F.softmax(h_inf/self.hidden_temp + 1e-6, 1)
                _logq, post_z = regime_dynamics(logq, self.inertia, temp)
            else:
                logq = F.softmax(h_inf[:,:-1,:]/self.hidden_temp + 1e-6, 1)
                _logq, post_z = regime_dynamics(logq, h_inf[:,-1,:], temp)
        else:
            logq = F.softmax(h_inf/self.hidden_temp + 1e-6, 1)
            _logq = logq

        return _logq.exp()

    def gen(self, x):
        h_vecs, h_pred = self.forward_pass(x)

        if self.inertia is not None:
            if type(self.inertia) == float:
                logp = F.log_softmax(h_vecs/self.hidden_temp + 1e-6, 1)
                _logp, gen_z = regime_dynamics(logp, self.inertia, 0.2)
            else:
                logp = F.log_softmax(h_vecs[:,:-1,:]/self.hidden_temp + 1e-6, 1)
                _logp, gen_z = regime_dynamics(logp, h_vecs[:,-1,:], 0.2)
        else:
            logp = F.log_softmax(h_vecs/self.hidden_temp + 1e-6, 1)
            _logp = logp
            gen_z = gumbel_softmax(_logp.permute(0,2,1), 0.2).permute(0,2,1)


        weight = torch.einsum('ijk,jel->ielk',[gen_z, self.params_weights[0]])
        bias = torch.einsum('ijk,jl->ilk',[gen_z, self.params_bias[0]])
        params = torch.einsum('idjt,ikjt->idt', [weight, h_pred.unsqueeze(1)]) + bias
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
        h_vecs, h_pred = self.forward_pass(x)

        if self.inertia is not None:
            if type(self.inertia) == float:
                logp = F.log_softmax(h_vecs/self.hidden_temp + 1e-6, 1)
                _logp, _ = regime_dynamics(logp, self.inertia, temp)
            else:
                logp = F.log_softmax(h_vecs[:,:-1,:]/self.hidden_temp + 1e-6, 1)
                _logp, _ = regime_dynamics(logp, h_vecs[:,-1,:], temp)
        else:
            logp = F.log_softmax(h_vecs/self.hidden_temp + 1e-6, 1)
            _logp = logp


        #h_inf = torch.cat((h_vecs, b_vecs), 1)
        #h_inf = h_vecs+b_vecs
        h_inf = b_vecs

        if self.inertia is not None:
            if type(self.inertia) == float:
                logq = F.softmax(h_inf/self.hidden_temp + 1e-6, 1)
                _logq, post_z = regime_dynamics(logq, self.inertia, temp)
            else:
                logq = F.softmax(h_inf[:,:-1,:]/self.hidden_temp + 1e-6, 1)
                _logq, post_z = regime_dynamics(logq, h_inf[:,-1,:], temp)
        else:
            logq = F.softmax(h_inf/self.hidden_temp + 1e-6, 1)
            _logq = logq
            #post_z = gumbel_softmax_sample(torch.log(q).permute(0,2,1), temp).permute(0,2,1)
            post_z = gumbel_softmax(_logq.permute(0,2,1), temp).permute(0,2,1)

        weight = torch.einsum('ijk,jel->ielk',[post_z, self.params_weights[0]])
        bias = torch.einsum('ijk,jl->ilk',[post_z, self.params_bias[0]])
        params = torch.einsum('idjt,ikjt->idt', [weight, h_pred.unsqueeze(1)]) + bias
        F.elu(params)
        for weight, bias in zip(self.params_weights[1:], self.params_bias[1:]):
            _weight = torch.einsum('ijk,jel->ielk',[post_z, weight])
            _bias = torch.einsum('ijk,jl->ilk',[post_z, bias])
            params = torch.einsum('idjt,ikjt->idt', [_weight, params.unsqueeze(1)]) + _bias
            F.elu(params)

        mus, logvars = torch.chunk(params, 2, dim=1)
        logvars = logvars.clamp(min=-3,max=0)

        logl = 0
        for dim in range(self.n_dims):
            logl = logl + loglikelihood(mus[:,dim,:].unsqueeze(1),
                                        logvars[:,dim,:].unsqueeze(1),
                                        y[:,dim,:].unsqueeze(1))

        return (
                torch.mean(logl),
                -categorical_kld(_logq, _logp),
               )

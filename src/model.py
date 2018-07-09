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

def gaussian_loglikelihood(mus, sigmas, preds):
    return torch.sum(-0.5 * torch.log(2 * np.pi * sigmas**2) - (1/(2*(sigmas**2)))*(mus-preds)**2 , 2)

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class Model(nn.Module):
    def __init__(self, n_layers, n_dims, bottlenecks, n_regimes,
                dilations, kernel_size):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.n_dims = n_dims
        self.bottlenecks = bottlenecks
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.n_regimes = n_regimes

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

        self.final_infer_conv1 = nn.Conv1d(
            n_regimes*2,
            n_regimes*2,
            1,
        )
        self.final_infer_conv2 = nn.Conv1d(
            n_regimes*2,
            n_regimes,
            1,
        )

        self.params_conv1 = nn.Conv1d(
            n_regimes*2,
            n_regimes*2,
            1
        )
        self.params_conv2 = nn.Conv1d(
            n_regimes*2,
            2 * n_dims,
            1
        )

    def inference_pass(self, y):
        y = F.pad(y, (sum(self.dilations), 0, 0, 0))
        for conv in self.inference_convs:
            y = conv(y)
            y = F.tanh(y)
        y = flip(y,2)
        return y

    def forward_pass(self, x):
        x = F.pad(x, (sum(self.dilations), 0, 0, 0))
        for conv in self.convs:
            x = conv(x)
            x = F.tanh(x)
        return x

    def infer(self, y):
        h_vecs = self.forward_pass(y[:,:,:-1])

        b_vecs = self.inference_pass(y[:,:,1:])

        h_inf = torch.cat((h_vecs, b_vecs), 1)
        h_inf = self.final_infer_conv1(h_inf)
        h_inf = F.relu(h_inf)
        h_inf = self.final_infer_conv2(h_inf)
        h_inf = F.relu(h_inf)


        q = F.softmax(h_inf, 1)
#        q = q + 0.00001
#        q = F.softmax(q, 1)
        return q

    def gen(self, x):
        h_vecs = self.forward_pass(x)
        p = F.softmax(h_vecs, 1)
        gen_z = gumbel_softmax(h_vecs.permute(0,2,1).cuda(), 0.1).permute(0,2,1)
        gaussian_params = self.params_conv1(torch.cat((gen_z, h_vecs), 1))
        gaussian_params = F.relu(gaussian_params)
        gaussian_params = self.params_conv2(gaussian_params)
        mus, sigmas = torch.chunk(gaussian_params, 2, 1)

        mus = F.sigmoid(mus)
        sigmas = F.tanh(sigmas)

        mu = mus[:,:,-1]
        sigma = sigmas[:,:,-1]
        print(sigma)

        normals = dist.Normal(mu, sigma)

        return normals.sample()



    def compute_loss(self, inputs, temp):
        x, y = inputs

        b_vecs = self.inference_pass(y)

        h_vecs = self.forward_pass(x)
        p = F.softmax(h_vecs, 1)
#        p = p + 0.00001
#        p = F.softmax(p, 1)

#        gen_z = gumbel_softmax(h_vecs.permute(0,2,1).cuda(), temp).permute(0,2,1)

        h_inf = torch.cat((h_vecs, b_vecs), 1)
        h_inf = self.final_infer_conv1(h_inf)
        h_inf = F.relu(h_inf)
        h_inf = self.final_infer_conv2(h_inf)

        h_inf = F.relu(h_inf)

        q = F.softmax(h_inf, 1)
#        q = q + 0.00001
#        q = F.softmax(q, 1)

        post_z = gumbel_softmax(q.permute(0,2,1).cuda(), temp).permute(0,2,1)

        gaussian_params = self.params_conv1(torch.cat((post_z, h_vecs), 1))
        gaussian_params = F.relu(gaussian_params)
        gaussian_params = self.params_conv2(gaussian_params)

        mus, sigmas = torch.chunk(gaussian_params, 2, 1)

        mus = F.sigmoid(mus)
        sigmas = F.tanh(sigmas)

#        return p, q, mus, sigmas

        logl = gaussian_loglikelihood(mus, sigmas, y)
        return torch.mean(logl), - categorical_kld(q, p), q, p,  post_z


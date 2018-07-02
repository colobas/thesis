import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

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
            )
        ]
        self.convs += [
            nn.Conv1d(
                bottlenecks[l-1],
                bottlenecks[l],
                kernel_size,
                dilation=dilations[l],
            ) for l in range(1,n_layers - 1)
        ]
        self.convs.append(
            nn.Conv1d(
                bottlenecks[-1],
                n_regimes,
                kernel_size,
                dilation=dilations[-1],
            )
        )

        dilations = list(reversed(dilations))

        self.inference_convs = [
            nn.Conv1d(
                n_dims,
                bottlenecks[0],
                kernel_size,
                dilation=dilations[0],
            )
        ]
        self.inference_convs += [
            nn.Conv1d(
                bottlenecks[l-1],
                bottlenecks[l],
                kernel_size,
                dilation=dilations[l],
            ) for l in range(1,n_layers - 1)
        ]
        self.inference_convs.append(
            nn.Conv1d(
                bottlenecks[-1],
                n_regimes,
                kernel_size,
                dilation=dilations[-1],
            )
        )


    def inference_pass(self, y):
        y = flip(y, 1)
        for conv in self.inference_convs:
            print(conv)
            y = conv(y)

        return flip(y,1)

    def forward_pass(self, x):
        for conv in self.convs:
            print(conv)
            x = conv(x)
        return x

    def forward(self, inputs):
        x, y = inputs

        b_vecs = self.inference_pass(y)
        h_vecs = self.forward_pass(x)

        return h_vecs

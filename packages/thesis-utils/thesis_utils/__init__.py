import datetime
import torch

from .mpl2tensor import figure2tensor
from .radam import RAdam
from .lookahead import Lookahead
from .ranger import Ranger

now_str = lambda : str(datetime.datetime.now()).replace(" ", "__")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def torch_onehot(y, n_cat):
    return (
        torch.zeros(len(y), n_cat)
            .scatter_(1, y.type(torch.LongTensor).unsqueeze(-1), 1)
    )

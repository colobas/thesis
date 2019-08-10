import io

import numpy as np
import torch

import PIL

def figure2tensor(f, dpi=100):
    buf = io.BytesIO()
    f.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)

    pic = PIL.Image.open(buf)

    # the following was adapted from torchvision
    # (https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py)
    # @ commit 0c75d99
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()

    buf.close() # this is not on torchvision

    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


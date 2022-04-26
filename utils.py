import torch
import random
import numpy as np

import copy
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a
    Args:
        seed (int): The desired seed.
    """
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    print("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan/base/LICENSE` for copyright attribution and LICENS
def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj = copy.copy(obj)
        name = obj.pop('name')
        kwargs = obj
    else:
        name = obj
        kwargs = {}

    return (name, kwargs)


def linear_scheduler(optimizer, epochs_warmup, epochs_anneal, verbose=True):
    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:
            return 1.0
        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)

    lr_fn = lambda epoch : lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn, verbose = verbose)


def get_scheduler(optimizer, scheduler):
    name, kwargs = extract_name_kwargs(scheduler)
    kwargs['verbose'] = True

    if name == 'linear':
        return linear_scheduler(optimizer, **kwargs)

    if name == 'step':
        return lr_scheduler.StepLR(optimizer, **kwargs)

    if name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if name == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)

    raise ValueError("Unknown scheduler '%s'" % name)
from os.path import join
import shutil
from pathlib import Path
from copy import deepcopy
import math

import glob
import re
import torch
from torch import nn


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
    

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    torch.save(state, join(save_dir, filename))
    if is_best:
        shutil.copyfile(join(save_dir, filename), join(save_dir, 'model_best.pth.tar'))


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0, average_quant_scales=False):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.device = next(model.parameters()).device
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.overwrite_quant_scales = not average_quant_scales
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            new_keys = False
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            ema_msd = self.ema.state_dict()
            for k, v in self.ema.state_dict().items():
                # for Brevitas activation scales, keep the newest value
                if self.overwrite_quant_scales and '.tensor_quant.scaling_impl.value' in k:
                    scale = 0
                else:
                    scale = d
                if v.dtype.is_floating_point:
                    v = v.to(self.device)
                    v *= scale
                    v += (1. - scale) * msd[k].detach()
            for k, v in msd.items():
                if k not in ema_msd:
                    new_keys = True
                    print('adding', k, 'to ema')
                    ema_msd[k] = v
            if new_keys:
                print('LOADED')
                self.ema.load_state_dict(ema_msd)

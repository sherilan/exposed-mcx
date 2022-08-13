import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import exposed.neural.utils as utils


class DictConcatenator(nn.Module):

    def __init__(self, inputs, axis=-1):
        super().__init__()
        self.keys = sorted(inputs)
        self.shapes = [inputs[k] for k in self.keys]
        self.out_shape = sum(utils.flatdim(shape) for shape in self.shapes)
        self.axis = axis

    def forward(self, x):
        return torch.cat(
            [
                utils.flatten(x[k], shape)
                for k, shape in zip(self.keys, self.shapes)
            ],
            axis=-1
        )

    @classmethod
    def from_spec(cls, observation_spec, keys=None, **kwargs):
        keys = list(observation_spec) if keys is None else keys
        return cls({k: observation_spec[k].shape for k in keys}, **kwargs)


class Normalizer(nn.Module):

    def __init__(self, shape, exp=None, dtype=torch.float64, eps=1e-10):
        super().__init__()
        self.shape = utils.asshape(shape)
        self.dtype = dtype
        self.eps = eps
        self.exp = exp
        self.register_buffer('m_1', torch.zeros(shape, dtype=dtype))
        self.register_buffer('m_2', torch.ones(shape, dtype=dtype))
        self.register_buffer('num', torch.zeros((), dtype=torch.long))

    @property
    def mean(self):
        return self.m_1

    @property
    def var(self):
        return (self.m_2 - self.m_1.square()).clip(0)

    @property
    def std(self):
        return self.var.sqrt()

    def forward(self, x, update=None):
        update = self.training if update is None else update
        if update:
            self.update(x)
        return self.normalize(x)

    def normalize(self, x):
        mean = self.mean.to(x.dtype)
        std = self.std.clip(self.eps).to(x.dtype)
        return (x - mean) / std

    def denormalize(self, x):
        mean = self.mean.to(x.dtype)
        std = self.std.to(x.dtype)
        return x * std + mean

    def update(self, x):
        shape = x.shape[:len(x.shape) - len(self.shape)]
        axis = list(range(len(shape)))
        num_new = self.num + np.product(shape)
        if self.exp is None:
            alpha = (self.num.double() / num_new.double()).float()
        else:
            alpha = self.exp ** np.product(shape)
        m_1 = x.to(self.dtype).mean(axis=axis)
        m_2 = x.to(self.dtype).square().mean(axis=axis)
        self.m_1.mul_(alpha).add_(m_1.mul(1 - alpha))
        self.m_2.mul_(alpha).add_(m_2.mul(1 - alpha))
        self.num.copy_(num_new)


class InputCaster(nn.Module):

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.register_buffer('device_detector', torch.zeros([]))

    @property
    def device(self):
        return self.device_detector.device

    def forward(self, x, dtype=None):
        return torch.as_tensor(x, dtype=dtype or self.dtype, device=self.device)

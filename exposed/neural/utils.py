import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

def conv_out_dim(ipt_dim, kernel, stride, padding, transpose=False):
    if transpose:
        return (ipt_dim - 1) * stride - 2 * padding + (kernel -1) + 1
    else:
        numerator = ipt_dim + 2 * padding - (kernel - 1) - 1
        out_dim = numerator / stride + 1
        return int(np.floor(out_dim))


def conv_out_shape(ipt_dims, kernels, strides, paddings, transpose=False):
    out_dims = list(ipt_dims)
    ndim = len(ipt_dims)
    for kernel, stride, padding in zip(kernels, strides, paddings):
        kernel = [kernel] * ndim if isinstance(kernel, int) else kernel
        stride = [stride] * ndim if isinstance(stride, int) else stride
        padding = [padding] * ndim if isinstance(padding, int) else padding
        for i in range(len(out_dims)):
            out_dims[i] = conv_out_dim(
                ipt_dim=out_dims[i],
                kernel=kernel[i],
                stride=stride[i],
                padding=padding[i],
                transpose=transpose
            )
    return tuple(out_dims)

def asshape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)

def flatdim(shape):
    return int(np.product(shape))

def get_ndim(shape):
    return 1 if isinstance(shape, int) else len(shape)

def flatten(x, shape):
    if isinstance(shape, int):
        return x
    else:
        return x.reshape(x.shape[:len(x.shape)-len(shape)] + (flatdim(shape),))

def unflatten(x, shape):
    if isinstance(shape, int):
        return x
    else:
        return x.reshape(x.shape[:-1] + asshape(shape))

def flatbatch(x, ndim=1):
    batch_shape = x.shape[:-ndim]
    data_shape = x.shape[-ndim:]
    x_flatbatch = x.reshape((-1,) + data_shape)
    return batch_shape, x_flatbatch

def balanced_kl(q, p, alpha):
    return (
              alpha * D.kl_divergence(q.detach(), p) +
        (1 - alpha) * D.kl_divergence(q, p.detach())
    )


class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Exp(nn.Module):

    def forward(self, x):
        return x.exp()

def Activation(activation, **kwargs):
    """
    Creates an nn.Module activation function
    """
    if activation is None:
        return nn.Identity()
    elif isinstance(activation, nn.Module):
        return activation
    elif callable(activation):
        return activation(**kwargs)
    else:
        acts = {
            'linear': nn.Identity,
            'identity': nn.Identity,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'mish': Mish,
            'exp': Exp,
            'softplus': nn.Softplus
        }
        act_key = str(activation).lower()
        if not act_key in acts:
            raise KeyError(f'Bad activation "{activation}"')
        return acts[act_key](**kwargs)

def Optimizer(name, params, **opts):
    optims = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adagrad': torch.optim.Adagrad,
        'rmsprop': torch.optim.RMSprop,
    }
    return optims[name](params, **opts)

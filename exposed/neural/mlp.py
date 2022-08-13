import torch.nn as nn

import exposed.neural.utils as utils


class MLP(nn.Sequential):
    def __init__(self, dims, acts, ipt_shape=None, out_shape=None):
        if len(dims):
            assert len(acts) == len(dims) - 1
            if not ipt_shape is None:
                assert utils.flatdim(ipt_shape) == dims[0]
            else:
                ipt_shape = dims[0]
            if not out_shape is None:
                assert utils.flatdim(out_shape) == dims[-1]
            else:
                out_shape = dims[-1]
        layers = []
        for ipt, out, act in zip(dims, dims[1:], acts):
            layers.append(nn.Linear(ipt, out))
            layers.append(utils.Activation(act))
        super().__init__(*layers)
        self.ipt_shape = utils.asshape(ipt_shape)
        self.out_shape = utils.asshape(out_shape)


    def forward(self, x):
        if not self.ipt_shape is None:
            x = utils.flatten(x, self.ipt_shape)
        x = super().forward(x)
        if not self.out_shape is None:
            x = utils.unflatten(x, self.out_shape)
        return x

    @classmethod
    def simple(
        cls,
        ipt_dim,
        out_dim=1,
        hid_dim=None,
        layers=1,
        hid_act=None,
        out_act=None
    ):
        hid_act = 'elu' if hid_act is None else hid_act
        assert layers >= 1
        hid_dim = ipt_dim if hid_dim is None else hid_dim
        dims = [ipt_dim] + [hid_dim] * (layers - 1) + [out_dim]
        acts = [hid_act] * (layers - 1) + [out_act]
        return cls(dims, acts, ipt_shape=(ipt_dim,), out_shape=(out_dim,))

    @classmethod
    def encoder(
        cls,
        ipt_shape,
        hidden,
        layers,
        act=None,
        out_shape=None,
    ):
        act = 'elu' if act is None else act
        if out_shape is None:
            out_shape = (hidden,) if layers else utils.flatdim(ipt_shape)
        dims = [utils.flatdim(ipt_shape)] + [hidden] * layers
        acts = [act] * layers
        return cls(dims, acts, ipt_shape=ipt_shape, out_shape=out_shape)

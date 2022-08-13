import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import exposed.neural.utils as utils

IdentityHead = nn.Identity

class ConstantHead(nn.Module):

    # TODO: add clip_lo / clip_hi ?

    def __init__(self, ipt_shape, out_shape, init=1., scale=None, shift=None, act=None):
        super().__init__()
        self.ipt_shape = utils.asshape(ipt_shape)
        self.out_shape = utils.asshape(out_shape)
        self.scale = scale
        self.shift = shift
        self.act = utils.Activation(act)
        self.value = nn.Parameter(
            torch.full(self.out_shape, 1.) *
            torch.tensor(init, dtype=torch.float32) *
            (1. if scale is None else scale) -
            (0. if shift is None else shift)
        )

    def forward(self, x):
        batch_shape = x.shape[:len(x.shape) - len(self.ipt_shape)]
        if not x.shape[len(batch_shape):] == self.ipt_shape:
            raise ValueError(f'Bad input shape for constant head: {x.shape}')
        value = self.value
        # if not self.scale is None:
        #     value = value * self.scale
        # if not self.shift is None:
        #     value = value + self.shift  # So wasteful to not just do this in init
        if not self.act is None:
            value = self.act(value)
        value = value.repeat(batch_shape + (1,) * len(self.out_shape))
        return value

class LinearHead(nn.Module):

    # TODO: add clip_lo / clip_hi

    def __init__(self, ipt_shape, out_shape, scale=None, shift=None, act=None):
        super().__init__()
        self.ipt_shape = utils.asshape(ipt_shape)
        self.out_shape = utils.asshape(out_shape)
        self.shift = shift # TODO: just change initialized bias
        self.scale = scale
        self.linear = nn.Linear(
            utils.flatdim(self.ipt_shape), utils.flatdim(self.out_shape)
        )
        if not scale is None:
            self.linear.weight.data *= scale
            self.linear.bias.data *= scale
        if not shift is None:
            self.linear.bias.data += shift
        self.act = utils.Activation(act)

    def forward(self, x):
        x_flat = utils.flatten(x, self.ipt_shape)
        y_flat = self.linear(x_flat)
        y = utils.unflatten(y_flat, self.out_shape)
        # if not self.scale is None:
        #     y = y * self.scale
        # if not self.shift is None:
        #     y = y + self.shift
        if not self.act is None:
            y = self.act(y)
        return y

class TransformedDistribution(D.TransformedDistribution):

    @property
    def mean(self):
        return self.base_dist.mean

class Independent(D.Independent):

    @property
    def mean(self):
        return self.base_dist.mean

class DistributionHead(nn.Module):

    transforms = None
    num_independent = None

    def forward(self, *args, **kwargs):
        # Compute base dist
        dist = self.get_dist(*args, **kwargs)
        # Maybe add transforms and group axes together
        if self.transforms:
            dist = TransformedDistribution(dist, self.transforms)
        if self.num_independent:
            dist = Independent(dist, self.num_independent)
        return dist

    def get_dist(self, *args, **kwargs):
        raise NotImplementedError()


class GaussianHead(DistributionHead):

    def __init__(
        self,
        ipt_shape,
        out_shape,
        conditional=False,
        mean_scale=None,
        mean_shift=None,
        mean_act=None,
        std_scale=None,
        std_shift=None,
        std_act='softplus',
        out_shift=None,
        out_scale=None,
    ):
        super().__init__()
        self.out_shift = out_shift
        self.out_scale = out_scale
        self.mean = LinearHead(
            ipt_shape=ipt_shape,
            out_shape=out_shape,
            scale=mean_scale,
            shift=mean_shift,
            act=mean_act,
        )
        self.std = (LinearHead if conditional else ConstantHead)(
            ipt_shape=ipt_shape,
            out_shape=out_shape,
            scale=std_scale,
            shift=std_shift,
            act=std_act,
        )
        self.transforms = self.get_transforms()
        self.num_independent = len(utils.asshape(out_shape))

    def get_dist(self, x):
        try:
            return D.Normal(
                loc=self.mean(x),
                scale=self.std(x),
            )
        except ValueError as e:
            print(e)
            import pdb; pdb.set_trace()

    def get_transforms(self):
        transforms = []
        if not self.out_shift is None or not self.out_scale is None:
            transforms.append(
                D.transforms.AffineTransform(
                    loc=0 if self.out_shift is None else self.out_shift,
                    scale=1 if self.out_scale is None else self.out_scale,
                )
            )
        return transforms


class TanhGaussianHead(GaussianHead):

    def get_transforms(self):
        return [D.transforms.TanhTransform()] + super().get_transforms()


def get_head(kind, **kwargs):
    heads = {
        'idenity': IdentityHead,
        'linear': LinearHead,
        'gaussian': GaussianHead,
        'tanh_gaussian': TanhGaussianHead,
    }
    try:
        Head = heads[kind]
    except KeyError:
        raise KeyError(f'Unknown config head kind: {kind}')
    return Head(**kwargs)

class MultiHead(nn.Module):

    class MultiDist(D.Distribution):

        arg_constraints = {}

        def __init__(self, dists):
            assert isinstance(dists, dict) and len(dists) > 0
            assert all(isinstance(d, D.Distribution) for d in dists.values())
            d0 = next(iter(dists.values()))
            batch_shape = d0.batch_shape
            event_shape = {
                dist: dist.event_shape for name, dist in dists.items()
            }
            super().__init__(batch_shape=batch_shape, event_shape=event_shape)
            self.dists = dists

        def __getitem__(self, key):
            return self.dists[key]

        def entropy(self, reduce=True):
            entropies = {
                name: dist.entropy() for name, dist in self.dists.items()
            }
            return sum(entropies.values()) if reduce else entropies

        def log_prob(self, value, reduce=True):
            log_probs = {
                name: dist.log_prob(value[name])
                for name, dist in self.dists.items()
            }
            return sum(log_probs.values()) if reduce else log_probs

        def sample(self, sample_shape=torch.Size(), greedy=False):
            sample = {}
            for name, dist in self.dists.items():
                # TODO: ovveride all sub-dists and implement greedy sampling
                if not greedy:
                    sample[name] = dist.sample(sample_shape=sample_shape)
                elif hasattr(dist, 'mean'):
                    sample[name] = dist.mean.detach()
                elif hasattr(dist, 'mode'):
                    sample[name] = dist.mode.detach()
                else:
                    raise Exception('No mean or mode for greedy sample')
            return sample

        def rsample(self, sample_shape=torch.Size()):
            return {
                name: dist.rsample(sample_shape=sample_shape)
                for name, dist in self.dists.items()
            }


    def __init__(self, heads, multi_dist=True, **shared_config):
        super().__init__()
        self.multi_dist = multi_dist
        self.heads = nn.ModuleDict()
        for name, head in heads.items():
            if isinstance(head, dict):
                config = {**shared_config, **head}
                kind = config.pop('kind')
                self.heads[name] = get_head(kind, **config)
            else:
                self.heads[name] = head

    def forward(self, x):
        outs = {name: head(x) for name, head in self.heads.items()}
        return self.MultiDist(outs) if self.multi_dist else outs

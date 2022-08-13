import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import exposed.neural.utils as utils

class RNN(nn.Module):

    def __init__(self, ipt_shape):
        super().__init__()
        self.ipt_shape = utils.asshape(ipt_shape)

    @property
    def ipt_ndim(self):
        return len(self.ipt_shape)

    @property
    def out_shape(self):
        raise NotImplementedError()

    def forward(self, x, h=None):
        # Make sure x, h are at least rank3 tensors
        if x.ndim == self.ipt_ndim:
            # Predict on single element
            return self.predict_single(x, h=h)
        else:
            # Predict batch on full sequence
            return self.predict_batch(x, h=h)

    def get_h0(self, x=None):
        return None

    def predict_single(self, x, h=None):
        x = x[None, None]
        h = h if h is None else h[None, None]
        x, h = self.predict_batch(x, h)
        x = x[0, 0]
        h = h if h is None else h[0, 0]
        return x, h

    def predict_batch(self, x, h=None):
        h0 = self.get_h0(x) if h is None else h
        return self.rnn_forward(x, h)

    def rnn_forward(self, x, h):
        raise NotImplementedError()

class Dummy(RNN):

    def __init__(self, ipt_shape, *args, **kwargs):
        super().__init__(ipt_shape)
        self.args = args
        self.kwargs = kwargs

    @property
    def out_shape(self):
        return self.ipt_shape

    def rnn_forward(self, x, h):
        return x, h

class GRU(RNN):

    def __init__(self, ipt_shape, hidden=None, layers=1):
        ipt_shape = utils.asshape(ipt_shape)
        assert len(ipt_shape) == 1
        assert layers >= 1
        super().__init__(ipt_shape)
        self.hidden = ipt_shape[0] if hidden is None else hidden
        self.gru = nn.GRU(ipt_shape[0], self.hidden, layers)

    @property
    def out_shape(self):
        return (self.hidden,)

    def rnn_forward(self, x, h):
        return self.gru(x, h)

class LSTM(RNN):

    def __init__(self, ipt_shape, hidden=None, layers=1):
        ipt_shape = utils.asshape(ipt_shape)
        assert len(ipt_shape) == 1
        assert layers >= 1
        super().__init__(ipt_shape)
        self.hidden = ipt_shape[0] if hidden is None else hidden
        self.lstm = nn.LSTM(ipt_shape[0], self.hidden, layers)

    @property
    def out_shape(self):
        return (self.hidden,)

    def rnn_forward(self, x, h):
        return self.lstm(x, h)

    def predict_single(self, x, h=None):
        x = x[None, None]
        if h is None:
            hc = None
        else:
            hc = (h[0][None, None], h[1][None, None])

        x, (h, c) = self.predict_batch(x, hc)
        return x[0, 0], (h[0, 0], c[0, 0])

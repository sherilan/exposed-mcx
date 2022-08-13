

import numpy as np
import scipy.signal as signal



class Buffer:

    def __init__(self, size):
        self.size = size
        self.data = [None] * size
        self.ptr = 0
        self.num = 0

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self[:][key]
        values = self.data[self.ptr:self.num] + self.data[:self.ptr]
        return values[key]

    def __call__(self, x):
        return self.push(x)

    def push(self, x):
        try:
            self.data[self.ptr] = x
            self.num += 1
            self.ptr = self.num % self.size
        except IndexError as e:
            print(e, self.data, self.ptr, x)
            raise e
        return self.data[0] if self.num < self.size else self.data[self.ptr]


class Butterworth:
    """
    Online butterworth filtering
    """

    def __init__(self, cutoff, order=4):
        self.xs = Buffer(order)
        self.ys = Buffer(order)
        self.cutoff = cutoff  # Nyquist normalized
        self.order = order
        self.b, self.a = signal.butter(
            N=order, Wn=cutoff, btype='low', analog=False, output='ba'
        )

    def __call__(self, x):
        # Initialize pad with initial value
        while self.xs.num < self.order:
            self.xs.push(x)
            self.ys.push(x)
        # Do online butterworth filtering
        b0 = self.b[0]
        b, a = self.b[1:self.xs.num + 1], self.a[1:self.xs.num + 1]
        y = (
            b0 * x +
            np.einsum('k, k ... -> ...', b, self.xs[::-1]) -
            np.einsum('k, k ... -> ...', a, self.ys[::-1])
        )
        self.xs.push(x)
        self.ys.push(y)
        return y


class FiniteDifference:
    """
    Online finite difference differentiator
    """

    def __init__(self, dt, init_val=0.):
        self.dt = dt
        self.prev = None
        self.init_val = 0

    def __call__(self, x):
        if self.prev is None:
            x_dot = np.full_like(x, self.init_val)
        else:
            x_dot = (x - self.prev) / self.dt
        self.prev = x
        return x_dot

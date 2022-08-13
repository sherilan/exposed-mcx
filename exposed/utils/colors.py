import numpy as np

class RGBA:

    R, G, B, A = 0, 1, 2, 3

    def __new__(cls, *data):
        if len(data) == 1:
            try:
                rgba = tuple(data[0])
            except TypeError:
                raise ValueError('Single argument RGBA values must be iterable')
        else:
            rgba = data

        if len(rgba) == 3:
            rgba = rgba + (1.0,)


        if not len(rgba) == 4:
            raise ValueError('RGBA argument must be 3 or 4 values')

        try:
            rgba = tuple(map(float, rgba))
        except ValueError:
            raise ValueError('RGBA arguments must all be numeric')

        if not all(0 <= c <= 1 for c in rgba):
            raise ValueError('RGBA arguments must be inside [0, 1]')

        return rgba

    @classmethod
    def black(cls, alpha=1, strength=1):
        v = 1 - strength
        return cls(v, v, v, alpha)

    @classmethod
    def white(cls, alpha=1, strength=1):
        v = strength
        return cls(v, v, v, alpha)

    @classmethod
    def red(cls, alpha=1, strength=1):
        return cls(strength, 0, 0, alpha)

    @classmethod
    def green(cls, alpha=1, strength=1):
        return cls(0, strength, 0, alpha)

    @classmethod
    def blue(cls, alpha=1, strength=1):
        return cls(0, 0, strength, alpha)

    @classmethod
    def blend(cls, c1, c2, c1_frac=0.5, alpha=None):
        c1 = np.array(cls(c1))
        c2 = np.array(cls(c2))
        c3 = c1_frac * c1 + (1 - c1_frac) * c2
        if not alpha is None:
            c3[3] = alpha
        return cls(c3)

    @classmethod
    def blend_hsv(cls, c1, c2, c1_frac=0.5, alpha=None):

        hsva1 = cls.to_hsva(c1)
        hsva2 = cls.to_hsva(c2)

        h1, h2 = hsva1[0], hsva2[0]

        if h2 < h1 and h1 - h2 > 0.5:
            # Case: shortest circle path crosses modulo 1 (positive dir)
            h = (c1_frac * h1 + (1 - c1_frac) * (h2 + 1)) % 1
        elif h2 > h1 and h2 - h1 > 0.5:
            # Case: shortest circle path crosses modulo 1 (negative dir)
            h = (c1_frac * (h1 + 1) + (1 - c1_frac) * h2) % 1
        else:
            # All good, no modulo line is being crossed
            h = c1_frac * h1 + (1 - c1_frac) * h2

        s = c1_frac * hsva1[1] + (1 - c1_frac) * hsva2[1]
        v = c1_frac * hsva1[2] + (1 - c1_frac) * hsva2[2]
        if alpha is None:
            a = c1_frac * hsva1[3] + (1 - c1_frac) * hsva2[3]
        else:
            a = alpha

        return cls.from_hsva([h, s, v, a])

    @classmethod
    def to_hsva(cls, rgba):
        rgba = r, g, b, a = cls(rgba)
        c_argmax = np.argmax([r, g, b])
        c_max = max(r, g, b)
        c_min = min(r, g, b)
        delta = c_max - c_min
        if delta < 1e-10:
            h = 0
        elif c_argmax == 0: # R
            h = (((g - b) / delta) % 6) / 6
        elif c_argmax == 1: # G
            h = ((b - r) / delta + 2) / 6
        elif c_argmax == 2: # B
            h = ((r - g) / delta + 4) / 6
        else:
            raise Exception('Whoups')
        if c_max < 1e-10:
            s = 0
        else:
            s = delta / c_max
        v = c_max
        return h, s, v, a

    @classmethod
    def from_hsva(cls, hsva):
        hsva = h, s, v, a = cls(hsva)
        c = v * s
        x = c * (1 - abs(((h * 6) % 2) - 1))
        m = v - c
        r, g, b = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
        ][int(h * 6)]
        return cls(r + m, g + m, b + m, a)

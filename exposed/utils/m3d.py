"""
Misc utilities for math in 3D.
"""

import numpy as np
import quaternion as qt

def get_shape(shape=None):
    if shape is None:
        return ()
    elif isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)

def get_random(random=None):
    if random is None:
        return np.random
    elif isinstance(random, int):
        return np.random.RandomState(random)
    else:
        return random

class Quaternion(np.ndarray):

    X, Y, Z = 0, 1, 2

    @property
    def values(self):
        return qt.as_float_array(self)

    @property
    def w(self):
        return self.values[..., 0]

    @property
    def x(self):
        return self.values[..., 1]

    @property
    def y(self):
        return self.values[..., 2]

    @property
    def z(self):
        return self.values[..., 3]

    @property
    def norm(self):
        return np.linalg.norm(self.values, axis=-1)

    @property
    def rmat(self):
        return RotationMatrix.from_quat(self)

    @property
    def normalized(self):
        return self.normalize(self)
    versor = normalized  # A versor is a normalized quaternion

    @property
    def upper_hemisphere(self):
        return self.positive_w(self)

    @property
    def angle(self):
        return 2 * np.arccos(self.w)

    def __repr__(self):
        if self.ndim == 0:
            prc = np.get_printoptions()['precision']
            fmt = lambda x: np.format_float_positional(x, prc)
            w, x, y, z = map(fmt, self.values)
            return f'{self.__class__.__name__}(w={w}, x={x}, y={y}, z={z})'
        else:
            return f'{self.__class__.__name__}(shape={self.shape})'

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, *args, **kwargs):
        return Quaternion(super().__getitem__(*args, **kwargs))

    def __new__(cls, *data):
        if len(data) == 1:
            data = np.asarray(data[0])
            if isinstance(data, Quaternion):
                return data
            if (
                isinstance(data, qt.quaternion) or
                isinstance(data, np.ndarray) and data.dtype == qt.quaternion
            ):
                q = data
            elif data.ndim >= 1 and data.shape[-1] == 4:
                q = qt.from_float_array(data)
            else:
                raise ValueError(
                    'Single quat argument must be a quaternion or an array-like '
                    'with 4 elements in the last axis.'
                )
        elif len(data) == 4:
            q = qt.from_float_array(np.stack(data, axis=-1))
        elif len(data) == 0:
            q = qt.from_float_array([1, 0, 0, 0])
        else:
            raise ValueError(
                'Quaternion data must be a single quat array or w,x,y,z'
            )
        return np.asarray(q).view(cls)

    def __array_finalize__(self, obj):
        assert obj.dtype == qt.quaternion, obj.dtype

    @classmethod
    def identity(cls, shape=()):
        o = np.ones(shape)
        z = np.zeros(shape)
        return cls(o, z, z, z)

    @classmethod
    def random(cls, shape=(), random=None):
        random = get_random(random)
        roll = random.uniform(0, 2 * np.pi, size=shape)
        pitch = np.arcsin(random.uniform(0, 1, size=shape))
        yaw =  random.uniform(-np.pi, np.pi, size=shape)
        return cls.from_roll_pitch_yaw(roll=roll, pitch=pitch, yaw=yaw)

    @classmethod
    def normalize(cls, q):
        xyzw = Quaternion(q).values
        norm = np.linalg.norm(xyzw, keepdims=True, axis=-1)
        return cls(xyzw / np.clip(norm, 1e-15, None))

    @classmethod
    def positive_w(cls, q):
        q = Quaternion(q)
        return cls(q * (1 - 2 * (qt.as_float_array(q)[..., 0] < 0)))

    @classmethod
    def from_rotation_vector(cls, rot_vec):
        return cls(qt.from_rotation_vector(rot_vec))

    @classmethod
    def from_direction_angle(cls, direction, angle):
        norm = np.linalg.norm(direction, axis=-1)
        rot_vec = angle * direction / np.clip(norm, 1e-15, None)
        return cls.from_rotation_vector(rot_vec)

    @classmethod
    def from_axis_angle(cls, angle, axis):
        angle = np.asarray(angle)
        rot_vec = np.zeros(angle.shape + (3,), dtype=angle.dtype)
        rot_vec[..., axis] = angle
        return cls.from_rotation_vector(rot_vec)

    @classmethod
    def from_roll(cls, angle):
        return cls.from_axis_angle(angle, Quaternion.X)

    @classmethod
    def from_pitch(cls, angle):
        return cls.from_axis_angle(angle, Quaternion.Y)

    @classmethod
    def from_yaw(cls, angle):
        return cls.from_axis_angle(angle, Quaternion.Z)

    @classmethod
    def from_roll_pitch_yaw(cls, roll, pitch, yaw):
        qr = Quaternion.from_roll(roll) # First roll around x
        qp = Quaternion.from_pitch(pitch) # Then pitch around (new) y
        qy = Quaternion.from_yaw(yaw) # Then yaw around (new) z
        return cls.positive_w(qy * qp * qr) # times vector to be rotated

    @classmethod
    def from_rpy(cls, rpy):
        rpy = np.asarray(rpy)
        return cls.from_roll_pitch_yaw(rpy[..., 0], rpy[..., 1], rpy[..., 2])

    @classmethod
    def from_rmat(cls, rmat):
        return cls(qt.from_rotation_matrix(RotationMatrix(rmat)))

# class Versor(Quaternion):
#
#     def __new__(cls, *data):
#         print('__new__ versor')
#         return super().__new__(Quaternion, *data).normalized.view(cls)
#
#     def __array_finalize__(self, obj):
#         print('In array finalize', type(self), obj)
#         norm = np.linalg.norm(obj.values)
#

class Vector3(np.ndarray):

    def __new__(cls, *data):
        if len(data) == 1:
            data = data[0]
            if isinstance(data, Vector3):
                return data
            data = np.array(data)
            if not data.ndim >= 1:
                raise ValueError('Vector3 data must be at least rank 1')
            if not data.shape[-1] == 3:
                raise ValueError('Vector3 data must have shape (..., 3)')
        elif len(data) == 3:
            data = np.stack(data, axis=-1)
        elif len(data) == 0:
            data = np.zeros(3)
        else:
            raise ValueError(
                'Vector3 argument must be an array where the last axis is 3d '
                'or three arguments (x, y, z)'
            )
        return np.asarray(data, dtype=np.float64).view(cls)

    @property
    def values(self):
        return np.asarray(self)

    @property
    def norm(self):
        return np.linalg.norm(self, ord=2, axis=-1)
    length = norm

    @property
    def normalized(self):
        norm = np.linalg.norm(self, ord=2, axis=-1, keepdims=True)
        return self / np.clip(norm, 1e-15, None)
    unit = normalized

    def cross(self, other):
        return np.cross(self, Vector3(other), axis=-1).view(type(self))

    def dot(self, other):
        return np.einsum('... C, ... C -> ...', self, Vector3(other))

    def __matmul__(self, other):
        if isinstance(other, Vector3):
            return self.dot(other)
        else:
            raise ValueError('Matmul operator only works between two Vector3')

    def __array_finalize__(self, obj):
        assert obj.dtype == np.float64
        assert obj.ndim >= 1
        assert obj.shape[-1] == 3

    @classmethod
    def zero(cls, shape=()):
        z = np.zeros(shape)
        return cls(z, z, z)
    identity = zero

    @classmethod
    def random_normal(cls, shape=(), loc=0., scale=1., random=None):
        shape = get_shape(shape)
        random = get_random(random)
        return cls(np.random.normal(loc, scale, size=shape + (3,)))

    @classmethod
    def random_uniform(cls, shape=(), lo=-1, hi=+1):
        shape = get_shape(shape)
        random = get_random(random)
        return cls(np.random.normal(lo, hi, size=shape + (3,)))

    @classmethod
    def from_axis(cls, axis, shape=(), length=1):
        z = np.zeros(shape)
        o = np.ones(shape)
        args = [z, z, z]
        args[axis] = o
        return cls(*args)

    @classmethod
    def from_x(cls, shape=(), length=1):
        return cls.from_axis(0, shape=shape, length=length)

    @classmethod
    def from_y(cls, shape=(), length=1):
        return cls.from_axis(1, shape=shape, length=length)

    @classmethod
    def from_z(cls, shape=(), length=1):
        return cls.from_axis(2, shape=shape, length=length)

    @classmethod
    def from_cross(cls, vec3_a, vec3_b):
        return cls(vec3_a).cross(cls(other))


class RotationMatrix(np.ndarray):

    def __new__(cls, *data):
        if len(data) == 1:
            data = data[0]
            if isinstance(data, RotationMatrix):
                return data
            data = np.array(data)
            if not data.ndim >= 2:
                raise ValueError('Rotation matrix data must be at least rank 2')
            if not data.shape[-2:] == (3, 3):
                raise ValueError('Rotation matrix data must have shape (..., 3, 3)')
        elif len(data) == 0:
            data = np.eye(3)
        else:
            raise ValueError(
                'Rotation matrix argument must be a tensor with shape '
                '(..., 3, 3) if provided.'
            )
        return np.asarray(data, dtype=np.float64).view(cls)

    @property
    def values(self):
        return np.asarray(self)

    @property
    def inverse(self):
        axes = list(range(self.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1] # Transpose last axis (3,3)
        return self.transpose(axes)

    @property
    def quat(self):
        return Quaternion.from_rmat(self)

    @property
    def ortho6(self):
        # See App B in: https://arxiv.org/pdf/1812.07035.pdf
        return np.asarray(self[..., :2]).reshape(self.shape[:-2] + (6,))

    @property
    def trace(self):
        return np.trace(self.values, axis1=-2, axis2=-1)

    @property
    def angle(self):
        return np.arccos(0.5 * (self.trace - 1))

    def transform(self, vec3):
        vec3 = Vector3(vec3)
        return np.einsum('... R C, ... C -> ... R', self, vec3).view(Vector3)

    def __matmul__(self, other):
        if isinstance(other, Vector3):
            return self.transform(other)
        elif isinstance(other, RotationMatrix):
            return np.matmul(self, other)
        else:
            raise TypeError(
                'Rotation Matrix can only be matmull\'ed with another rotation '
                'matrix or a vector.'
            )

    def __add__(self, other):
        raise NotImplementedError()

    def __iadd__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __array_finalize__(self, obj):
        assert obj.dtype == np.float64
        assert obj.ndim >= 2
        assert obj.shape[-2:] == (3, 3)

    # Please don't div either...

    @classmethod
    def identity(cls, shape=()):
        return cls(np.tile(np.eye(3), get_shape(shape) + (1, 1)))

    @classmethod
    def random(cls, shape=(), random=None):
        return cls.from_quat(Quaternion.random(shape, random))

    @classmethod
    def from_quat(cls, q):
        return cls(qt.as_rotation_matrix(Quaternion(q).view(np.ndarray)))

    @classmethod
    def from_columns(cls, c1, c2, c3):
        return cls(np.stack([Vector3(c1), Vector3(c2), Vector3(c3)], axis=-1))

    @classmethod
    def from_rows(cls, r1, r2, r3):
        return cls(np.stack([Vector3(r1), Vector3(r2), Vector3(r3)], axis=-2))

    @classmethod
    def from_ortho6(cls, ortho6):
        a1 = Vector3(ortho6[..., 0::2])
        a2 = Vector3(ortho6[..., 1::2])
        b1 = a1.normalized
        b2 = (a2 - (b1 @ a2)[..., None] * b1).normalized
        b3 = b1.cross(b2)
        return cls.from_columns(b1, b2, b3)


class TransformMatrix(np.ndarray):

    def __new__(cls, *data):
        if len(data) == 1:
            data = data[0]
            if isinstance(data, TransformMatrix):
                return data
            data = np.array(data)
            if not data.ndim >= 2:
                raise ValueError('Transform matrix data must be at least rank 2')
            if not data.shape[-2:] == (4, 4):
                raise ValueError('Rotation matrix data must have shape (..., 4, 4)')
        elif len(data) == 0:
            data = np.eye(4)
        else:
            raise ValueError(
                'Rotation matrix argument must be a tensor with shape '
                '(..., 4, 4) if provided.'
            )
        return np.asarray(data, dtype=np.float64).view(cls)

    @property
    def values(self):
        return np.asarray(self)

    @property
    def rotation(self):
        return RotationMatrix(self[..., :3, :3])
    rmat = rotation

    @property
    def translation(self):
        return Vector3(self[..., :3, 3])
    vec3 = translation

    @property
    def inverse(self):
        return self.from_rmat_vec3(self.rmat.inverse, -self.tran)

    def transform(self, vec3):
        vec3 = Vector3(vec3)
        return self.rotation @ vec3 + self.translation

    def __matmul__(self, other):
        if isinstance(other, Vector3):
            return self.transform(other)
        elif isinstance(other, TransformMatrix):
            return np.matmul(self, other)
        else:
            raise TypeError(
                'Rotation Matrix can only be matmull\'ed with another rotation '
                'matrix or a vector.'
            )

    def __add__(self, other):
        raise NotImplementedError()

    def __iadd__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __imul__(self, other):
        raise NotImplementedError()

    def __array_finalize__(self, obj):
        assert obj.dtype == np.float64
        assert obj.ndim >= 2
        assert obj.shape[-2:] == (4, 4)

    @classmethod
    def identity(cls, shape=()):
        return cls(np.tile(np.eye(4), get_shape(shape) + (1, 1)))

    @classmethod
    def from_rmat_vec3(cls, rmat, vec3):
        rmat = RotationMatrix(rmat)
        vec3 = Vector3(vec3)
        rshape = rmat.shape[:-2]
        vshape = vec3.shape[:-1]
        if rshape != rshape:
            raise ValueError('rmat.shape != vec3.shape')
        data = np.zeros(rshape + (4, 4))
        data[..., :3, :3] = rmat.values
        data[..., :3, 3] = vec3.values
        data[..., 3, 3] = 1.0
        return cls(data)

    @classmethod
    def from_rmat(cls, rmat):
        vec3 = Vector3.identity(rmat.shape[:-2])
        return cls.from_rmat_vec3(rmat, vec3)

    @classmethod
    def from_vec3(cls, vec3):
        rmat = RotationMatrix.identity(vec3.shape[:-1])
        return cls.from_rmat_vec3(rmat, vec3)

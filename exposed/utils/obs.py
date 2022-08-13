import copy

import numpy as np
import dm_control.composer.observation.observable as observable

import exposed.utils.m3d as m3d
import exposed.utils.mj as mj


class Observable(observable.Generic):

    def __init__(self, observe, enabled=True, **kwargs):
        super().__init__(observe, **kwargs)
        self._enabled = enabled # So much work to do this manually


class Position(Observable):

    def __init__(self, obj, frame=None, mapper=None, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj = obj
        self.frame = frame
        self.mapper = lambda x: x if mapper is None else mapper

    def observe(self, physics):
        return self.mapper(
            mj.get_pos_vec3(physics, object=self.obj, frame=self.frame)
        )

class Orientation(Observable):

    def __init__(self, obj, frame=None, rep='rmat', **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj = obj
        self.frame = frame
        self.rep = rep

    @property
    def rep(self):
        return self._rep

    @rep.setter
    def rep(self, rep):
        self._repf = self.get_rep(rep)
        self._rep = rep

    @property
    def repf(self):
        return self._repf

    def observe(self, physics):
        return self.repf(physics, self.obj, self.frame)

    def get_rep(self, rep):
        if callable(rep):
            return rep
        else:
            repfs = {
                'quat': self.rep_quat,
                'rmat': self.rep_rmat,
                'ortho6': self.rep_ortho6
            }
            try:
                return repfs[rep]
            except KeyError:
                raise ValueError(
                    f'`rep` arg must be a callable or among ({list(ornfs)})'
                )

    def rep_quat(self, physics, obj, frame):
        return mj.get_orn_quat(physics, obj, frame).values

    def rep_rmat(self, physics, obj, frame):
        return mj.get_orn_rmat(physics, obj, frame).values.reshape(-1)

    def rep_ortho6(self, physics, obj, frame):
        return mj.get_orn_rmat(physics, obj, frame).ortho6


class Direction(Observable):
    def __init__(self, obj_a, obj_b, frame=None, normalize=False, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj_a = obj_a
        self.obj_b = obj_b
        self.frame = frame
        self.normalize = normalize

    def observe(self, physics):
        return mj.get_dir_vec3(
            physics=physics,
            object_a=self.obj_a,
            object_b=self.obj_b,
            frame=self.frame,
            normalize=self.normalize
        )

class LinearVelocity(Observable):

    def __init__(self, obj, frame=None, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj = obj
        self.frame = frame

    def observe(self, physics):
        return mj.get_lin_vel_vec3(physics, object=self.obj, frame=self.frame)

class AngularVelocity(Observable):

    def __init__(self, obj, frame=None, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj = obj
        self.frame = frame

    def observe(self, physics):
        return mj.get_ang_vel_vec3(physics, object=self.obj, frame=self.frame)


class Distance(Observable):

    def __init__(self, obj_a, obj_b, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj_a = obj_a
        self.obj_b = obj_b

    def observe(self, physics):
        return mj.get_distance(
            physics=physics, object_a=self.obj_a, object_b=self.obj_b
        )

class Angle(Observable):

    def __init__(self, obj_a, obj_b, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj_a = obj_a
        self.obj_b = obj_b

    def observe(self, physics):
        return mj.get_angle(
            physics=physics, object_a=self.obj_a, object_b=self.obj_b
        )

class LinearSpeed(Observable):

    def __init__(self, obj, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj = obj

    def observe(self, physics):
        return mj.get_lin_speed(physics, object=self.obj)


class AngularSpeed(Observable):

    def __init__(self, obj, **kwargs):
        super().__init__(self.observe, **kwargs)
        self.obj = obj

    def observe(self, physics):
        return mj.get_ang_speed(physics, object=self.obj)

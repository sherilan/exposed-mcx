
import dm_control.mujoco.wrapper.mjbindings.constants as mjconstants
import dm_control.utils.inverse_kinematics as inverse_kinematics
import dm_env.specs as specs
import numpy as np

import exposed.utils.m3d as m3d
import exposed.utils.filters as filters



class Action:

    buffer = filters.Buffer(1)

    def __init__(self, enabled=True, delay=0):
        self.enabled = enabled
        self.delay = delay

    def reset(self, physics, random_state):
        self.delay = self.delay

    @property
    def delay(self):
        return len(self.buffer) - 1

    @delay.setter
    def delay(self, delay):
        self.buffer = filters.Buffer(size=delay + 1)

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    'Cannot add attribute %s in configure.' % key
                )
            self.__setattr__(key, value)

    def set(self, physics, action, random_state):
        action = self.buffer.push(action)
        if not action is None:
            self.apply(physics, action, random_state)

    def get_spec(self, physics, name=None):
        raise NotImplementedError()

    def apply(self, physics, action, random_state):
        raise NotImplementedError()


class Dummy(Action):

    def __init__(
        self,
        shape,
        dtype=np.dtype(float),
        minv=-mjconstants.mjMAXVAL,
        maxv=+mjconstants.mjMAXVAL
    ):
        super().__init__(**kwargs)
        self.shape = shape
        self.minv = np.full(self.shape, minv, dtype=dtype)
        self.maxv = np.full(self.shape, maxv, dtype=dtype)

    def get_spec(self, physics, name=None):
        return specs.BoundedArray(
            shape=self.shape,
            dtype=self.dtype,
            minimum=self.minv,
            maximum=self.maxv,
            name=name,
        )

    def apply(self, physics, action, random_state):
        pass # DO NOTHING


class Generic(Action):

    def __init__(self, actuators, **kwargs):
        super().__init__(**kwargs)
        self.actuators = actuators

    def get_spec(self, physics, name=None):
        num_act = len(self.actuators)
        binding = physics.bind(self.actuators)
        is_limited = binding.ctrllimited > 0
        control_range = binding.ctrlrange
        minima = np.full(num_act, -mjconstants.mjMAXVAL, dtype=float)
        maxima = np.full(num_act, +mjconstants.mjMAXVAL, dtype=float)
        minima[is_limited], maxima[is_limited] = control_range[is_limited].T
        return specs.BoundedArray(
            shape=(num_act,),
            dtype=np.dtype(float),
            minimum=minima,
            maximum=maxima,
            name=name,
        )

    def apply(self, physics, action, random_state):
        physics.bind(self.actuators).ctrl[:] = action

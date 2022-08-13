import dm_control.utils.rewards as rewards
import numpy as np

import exposed.utils.m3d as m3d
import exposed.utils.mj as mj


class RewardTransform:

    TRANSFORMS = {}

    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, reward):
        return self.f(reward, *self.args, **self.kwargs)

    @classmethod
    def get(cls, f, *args, **kwargs):
        if callable(f):
            return cls(f, *args, **kwargs)
        elif isinstance(f, str):
            try:
                f = cls.TRANSFORMS[f]
            except KeyError:
                raise KeyError(f'No wave transform called "{f}" is registerd')
            return cls(f, *args, **kwargs)
        elif isinstance(f, dict):
            try:
                name = f['name']
            except KeyError:
                raise KeyError(f'A name must be provided if `f` is given as dict')
            return cls.get(name, *f.get('args', ()), **f.get('kwargs', {}))
        elif isinstance(f, (tuple, list)):
            return cls.get(f[0], *f[1:])
        else:
            raise ValueError(
                'RewardTransform.get argument `f` must be a callable, a string, '
                'a dict, or a tuple where the first argument is a callable or string'
            )

    @staticmethod
    def register(name=None):
        def decorator(f):
            _name = f.__name__ if name is None else name
            assert _name not in RewardTransform.TRANSFORMS
            RewardTransform.TRANSFORMS[_name] = f
            return f
        return decorator

@RewardTransform.register()
def scale(reward, factor=1):
    return reward * factor

@RewardTransform.register()
def shift(reward, factor=0):
    return reward + factor

@RewardTransform.register()
def negate(reward):
    return -reward

@RewardTransform.register()
def sigmoid(reward, value_at_1=None):
    reward = 0.5 + 0.5 * np.tanh(0.5 * reward)
    if not value_at_1 is None:
        reward = reward * value_at_1 / 0.7310585786300049
    return reward

@RewardTransform.register()
def tanh(reward, value_at_1=None):
    reward = np.tanh(reward)
    if not value_at_1 is None:
        reward = reward * value_at_1 / 0.7615941559557649
    return reward

@RewardTransform.register()
def clip(reward, lo=None, hi=None):
    if lo is None and hi is None:
        return reward
    return np.clip(reward, lo, hi)


class Reward:

    def __init__(self, transforms=[], enabled=True):
        self.transforms = transforms
        self.enabled = enabled

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        self._transforms = [RewardTransform.get(t) for t in transforms]

    def before_step(self, physics, action, random_state):
        pass

    def after_step(self, physics, random_state):
        pass

    def get_reward(self, physics):
        reward = self.get_raw_reward(physics)
        for t in self.transforms:
            reward = t(reward)
        return reward

    def get_raw_reward(self, physics):
        raise NotImplementedError()

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError('Cannot add attribute %s in configure.' % key)
            self.__setattr__(key, value)


class JointVelCost(Reward):

    def __init__(self, joints, cost=0, power=1, **kwargs):
        super().__init__(**kwargs)
        self.joints = joints
        self.cost = cost
        self.power = power

    def get_raw_reward(self, physics):
        qvel = physics.bind(self.joints).qvel
        return -self.cost * sum(abs(qvel) ** self.power)

class TrackingCost(Reward):

    def __init__(self, object, target, pos_cost=1, orn_cost=1, potential=False, power=1, **kwargs):
        super().__init__(**kwargs)
        self.object = object
        self.target = target
        self.pos_cost = pos_cost
        self.orn_cost = orn_cost
        self.potential = potential
        self.power = power
        self.dist_before = None
        self.angle_before = None
        self.dist_after = None
        self.angle_after = None

    def before_step(self, physics, action, random_state):
        if not self.potential:
            return
        self.dist_before = mj.get_distance(physics, self.object, self.target)
        self.angle_before = mj.get_angle(physics, self.object, self.target)

    def after_step(self, physics, random_state):
        self.dist_after = mj.get_distance(physics, self.object, self.target)
        self.angle_after = mj.get_angle(physics, self.object, self.target)

    def get_raw_reward(self, physics):
        dist = self.dist_after
        angle = self.angle_after
        if self.potential:
            dist -= self.dist_before
            angle -= self.angle_before
        return (
            -self.pos_cost * (dist ** self.power)
            -self.orn_cost * (angle ** self.power)
        )


class PoseChangeCost(Reward):
    """
    Adds a cost to changes in a pose from step to step

    Args:
        body (mjcf.Body): body to calculate pose for
        pos_cost (float): cost for changes in position (L2 norm)
        orn_cost (float): cost for changes in orientation (angular norm)
    """

    # TODO: implement
    pass

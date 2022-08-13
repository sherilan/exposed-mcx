import importlib

import dm_control.composer as composer
import numpy as np

from exposed.tasks.base import BaseTask
from exposed.tasks.v2v import V2VTask


def get_env(name, func=None, time_limit=None, random_state=None, opts=None):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    if time_limit is None:
        time_limit = float('inf')
    if func is None:
        try:
            name, func = name.split(':')
        except ValueError:
            raise ValueError(f'Bad env name "{name}"')
    try:
        module = importlib.import_module('.' + name, package=__package__)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(name, package=__package__)
        except ModuleNotFoundError:
            raise ValueError(f'Could not import module: "{name}"')
    try:
        task = getattr(module, func)
    except AttributeError:
        raise ValueError(f'Module "{name}" has no constructor func "{func}"')
    return composer.Environment(
        task(**(opts or {})), time_limit=time_limit, random_state=random_state
    )

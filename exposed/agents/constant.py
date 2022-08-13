import time
import numpy as np
import pandas as pd

import exposed.tasks as tasks

class ConstantAgent:

    def __init__(self, action_spec, value):
        self.value = value
        self.action = {
            name: np.full((spec.shape), value)
            for name, spec in action_spec.items()
        }

    def get_policy(self, greedy=False):
        return lambda time_step: self.action


def load(env, params=None):

    cfg = {'value': 0} # TODO: load from params if available
    return ConstantAgent(action_spec=env.action_spec(), **cfg)



# python evaluate.py --task target_tracking:hexa_data_v1 --agent ppo:PPOAgent --params /path/to/ppo/for/this/task
# python visualize.py target_tracking:hexa_data_v1 --agent ppo:PPOAgent --params /path/to/ppo/for/this/task

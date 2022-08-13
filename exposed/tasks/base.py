import collections

import dm_control.composer as composer

import exposed.entities as entities
import exposed.utils.obs as obs
import exposed.utils.act as act
import exposed.utils.rew as rew


class BaseTask(composer.Task):

    def __init__(
        self,
        arena=None,
        obs=None,
        act=None,
        rew=None,
        episode_length=None,
        time_step=1/50,
        sub_steps=10,
    ):
        self.arena = arena
        self.obs_opts = obs or {}
        self.act_opts = act or {}
        self.rew_opts = rew or {}
        self.episode_length = episode_length or float('inf')
        self.set_timesteps(
            control_timestep=time_step,
            physics_timestep=time_step / sub_steps,
        )

    @property
    def has_invalid_state(self):
        return False

    @property
    def observables(self):
        observables = self.create_observables()
        if observables is None:
            raise ValueError('`create_observables()` returned None object!')
        for key, obs in observables.items():
            defaults = {
                'enabled': obs.enabled or key in self.obs_opts,
                'aggregator': 'mean',
            }
            settings = {**defaults, **self.obs_opts.get(key, {})}
            obs.configure(**settings)
        return observables

    @property
    def actions(self):
        actions = self.create_actions()
        if actions is None:
            raise ValueError('`create_actions()` returned None object!')
        for key, act in actions.items():
            defaults = {
                'enabled': act.enabled or key in self.act_opts,
            }
            settings = {**defaults, **self.act_opts.get(key, {})}
            act.configure(**settings)
        return actions

    @property
    def rewards(self):
        rewards = self.create_rewards()
        if rewards is None:
            raise ValueError('`create_rewards()` returned None object!')
        for key, reward in rewards.items():
            defaults = {
                'enabled': reward.enabled or key in self.rew_opts
            }
            settings = {**defaults, **self.rew_opts.get(key, {})}
            reward.configure(**settings)
        return rewards


    def create_observables(self):
        return collections.OrderedDict()

    def create_actions(self):
        return collections.OrderedDict()

    def create_rewards(self):
        return collections.OrderedDict()


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #  Defaults for dm_control composer.Task hooks
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    @property
    def root_entity(self):
        return self.arena

    def after_compile(self, physics, random_state):
        self.enabled_actions = {
            name: action
            for name, action in self.actions.items() if action.enabled
        }
        self.enabled_rewards = {
            name: reward
            for name, reward in self.rewards.items() if reward.enabled
        }

    def action_spec(self, physics):
        """Generates action spec (with dot-notation for nested dict specs)"""
        def dotdict(s, prefix=()):
            if isinstance(s, dict):
                for k, v in s.items():
                    yield from dotdict(v, prefix=prefix + (k,))
            else:
                yield ':'.join(prefix), s
        spec = collections.OrderedDict()
        for name, action in self.enabled_actions.items():
            for full_name, s in dotdict(
                action.get_spec(physics, name), prefix=(name,)
            ):
                spec[full_name] = s
        return spec

    def initialize_episode(self, physics, random_state):
        self.episode_step = 0

    def before_step(self, physics, action, random_state):
        # Unpack dot-notation actions
        formatted_action = {}
        for k, v in action.items():
            node = formatted_action
            parts = k.split(':')
            for part in parts[:-1]:
                node[part] = node = node.get(part, {})
            node[parts[-1]] = v
        # Apply actions
        for name, act in self.enabled_actions.items():
            try:
                values = formatted_action[name]
            except KeyError:
                breakpoint()
                raise KeyError(f'Missing action: {name}')
            act.set(physics, values, random_state)
        # Call reward hooks
        for reward in self.enabled_rewards.values():
            reward.before_step(physics, action, random_state)

    def before_substep(self, physics, action, random_state):
        pass

    def after_step(self, physics, random_state):
        self.episode_step += 1
        for reward in self.enabled_rewards.values():
            reward.after_step(physics, random_state)

    def get_reward(self, physics):
        total_reward = 0
        for reward in self.enabled_rewards.values():
            total_reward += reward.get_reward(physics)
        return total_reward

    def should_terminate_episode(self, physics):
        return self.episode_step >= self.episode_length

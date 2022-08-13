import pathlib
import argparse

import numpy as np
import pandas as pd

import exposed.tasks as tasks
import exposed.agents as agents
import exposed.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('task', type=str)
parser.add_argument('agent', type=str)
parser.add_argument('--exp', type=pathlib.Path)
parser.add_argument('--greedy', action='store_true')
parser.add_argument('--metrics', type=str, nargs='+')
parser.add_argument('--seed', type=int)
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--save', type=pathlib.Path)
parser.add_argument('--clean', action='store_true')

def rollout(env, policy, max_steps=None):
    observations = {key: [] for key in env.observation_spec()}
    actions = {key: [] for key in env.action_spec()}
    rewards = []
    time_step = env.reset()
    step = 0
    while True:
        step += 1
        action = policy(time_step)
        for key, obs in time_step.observation.items():
            observations[key].append(obs)
        for key, act in action.items():
            actions[key].append(act)
        time_step = env.step(action)
        rewards.append(time_step.reward) # Only collect rew after first act
        if time_step.step_type == 2:
            break
        if max_steps is not None and step > max_steps:
            break
    for key, obs in time_step.observation.items():
        observations[key].append(obs)
    for key, act in action.items():
        actions[key].append(act)

    observations = {
        key: np.stack(observations[key])
        for key in env.observation_spec()
    }
    actions = {
        key: np.stack(actions[key])
        for key in env.action_spec()
    }
    rewards = np.stack(rewards)
    return observations, actions, rewards


def save_hdf(path, data, clean=False):
    import tables, re
    with tables.open_file(path, mode='w' if clean else 'a') as h5:
        if clean:
            ep0 = 0
        else:
            rgx = re.compile(r'^episode_(\d+)$')
            idx = [-1] + [
                int(idx)
                for node in h5.root
                for idx in rgx.findall(node._v_name)
            ]
            ep0 = max(idx) + 1
        for ep, (observations, actions, rewards) in enumerate(data):
            name = f'episode_{str(ep0 + ep).zfill(5)}'
            node = h5.create_group(h5.root, name)
            obs = h5.create_group(node, 'observations')
            for key in observations:
                h5.create_array(obs, key, observations[key])
            act = h5.create_group(node, 'actions')
            for key in actions:
                h5.create_array(act, key, actions[key])
            h5.create_array(node, 'rewards', rewards)

def main(args):

    env = tasks.get_env(name=args.task, random_state=args.seed)
    exp = None if args.exp is None else utils.exp.Experiment.restore(args.exp)
    agent = agents.get_agent(name=args.agent, env=env, exp=exp)
    policy = agent.get_policy(greedy=args.greedy)
    if args.metrics is None:
        metrics = [
            k for k, v in env.observation_spec().items() if v.shape == ()
        ]
    else:
        metrics = args.metrics

    data = []
    for episode in range(args.episodes):
        print('Episode:', episode)
        observations, actions, rewards = rollout(env, policy, max_steps=args.max_steps)
        data.append((observations, actions, rewards))
        summary = pd.DataFrame()
        summary['reward'] = pd.Series(rewards).describe()
        for m in metrics:
            summary[m] = pd.Series(observations[m]).describe()
        print(summary.T)

    if args.save:
        if args.save.is_dir():
            path = args.save / f'{args.task}.h5'
        else:
            path = args.save
        save_hdf(path, data, clean=args.clean)


if __name__ == '__main__':
    main(parser.parse_args())


# python evaluate.py v2v:hs10_delay_0 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs15_delay_0 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs20_delay_0 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs25_delay_0 constant --episodes 10 --save output/constant
#
# python evaluate.py v2v:hs10_delay_50 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs15_delay_50 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs20_delay_50 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs25_delay_50 constant --episodes 10 --save output/constant
#
# python evaluate.py v2v:hs10_delay_100 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs15_delay_100 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs20_delay_100 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs25_delay_100 constant --episodes 10 --save output/constant
#
# python evaluate.py v2v:hs10_smooth_4_33 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs15_smooth_4_33 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs20_smooth_4_33 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs25_smooth_4_33 constant --episodes 10 --save output/constant
#
# python evaluate.py v2v:hs10_smooth_4_17 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs15_smooth_4_17 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs20_smooth_4_17 constant --episodes 10 --save output/constant
# python evaluate.py v2v:hs25_smooth_4_17 constant --episodes 10 --save output/constant

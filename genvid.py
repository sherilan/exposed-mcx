import argparse
import pathlib
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import moviepy.editor
import moviepy.video.io.bindings

import exposed.tasks as tasks
import exposed.agents as agents
import exposed.utils.exp as experiment

parser = argparse.ArgumentParser()
parser.add_argument('task', type=str)
parser.add_argument('--agents', type=str, nargs='+')
parser.add_argument('--greedy', action='store_true')
parser.add_argument('--metrics', type=str, nargs='+', default=['reward'])
parser.add_argument('--seed', type=int, default=np.random.randint(1<<32))
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--max-steps', type=int, default=1000)
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--fps', type=int, default=50)
parser.add_argument('--save', type=pathlib.Path, default=pathlib.Path(os.getcwd()))


METRIC_MAP = {
    'norm_pos_G_E': '$\\Delta p$ [m]',
    'norm_orn_G_E': '$\\Delta \\alpha$ [rad]',
    'reward': 'Reward'
}

def main(args):

    names = []
    pis = []
    envs = []

    for agent_name in args.agents:
        print(agent_name)
        env = tasks.get_env(name=args.task, random_state=args.seed)
        name, kind, *rundir = agent_name.split(':')
        rundir, = [None] if not rundir else rundir
        exp = experiment.Experiment.restore(rundir) if rundir else None
        agent = agents.get_agent(name=kind, env=env, exp=exp)
        names.append(name)
        pis.append(agent.get_policy(greedy=args.greedy))
        envs.append(env)

    all_frames = []
    all_mets = []

    for ep in range(args.episodes):
        print(f'Episode {ep}')
        episode_frames = []
        episode_mets = []
        steps = [env.reset() for env in envs]
        while not any(step.step_type == 2 for step in steps):
            steps = [env.step(pi(step)) for env, pi, step in zip(envs, pis, steps)]
            episode_frames.append([env.physics.render() for env in envs])
            episode_mets.append([
                [
                    step.reward if m == 'reward' else step.observation[m]
                    for m in args.metrics
                ]
                for step in steps
            ])
            if args.max_steps > 0 and len(episode_frames) >= args.max_steps:
                break
        all_frames.append(np.array(episode_frames[args.skip:]))
        all_mets.append(np.array(episode_mets[args.skip:]))

    all_frames = np.stack(all_frames)
    all_mets = np.stack(all_mets)

    E = all_frames.shape[0]
    T = all_frames.shape[1]
    N = all_frames.shape[2]
    M = all_mets.shape[-1]

    fig = plt.figure(figsize=(10, 8), dpi=200)
    gs = plt.GridSpec(M + 1, N, height_ratios=[2] + ([1] * M))
    img_axes = [plt.subplot(gs.new_subplotspec((0, n), 1, 1)) for n in range(N)]
    met_axes = [plt.subplot(gs.new_subplotspec((m + 1, 0), 1, N)) for m in range(M)]

    met_los = all_mets.min(axis=(1, 2))
    met_his = all_mets.max(axis=(1, 2))
    met_dis = met_his - met_los

    def render(t):
        idx = int(t * args.fps)
        e, s = idx // T, idx % T
        frames = all_frames[e, s]
        for ax, frame, name, in zip(img_axes, frames, names):
            ax.clear()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(frame)
            ax.set_title(name)

        mets = all_mets[e, :s+1]
        xx = np.arange(s + 1)
        for i, ax in enumerate(met_axes):
            ax.clear()
            ax.set_title(METRIC_MAP.get(args.metrics[i], args.metrics[i]))
            for j, name in enumerate(names):
                mets = all_mets[e, :s+1, j, i]
                ax.plot(xx, mets, label=f'{name} (avg={mets.mean():.4f})')
            ax.legend(loc='upper left')
            ax.set_xlim(0, T)
            ax.set_ylim(
                met_los[e, i] - 0.1 * met_dis[e, i],
                met_his[e, i] + 0.1 * met_dis[e, i]
            )
        fig.subplots_adjust(hspace=0.2, wspace=0.05)
        return moviepy.video.io.bindings.mplfig_to_npimage(fig)

    savepath = args.save / f'{args.task}.mp4' if args.save.is_dir() else args.save
    ani = moviepy.editor.VideoClip(render, duration=(E * T)/args.fps)
    ani.write_videofile(str(savepath), fps=args.fps)


if __name__ == '__main__':
    main(parser.parse_args())

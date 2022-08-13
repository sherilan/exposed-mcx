import collections

import dm_env.specs as specs
import numpy as np
import ray
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

import exposed.tasks as tasks
import exposed.neural as neural

class Base(nn.Module):

    def __init__(
        self,
        inputs,
        hidden_dim=128,
        mlp1_layers=2,
        rnn_layers=0,
        mlp2_layers=0,
        rnn_type='gru',
        activation='elu',
        residual=False,
    ):
        super().__init__()
        self.inputs = inputs
        self.residual = residual
        self.concat = neural.inputs.DictConcatenator(inputs=inputs,)
        self.norm = neural.inputs.Normalizer(shape=self.concat.out_shape)
        self.mlp1 = neural.mlp.MLP.encoder(
            ipt_shape=self.concat.out_shape,
            hidden=hidden_dim,
            layers=mlp1_layers,
            act=activation,
        )
        if rnn_layers == 0:
            self.rnn = neural.rnn.Dummy(
                ipt_shape=self.mlp1.out_shape,
                hidden=hidden_dim,
                layers=rnn_layers,
            )
        elif rnn_type == 'gru':
            self.rnn = neural.rnn.GRU(
                ipt_shape=self.mlp1.out_shape,
                hidden=hidden_dim,
                layers=rnn_layers,
            )
        elif rnn_type == 'lstm':
            self.rnn = neural.rnn.LSTM(
                ipt_shape=self.mlp1.out_shape,
                hidden=hidden_dim,
                layers=rnn_layers,
            )
        else:
            raise ValueError(f'RNN type "{rnn_type}" not understood!')
        self.mlp2 = neural.mlp.MLP.encoder(
            ipt_shape=self.rnn.out_shape,
            hidden=hidden_dim,
            layers=mlp2_layers,
            act=activation,
        )

    def forward(self, x, h=None, update_norm=False):
        x = self.concat(x)
        x = self.norm(x, update=update_norm)
        x = self.mlp1(x)
        y, h = self.rnn(x, h)
        x = (x + y) if self.residual else y
        x = self.mlp2(x)
        return x, h

class Actor(Base):

    def __init__(self, base, heads):
        super().__init__(**base)
        self.head = neural.heads.MultiHead(
            heads={
                name: dict(ipt_shape=self.mlp2.out_shape, **head)
                for name, head in heads.items()
            }
        )

    def forward(self, x, h=None, ret_h=False, **kwargs):
        x, h = super().forward(x, h, **kwargs)
        pi = self.head(x)
        return (pi, h) if ret_h else pi

    def objective(self, obs, act, adv, logp_old, clip=0.2, alpha=0, **kwargs):
        """PPO Policy update clipped objective"""
        pi = self(obs, **kwargs)
        logp_new = pi.log_prob(act)
        ratio = torch.exp(logp_new - logp_old)
        clip_lo, clip_hi = 1 - clip, 1 + clip
        ratio_clipped = torch.clamp(ratio, min=clip_lo, max=clip_hi)
        entropy = -logp_new.mean()
        objective = torch.min(ratio * adv, ratio_clipped * adv)
        loss = - objective.mean() - alpha * entropy
        info = {}
        info['Entropy'] = entropy.detach()
        info['Loss'] = loss.detach()
        info['KL'] = (logp_old - logp_new).detach().mean()
        info['Clipped'] = ((ratio < clip_lo) | (ratio > clip_hi)).float().mean()
        return loss, info


class Critic(Base):

    def __init__(self, base, loss):
        super().__init__(**base)
        self.head = neural.heads.LinearHead(
            ipt_shape=self.mlp2.out_shape,
            out_shape=(),
        )
        self.out_norm = neural.inputs.Normalizer(shape=())
        self.loss = loss

    def forward(self, x, h=None, ret_h=False, update_norm=False, denorm=True):
        x, h = super().forward(x, h, update_norm=update_norm)
        v = self.head(x)
        if denorm:
            v = self.out_norm.denormalize(v)
        return (v, h) if ret_h else v

    def objective(self, obs, ret, update_norm=False):
        """PPO Baseline regression objective"""
        val = self(obs, update_norm=update_norm, denorm=False)
        assert val.shape == ret.shape
        ret = self.out_norm(ret, update=update_norm)
        if self.loss == 'mse':
            loss = F.mse_loss(val, ret)
        elif self.loss == 'huber':
            loss = F.huber_loss(val, ret)
        else:
            raise ValueError(f'Critic loss type "{self.loss}" not understood')
        info = {}
        info['Loss'] = loss.detach()
        info['Value'] = val.mean().detach()
        return loss, info


class PPOAgent(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.to_torch = neural.inputs.InputCaster()
        self.actor = Actor(
            base={**config.get('base', {}), **config.get('actor_base', {})},
            heads=config['heads'],
        )
        self.critic = Critic(
            base={**config.get('base', {}), **config.get('critic_base', {})},
            loss=config.get('critic_loss', 'mse'),
        )
        self.inputs = {**self.actor.inputs, **self.critic.inputs}

    def cpu_state_dict(self):
        device = self.to_torch.device
        self.to('cpu')
        state_dict = self.state_dict()
        self.to(device)
        return state_dict

    def save(self, path):
        torch.save(self.cpu_state_dict(), path)

    def get_policy(self, greedy=False, critic=False):
        h_actor = h_critic = None
        def act(time_step):
            nonlocal h_actor, h_critic
            if time_step.step_type == 0:
                h_actor = h_critic = None
            obs = {
                k: self.to_torch(o)
                for k, o in time_step.observation.items()
                if k in self.actor.inputs
                or critic and k in self.critic.inputs
            }
            with torch.no_grad():
                pi, h_actor = self.actor(obs, h=h_actor, ret_h=True)
            action = {
                k: a.cpu().numpy()
                for k, a in pi.sample(greedy=greedy).items()
            }
            if critic:
                with torch.no_grad():
                    value, h_critic = self.critic(obs, h=h_critic, ret_h=True)
                    value = value.cpu().numpy()
                return action, value
            else:
                return action
        return act

def infer_inputs(obs_spec, inputs_config=None):
    return {
        o: list(spec.shape) for o, spec in obs_spec.items()
        if not inputs_config or o in inputs_config
    }

def infer_heads(act_spec, heads_config=None):
    heads = {}
    for a, spec in act_spec.items():
        heads[a] = head = {}
        if isinstance(spec, specs.DiscreteArray):
            raise NotImplementedError()
        elif isinstance(spec, specs.BoundedArray):
            head['kind'] = 'gaussian'
            head['out_shape'] = spec.shape
        else:
            raise NotImplementedError()
        if heads_config and a in heads_config:
            head.update(heads_config.get(a))
    return heads

def map_dicts(dicts, fn, **kwargs):
    def recurse(d, ds):
        if isinstance(d, dict):
            return {k: recurse(d[k], [d_[k] for d_ in ds]) for k in d}
        else:
            return fn(ds, **kwargs)
    return recurse(dicts[0], dicts)

def stack_dicts(dicts, axis=0):
    return map_dicts(dicts, np.stack, axis=axis)

def concat_dicts(dicts, axis=0):
    return map_dicts(dicts, np.concatenate, axis=axis)

def get_device(device=None):
    import warnings
    if device is None:
        return 'cpu'
    if isinstance(device, int):
        device = f'cuda:{device}'
    if not isinstance(device, str):
        raise Exception('Device must be a string or int (cuda index)')
    if device.startswith('cpu'):
        return torch.device(device)
    elif device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn(
            f'Tried to use cuda device "{device}" but cuda is not available. '
            f'Falling back to CPU.'
        )
        return torch.device('cpu')
    return torch.device(device)


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[:0:-1]

def discounted_return(*, rew, gam, last=0.0):
    rew = np.concatenate([rew, np.zeros_like(rew[:1]) + last])
    return discount_cumsum(rew, discount=gam)

def td_error(*, rew, val, val_next, gam=1.0):
    return rew + gam * val_next - val

def gae(*, rew, val, lam, gam, last=0.0):
    val_next = np.concatenate([val[1:], np.zeros_like(val[:1]) + last])
    deltas = td_error(rew=rew, val=val, val_next=val_next, gam=gam)
    return discounted_return(rew=deltas, gam=lam * gam, last=0)


@ray.remote
class Worker:

    def __init__(self, index, config):
        torch.manual_seed(config['experiment']['seed'] + 100 * index)
        self.index = index
        self.train_env = tasks.get_env(
            config['sampling']['env_name'],
            random_state=config['experiment']['seed'] + 100 * index,
            opts=config['sampling']['env_opts']
        )
        self.eval_env = tasks.get_env(
            config['sampling']['env_name'],
            random_state=config['experiment']['seed'] + 100 * index + 50,
            opts=config['sampling']['env_opts']
        )
        self.agent = PPOAgent(config=config['agent'])
        self.agent.to(config['sampling']['device']).eval()
        self.workers = config['sampling']['workers']
        self.train_trajs = config['sampling'].get('train_trajs') or self.workers
        self.train_traj_len = config['sampling']['train_traj_len']
        self.eval_trajs = config['sampling'].get('eval_trajs') or self.workers
        self.metrics = config['sampling'].get('metrics') or [
            k for k, v in self.train_env.observation_spec().items()
            if v.shape == ()
        ]
        self.min_rew = config['sampling'].get('min_rew', -float('inf'))
        self.lam = config['training']['gae_lam']
        self.gam = config['training']['gamma']

    def sample_train(self, parameters):
        local_trajs = (self.train_trajs + self.index) // self.workers
        if not local_trajs:
            return
        self.agent.load_state_dict(parameters)
        data = []
        rollout = 0
        while rollout < local_trajs:
            policy = self.agent.get_policy(critic=True)
            time_step = self.train_env.reset()
            action, value = policy(time_step)
            ep = []
            for i in range(self.train_traj_len):
                assert not time_step.step_type == 2 # Always rollout to completion
                if self.train_env.task.has_invalid_state:
                    break  # Sometimes (rarely) the wave data results in impossible ik problems
                time_step_next = self.train_env.step(action)
                action_next, value_next = policy(time_step_next)
                ep.append(dict(
                    obs={
                        k:v for k, v in time_step.observation.items()
                        if k in self.agent.inputs
                    },
                    act=action,
                    rew=time_step_next.reward,
                    val=value,
                    met={
                        k:v for k, v in time_step.observation.items()
                        if k in self.metrics
                    },
                ))
                time_step = time_step_next
                action, value = action_next, value_next

            # Handle edge cases where bad vessel motion causes excessive rews
            if len(ep) < self.train_traj_len:
                print('Rejecting traj due to invalid env state')
                continue
            if -(np.mean([e['rew'] ** 2 for e in ep])) < self.min_rew:
                print('Rejecting traj due to excessive rew')
                continue

            rollout += 1
            ep = stack_dicts(ep, axis=0)
            ep['ret'] = discounted_return(
                rew=ep['rew'],
                gam=self.gam,
                last=value_next,
            )
            ep['adv'] = gae(
                rew=ep['rew'],
                val=ep['val'],
                lam=self.lam,
                gam=self.gam,
                last=value_next
            )
            data.append(ep)

        return stack_dicts(data, axis=1) # Keep time axis first


    def sample_eval(self, parameters):
        self.agent.set_state_dict(parameters)


def load(env, exp):
    if not exp:
        raise Exception('PPOAgent needs a pretrained rundir to load from')
    agent = PPOAgent(config=exp.cfg.agent)
    missing_ob = set(agent.inputs) - set(env.observation_spec())
    if missing_ob:
        raise Exception(f'PPOAgent was trained to use missing ob: {missing_ob}')
    missing_ac = set(env.action_spec()) - set(agent.actor.head.heads)
    if missing_ac:
        raise Exception(f'PPOAgent doesn not have action heads for {missing_ac}')
    params = exp.dir / 'params.pt'
    device = torch.device('cpu')
    agent.load_state_dict(torch.load(params, map_location=device))
    return agent

def main():

    import exposed.utils.exp as experiment
    exp = experiment.Experiment.from_cli()

    # Update input and heads config
    with exp.update_config() as cfg:
        env = tasks.get_env(
            exp.cfg.sampling.env_name, opts=exp.cfg.sampling.env_opts
        )
        cfg.agent.actor_base.inputs = infer_inputs(
            obs_spec=env.observation_spec(),
            inputs_config={
                **cfg.agent.base.get('inputs', {}),
                **cfg.agent.actor_base.get('inputs', {})
            }
        )
        cfg.agent.critic_base.inputs = infer_inputs(
            obs_spec=env.observation_spec(),
            inputs_config={
                **cfg.agent.base.get('inputs', {}),
                **cfg.agent.critic_base.get('inputs', {})
            }
        )
        cfg.agent.heads = infer_heads(
            act_spec=env.action_spec(),
            heads_config=cfg.agent.get('heads', {})
        )
        del env  # Only needed to infer shapes

    # Setup agent
    print(exp.cfg)
    torch.manual_seed(exp.cfg.experiment.seed)
    agent = PPOAgent(config=exp.cfg.agent)
    agent.to(get_device(exp.cfg.training.device))
    print(agent)
    # Setup optimizers
    actor_optimizer = neural.utils.Optimizer(
        params=agent.actor.parameters(), **exp.cfg.training.actor_optim
    )
    critic_optimizer = neural.utils.Optimizer(
        params=agent.critic.parameters(), **exp.cfg.training.critic_optim
    )
    # Setup up workers
    ray.init()
    workers = [
        Worker
            .options(num_gpus=1 / (1 + exp.cfg.sampling.workers))
            .remote(i, exp.cfg.as_native_dict())
        for i in range(exp.cfg.sampling.workers)
    ]

    env_steps = 0
    for epoch in exp.log.epochs(exp.cfg.training.epochs):

        # Sample data with multiple workers and stack along batch axis
        params = agent.cpu_state_dict()
        jobs = [w.sample_train.remote(params) for w in workers]
        data = concat_dicts(dicts=[d for d in ray.get(jobs) if d], axis=1)

        # Convert obs/act/adv/ret to torch tensors
        obs = {k: agent.to_torch(v) for k, v in data['obs'].items()}  # T, B, O
        act = {k: agent.to_torch(v) for k, v in data['act'].items()}  # T, B, A
        adv = agent.to_torch(data['adv'])  # T, B
        ret = agent.to_torch(data['ret'])  # T, B

        # Normalized advantages tend to work better
        adv = (adv - adv.mean()) / adv.std().clamp(1e-6)

        # Accumulate env steps
        env_steps += data['rew'].size

        # Train actor
        with torch.no_grad():
            logp = agent.actor(obs).log_prob(act)
        for actor_it in range(exp.cfg.training.actor_its):
            actor_optimizer.zero_grad()
            actor_loss, actor_info = agent.actor.objective(
                obs=obs,
                act=act,
                adv=adv,
                logp_old=logp,
                clip=exp.cfg.training.clip_ratio,
                alpha=exp.cfg.training.ent_alpha,
                update_norm=actor_it == 0,
            )
            if (
                exp.cfg.training.max_kl and
                actor_info['KL'] > 1.5 * exp.cfg.training.max_kl
            ):
                break
            else:
                actor_loss.backward()
                actor_info['grad_norm'] = nn.utils.clip_grad_norm_(
                    parameters=agent.actor.parameters(),
                    max_norm=cfg.training.actor_clip.value,
                    norm_type=cfg.training.actor_clip.type
                )
                actor_optimizer.step()
                actor_it += 1

        # Train critic
        for critic_it in range(exp.cfg.training.critic_its):
            critic_optimizer.zero_grad()
            critic_loss, critic_info = agent.critic.objective(
                obs=obs, ret=ret, update_norm=critic_it == 0,
            )
            critic_loss.backward()
            critic_info['grad_norm'] = nn.utils.clip_grad_norm_(
                parameters=agent.critic.parameters(),
                max_norm=cfg.training.critic_clip.value,
                norm_type=cfg.training.critic_clip.type
            )
            critic_optimizer.step()
            critic_it += 1


        # Log
        with exp.log.prefix('Actor'):
            exp.log << {'Iterations': actor_it}
            exp.log << actor_info
        with exp.log.prefix('Critic'):
            exp.log << critic_info
        with exp.log.prefix('Data'):
            exp.log << {
                'EnvSteps': env_steps,
                'AvgReward': data['rew'].mean(),
                'RawReturn': data['rew'].sum(axis=0).mean(),
                'DiscReturn': data['ret'].mean(),
                'ValuePred': data['val'].mean(),
                **{k: m.mean() for k, m in data['met'].items()}
            }

        # Maybe save model
        if (
            exp.dir and
            exp.cfg.training.save_freq and
            (epoch + 1) % exp.cfg.training.save_freq  == 0
        ):
            torch.save(agent.cpu_state_dict(), exp.dir / 'params.pt')
            exp.log('Model parameters saved!')



if __name__ == '__main__':
     main()

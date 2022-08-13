import argparse
import contextlib
import pathlib
import re
import sys

import yaml
import numpy as np
import torch

class Experiment:

    def __init__(self, cfg=None, dir=None, run=None, log=None):
        self.cfg = Config(cfg or {})
        self.dir = None if dir is None else pathlib.Path(dir)
        self.run = run
        self.log = log or Logger()

    @contextlib.contextmanager
    def update_config(self):
        with self.cfg.unfreeze() as cfg:
            yield cfg
        self.save_config()

    def save_config(self):
        cfg = self.cfg.as_native_dict()
        if not self.dir is None:
            with open(self.dir / 'config.yml', 'w') as f:
                yaml.safe_dump(cfg, f)
        if self.run:
            self.run.config.update(cfg, allow_val_change=True)


    @classmethod
    def create(
        cls,
        basedir=None,
        config=None,
        wandb=None,
        name=None,
        seed=None,
        extra_conf=None
    ):
        # Directory for experiment
        if basedir is None:
            dir = None
        else:
            basedir = pathlib.Path(basedir)
            if not basedir.exists():
                raise Exception(f'Basedir {basedir} does not exist')
            if not basedir.is_dir():
                raise Exception(f'Basedir {basedir} is not a directory')
            if name is None:
                matches = [re.match('^(\d+)$', p.name) for p in basedir.iterdir()]
                nums = [int(m.groups()[0]) for m in matches if m]
                next_num = max(nums) + 1 if nums else 0
                dir = basedir / str(next_num).zfill(5)
                dir.mkdir()
            else:
                dir = basedir / name
                dir.mkdir()

        # TODO: logger
        log = Logger(None if dir is None else dir / 'log')

        # Config
        if config is None:
            if basedir is None:
                cfg = {}
            else:
                config = basedir / 'config.yml'
                if config.exists():
                    with open(config, 'r') as f:
                        cfg = yaml.safe_load(f)
                else:
                    cfg = {}
        elif isinstance(config, dict):
            cfg = config
        else:
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
        cfg = Config(cfg)
        with cfg.unfreeze():
            cfg.experiment.dir = None if dir is None else str(dir)
            if wandb is None:
                cfg.experiment.wandb = cfg.experiment.get('wandb')
            else:
                cfg.experiment.wandb = wandb
            if seed is None:
                random_seed = np.random.randint(1<<32)
                cfg.experiment.seed = cfg.experiment.get('seed', random_seed)
            else:
                cfg.experiment.seed = seed
            if extra_conf:
                for k, v, strict in extra_conf:
                    cfg_node = cfg
                    parts = k.split('.')
                    for i, part in enumerate(parts):
                        if strict and not part in cfg_node:
                            raise ValueError(f'Node {k} not found in config.')
                        if i < len(parts) - 1:
                            cfg_node = cfg_node[part]
                        else:
                            cfg_node[parts[-1]] = v

        # Weights and biases
        if cfg.experiment.wandb is None:
            run = None
        else:
            import wandb as wandb_lib
            if ':' in wandb:
                wandb, wandb_group = wandb.split(':')
            else:
                wandb, wandb_group = wandb, None
            run = wandb_lib.init(
                project=wandb, dir=dir, group=wandb_group, name=name
            )
            log.run = run

        exp = cls(cfg=cfg, dir=dir, run=run, log=log)
        exp.save_config()
        return exp


    @classmethod
    def restore(cls, dir):
        dir = pathlib.Path(dir)
        with open(dir / 'config.yml') as f:
            cfg = yaml.safe_load(f)
        return cls(cfg=cfg, dir=dir)

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('basedir', nargs='?', type=pathlib.Path, help='Path to basedir of experiment')
        parser.add_argument('--config', type=pathlib.Path, help='Path to config (default to <basedir>/config.yml)')
        parser.add_argument('--wandb', type=str, help='Name of wandb project to run under')
        parser.add_argument('--name', type=str, help='Optional name of run')
        parser.add_argument('--seed', type=int, help='Fix seed ')
        parser_args = [
            v for a, v in zip(sys.argv, sys.argv[1:])
            if not a.lower().startswith('-c.')
            and not v.lower().startswith('-c.')
        ]
        override_args = [
            (a[3:], yaml.safe_load(v), a[1] == 'c')
            for a, v in zip(sys.argv, sys.argv[1:])
            if a.lower().startswith('-c.')
        ]
        return cls.create(
            extra_conf=override_args,
            **vars(parser.parse_args(parser_args))
        )


class Logger:

    def __init__(self, file=None, run=None):
        self.file = file
        self.run = run
        self.clear()

    def clear(self):
        self.data = {}
        self.prefixes = []

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def log(self, data):
        if isinstance(data, str):
            print(data)
        elif isinstance(data, dict):
            prefix = '/'.join(self.prefixes)
            self.data.update({'/'.join([prefix, str(k)]): v for k, v in data.items()})
        else:
            raise ValueError(f'Unexpected data passed to logger: {data}')

    def __lshift__(self, data):
        self.log(data)
        return self

    @contextlib.contextmanager
    def prefix(self, *prefixes):
        try:
            self.push_prefix(*prefixes)
            yield
        finally:
            self.pop_prefix(n=len(prefixes))

    def push_prefix(self, *prefixes):
        for prefix in prefixes:
            self.prefixes.append(str(prefix))

    def pop_prefix(self, n=1):
        return tuple(self.prefixes.pop(-1) for i in range(n))

    def dump(self, epoch):
        self.log(('=-' * 39) + '-')
        self.log(f'Epoch {epoch}')
        if self.data:
            if self.run:
                self.run.log(self.data)
            padto = max(len(str(k)) for k in self.data)
            for k, v in self.data.items():
                try:
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().item()
                    print(f'{str(k).rjust(padto)}: {v: .5f}')
                except Exception as e:
                    print(f'{str(k).rjust(padto)}: {e}')
        self.log('=-' * 40)

    @contextlib.contextmanager
    def epoch(self, epoch):
        self.clear()
        yield self
        self.dump(epoch)

    def epochs(self, epochs, start=0):
        for epoch in range(start, epochs):
            with self.epoch(epoch):
                yield epoch


import contextlib
class Config(dict):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, (dict, Config)):
                cfg = Config(v)
                super().__setitem__(k, cfg)
        self.__frozen = True

    def toggle_freeze(self, frozen):
        self.__frozen = frozen
        super().__setattr__('_Config__frozen', frozen)
        for v in self.values():
            if isinstance(v, Config):
                v.toggle_freeze(frozen)

    def as_native_dict(self):
        return {
            k: v.as_native_dict() if isinstance(v, Config) else v
            for k, v in self.items()
        }

    @contextlib.contextmanager
    def unfreeze(self):
        old = self.__frozen
        try:
            self.toggle_freeze(False)
            yield self
        finally:
            self.toggle_freeze(old)

    def __getattr__(self, k):
        try:
            return super().__getattribute__(k)
        except AttributeError as e:
            try:
                return self[k]
            except KeyError:
                raise e

    def __setattr__(self, k, v):
        if k == '_Config__frozen':
            super().__setattr__('_Config__frozen', v)
        else:
            self[k] = v

    def __getitem__(self, k):
        try:
            return super().__getitem__(k)
        except KeyError as e:
            if self.__frozen:
                raise e
            else:
                self[k] = {}
                return self[k]

    def __setitem__(self, k, v):
        if self.__frozen:
            raise RuntimeError(f'Cannot set value {k} on frozen Config')
        if isinstance(v, (dict, Config)):
            v = Config(v)
            v.toggle_freeze(self.__frozen)
        super().__setitem__(k, v)

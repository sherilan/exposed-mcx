import numpy as np
import pandas as pd
import scipy.signal as signal

import exposed.utils as utils

class Motion:

    def reset(self, dt, num, random):
        pass

    def step(self):
        raise NotImplementedError()

    def get(self, key):
        pass

    def __getitem__(self, key):
        return self.get(key)


class VesselMotion(Motion):

    class Vessel:
        def __init__(self, pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=1.0):
            self.pos = utils.m3d.Vector3(pos)
            self.quat = utils.m3d.Quaternion(quat)
            self.scale = scale

        def default(self):
            return {'pos': [self.pos], 'quat': [self.quat]}

        def transform(self, seq=None):
            pos = self.pos + utils.m3d.Vector3(self.scale * seq[list('xyz')])
            quat = self.quat * utils.m3d.Quaternion.from_rpy(seq[list('rpj')])
            return {'pos': pos.values, 'quat': quat.values}

    class Dataset:
        def __init__(self, data, freq, pad=0.1, name=None):
            self.data = data
            self.freq = freq
            self.pad = pad
            self.name = name

        def sample_sequence(self, dt, num, random):
            """Samples a random sequence from the dataset"""
            freq_ratio = (self.freq * dt)
            num_samp = 1 + int(num * freq_ratio)
            num_pad = int(self.pad * num_samp)
            num_samp_pad = num_samp + 2 * num_pad
            try:
                assert num_samp_pad < len(self.data), f'Too long sequence ({num_samp_pad})'
            except AssertionError:
                breakpoint()
            start = random.choice(len(self.data) - num_samp_pad)
            data = self.data.iloc[start:start + num_samp_pad].values
            if abs(1 - freq_ratio) > 1e-6:
                num_resamp = int(round(num_samp_pad / freq_ratio))
                data = signal.resample(data, num_resamp)
            num_pad_deratiod = int(num_pad / freq_ratio)
            data = data[num_pad_deratiod: num_pad_deratiod + num]
            delta = pd.Timedelta('1s') * dt
            index = pd.timedelta_range(0, periods=len(data), freq=delta)
            return pd.DataFrame(data, index=index, columns=self.data.columns)

        def __repr__(self):
            return f'<VesselDataset({self.name}, n={len(self.data)})>'

        @classmethod
        def load(cls, path, freq=100, pad=0.1):
            path = utils.data.get_file(path)
            names = list('txyzrpj')
            drop_names = list('t')  # drop time stamps
            dist_names = list('xyz')
            rot_names = list('rpj')
            data = pd.DataFrame(np.load(str(path)), columns=names)
            data = data.drop(columns=drop_names)
            data[dist_names] *= 1e-3  # milimeters -> meters
            data[rot_names] = np.deg2rad(data[rot_names]) # deg -> rad
            return cls(data, freq, pad=pad, name=path)

    def __init__(self, vessels, datasets):
        self.vessels = vessels
        self.datasets = datasets
        self.trajectories = {
            name: vessel.default() for name, vessel in vessels.items()
        }
        self.index = 0

    def reset(self, dt, num, random):
        """Resets the motion generator by sampling a new sequence"""
        # Pick a random dataset
        dataset = random.choice(self.datasets)
        # Sample random trajectory for each vessel
        trajectories = {}
        for name, vessel in self.vessels.items():
            seq = dataset.sample_sequence(dt=dt, num=num, random=random)
            trajectories[name] = vessel.transform(seq)
        # Set state
        self.trajectories = trajectories
        self.index = 0

    def step(self):
        self.index += 1

    def get(self, key):
        if isinstance(key, tuple):
            vessel, column = key
        else:
            vessel, column = 0, key
        return self.trajectories[vessel][column][self.index]

    @classmethod
    def load(cls, vessels, datasets, **kwargs):
        vessels = {name: cls.Vessel(**v) for name, v in vessels.items()}
        datasets = [cls.Dataset.load(path, **kwargs) for path in datasets]
        return cls(vessels=vessels, datasets=datasets)


class TrajectoryMotion(Motion):

    def __init__(self, x=0, y=0, z=0, r=0, a=0):
        self.x, self.y, self.z, self.r, self.a = x, y, z, r, a
        self.minvals = [
            v[0] if isinstance(v, (tuple, list)) else v for v in [x, y, z, r, a]
        ]
        self.maxvals = [
            v[1] if isinstance(v, (tuple, list)) else v for v in [x, y, z, r, a]
        ]
        self.trajectory = {  # placeholder
            'pos': (0, 0, 0),
            'quat': (1, 0, 0, 0),
        }

    def reset(self, dt, num, random):
        x, y, z, r, a = random.uniform(self.minvals, self.maxvals)
        pos = [x, y, z]
        axis = np.array([np.cos(r), np.sin(r), 0])
        quat = utils.m3d.Quaternion.from_direction_angle(axis, a).values
        self.trajectory = {'pos': pos, 'quat': quat}

    def step(self):
        pass

    def get(self, key):
        return self.trajectory[key]

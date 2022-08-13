import collections
import functools
import os

import numpy as np
import dm_env.specs

import exposed.entities as entities
import exposed.tasks as tasks
import exposed.utils.obs as obs
import exposed.utils.act as act
import exposed.utils.rew as rew
import exposed.utils.m3d as m3d
import exposed.utils.colors as colors
import exposed.utils.data as data
import exposed.utils.motion as motion
import exposed.utils.filters as filters
import exposed.utils.mj as mj


class V2VTask(tasks.BaseTask):


    def __init__(
        self,
        vessel_motion,
        trajectory_motion,
        episode_length=1000,
        **kwargs
    ):

        super().__init__(
            arena=entities.ExposedArena(),  # TODO: add opts
            episode_length=episode_length,
            **kwargs,
        )

        self.vessel_motion = vessel_motion
        self.trajectory_motion = trajectory_motion

        # Create vessels
        self.vessel_a = self.arena.add_boat()
        self.vessel_b = self.arena.add_boat()

        # Add robot arm to first vessel
        self.arm = entities.Ur10eArm()
        self.vessel_a.entity.attach(self.arm)

        # Setup relevant frames (to easily reference transforms)
        self.frame_Va = entities.Frame('Va')
        self.frame_B = entities.Frame('B')
        self.frame_E = entities.Frame('E')
        self.frame_Vb = entities.Frame('Vb')
        self.frame_G = entities.Frame('G', size=0.05)
        self.frame_Ghat = entities.Frame('Gh')

        # Attach frames to kinematic chain
        self.vessel_a.entity.attach(self.frame_Va)
        self.frame_Va.attach(self.frame_B)
        self.arm.attach(self.frame_E)
        self.vessel_b.entity.attach(self.frame_Vb)
        self.frame_Vb.attach(self.frame_G)
        self.frame_G.attach(self.frame_Ghat)

        self.ik_invalid_state = False

    @property
    def has_invalid_state(self):
        return self.ik_invalid_state


    def create_observables(self):
        observables = super().create_observables()
        # Relative translation from base to end-effector
        observables['pos_B_E'] = obs.Position(
            obj=self.frame_E.body,
            frame=self.frame_B.body,
        )
        # Relative orientation from base to end-effector
        observables['orn_B_E'] = obs.Orientation(
            obj=self.frame_E.body,
            frame=self.frame_B.body,
        )
        # Relative translation from base to goal pose
        observables['pos_B_G'] = obs.Position(
            obj=self.frame_G.body,
            frame=self.frame_B.body,
        )
        # Relative orientation from base to goal pose
        observables['orn_B_G'] = obs.Orientation(
            obj=self.frame_G.body,
            frame=self.frame_B.body,
        )
        # Relative translation from end-effector to goal pose (seen from B)
        # Essentially r^B_G - r^B_E
        # TODO: see if it is better to do it locally (just r^E_G)
        observables['pos_E_G_from_B'] = obs.Direction(
            obj_a=self.frame_E.body,
            obj_b=self.frame_G.body,
            frame=self.frame_B.body,
        )
        # Relative orientation from end-effector to goal pose
        # TODO: maybe this thing should not be local
        observables['orn_E_G'] = obs.Orientation(
            obj=self.frame_E.body,
            frame=self.frame_G.body,
        )
        observables['norm_pos_G_E'] = obs.Distance(
            obj_a=self.frame_E.body, obj_b=self.frame_G.body,
        )
        observables['norm_orn_G_E'] = obs.Angle(
            obj_a=self.frame_E.body, obj_b=self.frame_G.body,
        )
        return observables

    def create_actions(self):
        actions = super().create_actions()
        actions['ik'] = V2VIKAction(self)
        return actions

    def create_rewards(self):
        rewards = super().create_rewards()
        rewards['tracking'] = rew.TrackingCost(
            object=self.frame_E.body,
            target=self.frame_G.body,
        )
        # TODO: maybe add something for T_G_Ghat smoothing
        return rewards

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self.ik_invalid_state = False
        # Reset motion generators
        self.vessel_motion.reset(
            dt=physics.timestep(),
            num=(1 + self.episode_length) * self.physics_steps_per_control_step,
            random=random_state,
        )
        self.trajectory_motion.reset(
            dt=physics.timestep(),
            num=(1 + self.episode_length),
            random=random_state,
        )
        # Reset position of vessels
        self.vessel_a.move(
            physics=physics,
            pos=self.vessel_motion['Va', 'pos'],
            quat=self.vessel_motion['Va', 'quat'],
            reset=True,
        )
        self.vessel_b.move(
            physics=physics,
            pos=self.vessel_motion['Vb', 'pos'],
            quat=self.vessel_motion['Vb', 'quat'],
            reset=True,
        )
        # Sample position for G frame
        self.frame_G.move(
            physics=physics,
            pos=self.trajectory_motion['pos'],
            quat=self.trajectory_motion['quat'],
        )
        # Reset G_hat frame (back to coincide with G frame)
        self.frame_Ghat.move(
            physics=physics,
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
        )
        # Set arm to reference location
        self.arm.reset(
            physics=physics,
            target=self.frame_Ghat.body,
            qinit=[-0.1, -1.0,  2.4, -2.9,  -1.5, -0.0],
        )
        # Allow simulation to stabilize a bit
        for _ in range(100):
            physics.step()

    def before_step(self, physics, action, random_state):
        # Update trajectory position of frame_G
        self.trajectory_motion.step()
        self.frame_G.move(
            physics=physics,
            pos=self.trajectory_motion['pos'],
            quat=self.trajectory_motion['quat'],
        )
        super().before_step(physics, action, random_state)


    def before_substep(self, physics, action, random_state):
        super().before_substep(physics, action, random_state)
        self.vessel_motion.step()
        self.vessel_a.move(
            physics=physics,
            pos=self.vessel_motion['Va', 'pos'],
            quat=self.vessel_motion['Va', 'quat'],
        )
        self.vessel_b.move(
            physics=physics,
            pos=self.vessel_motion['Vb', 'pos'],
            quat=self.vessel_motion['Vb', 'quat'],
        )

    def after_step(self, physics, random_state):
        super().after_step(physics, random_state)
        dist = mj.get_distance(physics, self.frame_G.body, self.frame_E.body)
        rgba = colors.RGBA.blend_hsv(
            c1=colors.RGBA.green(),
            c2=colors.RGBA.red(),
            c1_frac=max(0, 1 - dist * 100)  # 1 cm away = full red
        )
        self.frame_G.set_color(physics, rgba)

    @classmethod
    def create(
        cls,
        datasets,
        vessel_a=None,
        vessel_b=None,
        trajectory=None,
        **kwargs,
    ):
        if vessel_a is None:
            vessel_a = {}
        if vessel_b is None:
            vessel_b = {'pos': (0.65, 0.0, 0.0), 'quat': (0, 0, 0, 1)}
        if trajectory is None:
            trajectory = {
                'x': [-0.1, 0.1],
                'y': [-0.1, 0.1],
                'z': [0.1, 0.3],
                'r': 0.25 * np.pi,
                'a': np.pi,
            }
        vessels={'Va': vessel_a, 'Vb': vessel_b}
        return cls(
            vessel_motion=motion.VesselMotion.load(
                vessels={'Va': vessel_a, 'Vb': vessel_b},
                datasets=datasets,
            ),
            trajectory_motion=motion.TrajectoryMotion(**trajectory),
            **kwargs
        )



class V2VIKAction(act.Action):
    """
    Dedicated Inverse Kinematics controller for the V2V task.
    """

    def __init__(
        self,
        task,
        pos_scale=0.01, # "centimeter"
        orn_scale=0.01, # "centiradians"
        qpos_delay=0,
        qpos_smooth=None,
        pred_orn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task = task
        self.pos_scale = pos_scale
        self.orn_scale = orn_scale
        self.qpos_delay = qpos_delay
        self.qpos_smooth = qpos_smooth
        self.qpos_fddiff = filters.FiniteDifference(dt=self.dt)
        self.pred_orn = pred_orn

    @property
    def dt(self):
        """Time in seconds between each action update"""
        return self.task.control_timestep

    @property
    def qpos_delay(self):
        """Fixed delay before qpos values is sent to PD controller"""
        return self.qpos_delay_params

    @qpos_delay.setter
    def qpos_delay(self, delay):
        size = 1 + int(round(delay / self.dt))
        self.qpos_buffer = filters.Buffer(size=size)
        self.qpos_delay_params = delay

    @property
    def qpos_smooth(self):
        """Online butterworth smoothing for calculated qpos values"""
        return self.qpos_smooth_params

    @qpos_smooth.setter
    def qpos_smooth(self, params):
        if params is None:
            self.qpos_filter = lambda qpos: qpos
        else:
            order, cutoff = params
            self.qpos_filter = filters.Butterworth(order=order, cutoff=cutoff)
        self.qpos_smooth_params = params

    def get_spec(self, physics, name=None):
        spec = collections.OrderedDict()
        spec['pos'] = dm_env.specs.BoundedArray(
            shape=(3,),
            dtype=np.dtype(float),
            minimum=np.full(3, -1000, dtype=float),
            maximum=np.full(3, +1000, dtype=float),
            name=f'{name}.pos',
        )
        if self.pred_orn:
            spec['orn'] = dm_env.specs.BoundedArray(
                shape=(3,),
                dtype=np.dtype(float),
                minimum=np.full(3, -1000, dtype=float),
                maximum=np.full(3, +1000, dtype=float),
                name=f'{name}.orn',
            )
        return spec

    def apply(self, physics, action, random_state):

        # Move Ghat frame (relative to G)
        # TODO: maybe this need to be from frame B point of view
        action_pos = action['pos'] * self.pos_scale
        if self.pred_orn:
            action_orn = action['orn'] * self.orn_scale
        else:
            action_orn = np.zeros(3)
        self.task.frame_Ghat.move(
            physics=physics,
            pos=action_pos,
            quat=m3d.Quaternion.from_rotation_vector(action_orn).values,
        )

        # Do Inverse kinematics
        ik_result = self.task.arm.inverse_kinematics(
            physics=physics,
            target=self.task.frame_Ghat.body,
        )
        self.task.ik_invalid_state = not ik_result.success

        # Delay, smooth, and differentiate
        qpos = ik_result.qpos
        qpos = self.qpos_buffer(qpos)  # Fixed delay
        qpos = self.qpos_filter(qpos)  # Butterworth filter
        qvel = self.qpos_fddiff(qpos)  # FD Differentiation

        # Set ref values for PD controller
        self.task.arm.set_pos_ctrl(physics=physics, qpos=qpos)
        self.task.arm.set_vel_ctrl(physics=physics, qvel=qvel)




# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#  Constructors
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# ---- Training

def train_task(delay, smooth, **opts):
    args = {
        'datasets': [
            os.path.join('train', f'loke_Hs={hs:.1f}_Tp={tp}.train.npy')
            for hs, tp in [(1.0, 6), (1.5, 8), (2.0, 9), (2.5, 10)]
        ],
        'act': {'ik':
            {
                **{'qpos_delay': delay, 'qpos_smooth': smooth},
                **opts.pop('act', {}).get('ik', {}) # Ugly blend
            }
        },
    }
    return V2VTask.create(**{**args, **opts})

def train_delay_0(**opts):
    return train_task(delay=0, smooth=None, **opts)

def train_delay_20(**opts):
    return train_task(delay=20e-3, smooth=None, **opts)

def train_delay_40(**opts):
    return train_task(delay=40e-3, smooth=None, **opts)

def train_delay_100(**opts):
    return train_task(delay=100e-3, smooth=None, **opts)

def train_smooth_20(**opts):
    return train_task(delay=0, smooth=(4, 0.832), **opts)

def train_smooth_40(**opts):
    return train_task(delay=0, smooth=(4, 0.416), **opts)

def train_smooth_100(**opts):
    return train_task(delay=0, smooth=(4, 0.1664), **opts)

# ---- Evaluation

def eval_task(hs, tp, delay, smooth):
    return V2VTask.create(
        datasets=[os.path.join('test', f'loke_Hs={hs:.1f}_Tp={tp}.test.npy')],
        act={'ik': {'qpos_delay': delay, 'qpos_smooth': smooth}}
    )

# -- No delay

def eval_hs10_delay_0():
    return eval_task(hs=1.0, tp=6, delay=0, smooth=None)

def eval_hs15_delay_0():
    return eval_task(hs=1.5, tp=8, delay=0, smooth=None)

def eval_hs20_delay_0():
    return eval_task(hs=2.0, tp=9, delay=0, smooth=None)

def eval_hs25_delay_0():
    return eval_task(hs=2.5, tp=10, delay=0, smooth=None)

# -- 20 ms buffered delay

def eval_hs10_delay_20():
    return eval_task(hs=1.0, tp=6, delay=20e-3, smooth=None)

def eval_hs15_delay_20():
    return eval_task(hs=1.5, tp=8, delay=20e-3, smooth=None)

def eval_hs20_delay_20():
    return eval_task(hs=2.0, tp=9, delay=20e-3, smooth=None)

def eval_hs25_delay_20():
    return eval_task(hs=2.5, tp=10, delay=20e-3, smooth=None)

# -- 40 ms buffered delay

def eval_hs10_delay_40():
    return eval_task(hs=1.0, tp=6, delay=40e-3, smooth=None)

def eval_hs15_delay_40():
    return eval_task(hs=1.5, tp=8, delay=40e-3, smooth=None)

def eval_hs20_delay_40():
    return eval_task(hs=2.0, tp=9, delay=40e-3, smooth=None)

def eval_hs25_delay_40():
    return eval_task(hs=2.5, tp=10, delay=40e-3, smooth=None)

# -- 100 ms buffered delay

def eval_hs10_delay_100():
    return eval_task(hs=1.0, tp=6, delay=100e-3, smooth=None)

def eval_hs15_delay_100():
    return eval_task(hs=1.5, tp=8, delay=100e-3, smooth=None)

def eval_hs20_delay_100():
    return eval_task(hs=2.0, tp=9, delay=100e-3, smooth=None)

def eval_hs25_delay_100():
    return eval_task(hs=2.5, tp=10, delay=100e-3, smooth=None)

# -- 4th order butterworth smoothing with Wn=0.832 (approx 20 ms delay)
# 0.416 / (20e-3 * 25Hz) = 0.832

def eval_hs10_smooth_20():
    return eval_task(hs=1.0, tp=6, delay=0, smooth=(4, 0.832))

def eval_hs15_smooth_20():
    return eval_task(hs=1.5, tp=8, delay=0, smooth=(4, 0.832))

def eval_hs20_smooth_20():
    return eval_task(hs=2.0, tp=9, delay=0, smooth=(4, 0.832))

def eval_hs25_smooth_20():
    return eval_task(hs=2.5, tp=10, delay=0, smooth=(4, 0.832))


# -- 4th order butterworth smoothing with Wn=0.416 (approx 40 ms delay)
# 0.416 / (40e-3 * 25Hz) = 0.416

def eval_hs10_smooth_40():
    return eval_task(hs=1.0, tp=6, delay=0, smooth=(4, 0.416))

def eval_hs15_smooth_40():
    return eval_task(hs=1.5, tp=8, delay=0, smooth=(4, 0.416))

def eval_hs20_smooth_40():
    return eval_task(hs=2.0, tp=9, delay=0, smooth=(4, 0.416))

def eval_hs25_smooth_40():
    return eval_task(hs=2.5, tp=10, delay=0, smooth=(4, 0.416))


# -- 4th order butterworth smoothing with Wn=0.17 (approx 100 ms delay)
# 0.416 / (100e-3 * 25Hz) = 0.1664

def eval_hs10_smooth_100():
    return eval_task(hs=1.0, tp=6, delay=0, smooth=(4, 0.1664))

def eval_hs15_smooth_100():
    return eval_task(hs=1.5, tp=8, delay=0, smooth=(4, 0.1664))

def eval_hs20_smooth_100():
    return eval_task(hs=2.0, tp=9, delay=0, smooth=(4, 0.1664))

def eval_hs25_smooth_100():
    return eval_task(hs=2.5, tp=10, delay=0, smooth=(4, 0.1664))

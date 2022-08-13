import os

import numpy as np
import dm_control.entities.manipulators.base as base
import dm_control.mjcf as mjcf
import dm_control.composer as composer

# Ur10e arm converted from urdf to mujoco xml
XML_PATH = os.path.join(os.path.dirname(__file__), 'ur10e_arm/ur10e_robot.xml')
WRIST_SITE = 'toolsite'
VELOCITY_GAIN = 1_000. #float(os.environ['VELOCITY_GAIN']) #1_000. #1000.
POSITION_GAIN = 100_000. #float(os.environ['POSITION_GAIN']) # 10_000.
# TORQUE: https://www.universal-robots.com/articles/ur/robot-care-maintenance/max-joint-torques/
# VELOCITY: https://www.universal-robots.com/media/1807466/ur10e-rgb-fact-sheet-landscape-a4-125-kg.pdf
JOINT_SPEC = [
    {'name': 'shoulder_pan_joint',  'max_torque': 330, 'max_velocity': np.deg2rad(120)},
    {'name': 'shoulder_lift_joint', 'max_torque': 330, 'max_velocity': np.deg2rad(120)},
    {'name': 'elbow_joint',         'max_torque': 150, 'max_velocity': np.deg2rad(180)},
    {'name': 'wrist_1_joint',       'max_torque': 56,  'max_velocity': np.deg2rad(180)},
    {'name': 'wrist_2_joint',       'max_torque': 56,  'max_velocity': np.deg2rad(180)},
    {'name': 'wrist_3_joint',       'max_torque': 56,  'max_velocity': np.deg2rad(180)},
]

import dm_control.utils.inverse_kinematics as ik


class Ur10eArm(base.RobotArm):

    def _build(self, name=None, pos_ctrl=True, vel_ctrl=True, pos_gain=None, vel_gain=None):
        self._mjcf_root = mjcf.from_path(XML_PATH)
        if name:
          self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        self._joints = [
            self._mjcf_root.find('joint', js['name']) for js in JOINT_SPEC
        ]
        assert not any(j is None for j in self._joints)
        self._wrist_site = self._mjcf_root.find('site', WRIST_SITE)
        self._bodies = self.mjcf_model.find_all('body')
        # Add actuators.
        actuators = []

        self._actuators = []
        if vel_ctrl:
            self.vel_actuators = [
                joint.root.actuator.add(
                  'velocity',
                  joint=joint,
                  name=joint.name,
                  kv=VELOCITY_GAIN if vel_gain is None else vel_gain,
                  ctrllimited=True,
                  ctrlrange=(-js['max_velocity'], js['max_velocity']),
                  forcelimited=True,
                  forcerange=(-js['max_torque'], js['max_torque'])
                 )
                for js, joint in zip(JOINT_SPEC, self._joints)
            ]
        else:
            self.vel_actuators = []
        if pos_ctrl:
            self.pos_actuators = [
                joint.root.actuator.add(
                    'position',
                    joint=joint,
                    name=f'{joint.name}_pos',
                    # ctrllimited=True,
                    kp=POSITION_GAIN if pos_gain is None else pos_gain,
                    forcelimited=True,
                    forcerange=(-js['max_torque'], js['max_torque'])
                )
                for js, joint in zip(JOINT_SPEC, self._joints)
            ]
        else:
            self.pos_actuators = []
        self._actuators = self.vel_actuators + self.pos_actuators
        # Add torque sensors.
        self._joint_torque_sensors = [
            joint.root.sensor.add(
                'torque',
                site=joint.parent.add(
                    'site',
                    size=[1e-3],
                    group=composer.SENSOR_SITES_GROUP,
                    name=joint.name + '_site',
                ),
                name=joint.name + '_torque',
            )
            for joint in self._joints
        ]

    @property
    def joint_names(self):
        """List of full identifiers for joints """
        return [j.full_identifier for j in self.joints]

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def joint_torque_sensors(self):
        """List of torque sensors for each joint belonging to the arm."""
        return self._joint_torque_sensors

    @property
    def wrist_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._wrist_site

    @property
    def tool_site(self):
        return self.wrist_site  # TODO: pick ee tip unless otherwise stated

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    def inverse_kinematics(self, physics, target=None, pos=None, quat=None, qinit=None, filter_qpos=True, **kwargs):
        """Generates ik solution for a specific wrist site"""
        if not target is None:
            if pos is None:
                pos = physics.bind(target).xpos
            if quat is None:
                quat = physics.bind(target).xquat
        if not qinit is None:
            physics.bind(self.joints).qpos[:] = qinit
        joint_names = [joint.full_identifier for joint in self._joints]
        solution = ik.qpos_from_site_pose(
            physics,
            self.tool_site.full_identifier,
            pos,
            quat,
            joint_names,
            **kwargs
        )
        if filter_qpos:
            res = solution._asdict()
            idx = physics.named.data.qpos._convert_key(joint_names)
            res['qpos'] = res['qpos'][idx]
            solution = type(solution)(**res)
        return solution

    def reset(self, physics, *args, **kwargs):
        ik_solution = self.inverse_kinematics(
            physics, *args, filter_qpos=True, **kwargs
        )
        physics.bind(self.joints).qpos = ik_solution.qpos
        if self.pos_actuators:
            self.set_pos_ctrl(physics, ik_solution.qpos)
        if self.vel_actuators:
            self.set_vel_ctrl(physics, np.zeros(len(self.vel_actuators)))

    def set_pos_ctrl(self, physics, qpos):
        physics.bind(self.pos_actuators).ctrl[:] = qpos

    def set_vel_ctrl(self, physics, qvel):
        physics.bind(self.vel_actuators).ctrl[:] = qvel

class Ur10eArmObservables(base.JointsObservables):
    """Jaco arm obserables."""

    @composer.define.observable
    def joints_pos(self):
        # Because most of the Jaco arm joints are unlimited, we return the joint
        # angles as sine/cosine pairs so that the observations are bounded.
        def get_sin_cos_joint_angles(physics):
            joint_pos = physics.bind(self._entity.joints).qpos
            return joint_pos
            return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T
        return composer.observation.observable.Generic(get_sin_cos_joint_angles)

    @composer.define.observable
    def joints_torque(self):
        # MuJoCo's torque sensors are 3-axis, but we are only interested in torques
        # acting about the axis of rotation of the joint. We therefore project the
        # torques onto the joint axis.
        def get_torques(physics):
            torques = physics.bind(self._entity.joint_torque_sensors).sensordata
            joint_axes = physics.bind(self._entity.joints).axis
            return np.einsum('ij,ij->i', torques.reshape(-1, 3), joint_axes)
        return composer.observation.observable.Generic(get_torques)

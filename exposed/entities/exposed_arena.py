import dm_control.manipulation as manipulation
import numpy as np

import exposed.entities.boat as boat


class ExposedArena(manipulation.shared.arenas.Standard):

    class Freebody:

        def __init__(self, arena, entity, relpose=(0, 0, 0, 1, 0, 0, 0)):
            self.arena = arena
            self.entity = entity
            self.mocap = arena.mjcf_model.worldbody.add(
                'body', pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap='true'
            )
            self.frame = arena.attach(entity)
            self.joint = self.frame.add('freejoint')
            self.weld = arena.mjcf_model.equality.add(
                'weld', body1=self.mocap, body2=self.frame, relpose=relpose
            )

        def move(self, physics, pos=None, quat=None, reset=False):
            mocap_b = physics.bind(self.mocap)
            if not pos is None:
                mocap_b.mocap_pos = pos
            if not quat is None:
                mocap_b.mocap_quat = quat
            if reset:
                joint_b = physics.bind(self.joint)
                joint_b.qpos[:] = np.concatenate(
                    [mocap_b.mocap_pos, mocap_b.mocap_quat]
                )
                joint_b.qvel[:] = [0] * 6


    def _build(self, name='arena', width=3, depth=2, z0=-0.3):
        super()._build(name=name)
        # Make the sea (pool) floor a bit larger and move it further down
        ground = self.root_body.find('geom', 'ground')
        ground.pos = [0, 0, -depth]
        ground.size = (width, width, 0.1)
        # Create a plane for the water
        self.water = self.mjcf_model.worldbody.add(
            'site',
            name='water',
            type='box',
            size=(width, width, 0.1),
            pos=[0, 0, -0.1/2 + z0],
            material=self.mjcf_model.asset.add(
                'material',
                name='water_material',
                reflectance=0.2,
                rgba=[0.1, 0.3, 0.6, 0.2],
            ),
        )
        # # Add a "boat" platform with mocap control enabled
        # self.boat_mocap = self.mjcf_model.worldbody.add(
        #     'body', pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap='true', name='boat_mover'
        # )
        # # Create a boat entity and attach it as a free-moving (6DOF) object
        # self.boat = boat.Boat()
        # self.boat_frame = self.attach(self.boat)
        # self.boat_joint = self.boat_frame.add('freejoint')
        # # Weld it to the boat mocap body with a constraint
        # self.boat_weld = self.mjcf_model.equality.add(
        #     'weld', body1=self.boat_mocap, body2=self.boat_frame,
        #     relpose=[0.0, 0.0, 0.01,  1, 0, 0, 0],
        # )

    def add_freebody(self, entity, **kwargs):
        return self.Freebody(self, entity, **kwargs)

    def add_boat(self, **kwargs):
        return self.Freebody(self, boat.Boat(), **kwargs)

    def move_boat(self, physics, pos, quat=None, reset=False):
        # Default to identity rotation quaternion
        quat = [1, 0, 0, 0] if quat is None else quat
        # Update mocap position for the boat
        boat_binding = physics.bind(self.boat_mocap)
        boat_binding.mocap_pos[:] = pos
        boat_binding.mocap_quat[:] = quat
        # If reset, also move the boat (free) joint to the reference pose
        # (which would otherwise be dynamically updated by the weld constraint)
        if reset:
            boat_joint_binding = physics.bind(self.boat_joint)
            boat_joint_binding.qpos[:] = np.concatenate(
                [boat_binding.mocap_pos, boat_binding.mocap_quat]
            )
            boat_joint_binding.qvel[:] = [0] * 6

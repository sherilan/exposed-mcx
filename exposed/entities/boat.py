
import dm_control.composer as composer
import dm_control.mjcf as mjcf

import os
MESH_PATH = os.path.join(os.path.dirname(__file__), 'boat', 'boat.stl')

class Boat(composer.Entity):

    def _build(self, name='boat', scale=1, quat=(1, 0, 0, 0)):
        self.root = mjcf.RootElement(name)
        self.mesh = self.root.asset.add(
            'mesh', file=MESH_PATH, scale=[scale, scale, scale]
        )
        self.boat = self.root.worldbody.add(
            'geom',
            name='boat',
            type='mesh',
            mesh=self.mesh,
            pos=[0.1 - 0.1 * scale, 0, 0],
            quat=quat,
            contype=0,
            conaffinity=0,
            mass=10000,
        )
        self.deck = self.root.worldbody.add(
            'site',
            name='deck',
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0]
        )

    @property
    def mjcf_model(self):
        return self.root

    @property
    def attach_site(self):
        return self.deck

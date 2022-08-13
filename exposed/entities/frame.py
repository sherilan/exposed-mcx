import dm_control.composer as composer
import dm_control.mjcf as mjcf
import numpy as np

import exposed.utils.colors as colors


class Frame(composer.Entity):

    def _build(
        self,
        name='frame',
        type='sphere',
        size=0.01,
        color=colors.RGBA.black(alpha=0.8),
    ):
        self.root = mjcf.RootElement(name)
        self.site = self.root.worldbody.add(
            'site',
            name='site',
            type=type,
            size=[size, size, size],
            rgba=np.asarray(color),
        )

    @property
    def mjcf_model(self):
        return self.root

    @property
    def body(self):
        return self.root.worldbody

    def move(self, physics, pos, quat=None):
        binding = physics.bind(self.body)
        binding.pos = pos
        if not quat is None:
            binding.quat = quat

    def set_color(self, physics, rgba):
        physics.bind(self.site).rgba = rgba

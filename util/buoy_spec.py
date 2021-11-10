import numpy as np
import alluvion as al


class BuoySpec:
    def __init__(self, dp, unit, map_radial_size=24):
        self.radius = unit.from_real_length(3.0088549658278843e-3)
        self.height = unit.from_real_length(38.5e-3)
        self.mass = unit.from_real_mass(1.06e-3)  # TODO: randomize
        self.comy = unit.from_real_length(-8.852102803738316e-3)
        self.volume = self.radius * self.radius * np.pi * self.height
        # self.neutral_buoyant_force = -self.volume * density0 * gravity.y
        self.mesh = al.Mesh()
        self.mesh.set_cylinder(self.radius, self.height, 24, 24)
        self.mesh.translate(dp.f3(0, -self.comy, 0))
        self.inertia = unit.from_real_moment_of_inertia(
            dp.f3(7.911343969145678e-8, 2.944622178863632e-8,
                  7.911343969145678e-8))
        self.map_radial_size = map_radial_size
        self.map_dim = al.uint3(
            self.map_radial_size,
            int(self.map_radial_size * self.height / self.radius / 2),
            self.map_radial_size)
        self.dp = dp
        self.inset = 0.403  # found empirically to match buoyant force

    def create_distance(self, inset=None):
        if inset is None:
            inset = self.inset
        return self.dp.CylinderDistance.create(self.radius - inset,
                                               self.height - inset * 2,
                                               self.comy)

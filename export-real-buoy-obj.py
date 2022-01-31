import numpy as np
import alluvion as al

from util import BuoySpec, Unit


dp = al.Depot(np.float32)
unit = Unit(real_kernel_radius=1,
            real_density0=1,
            real_gravity=-1)

buoy = BuoySpec(dp, unit)

buoy.mesh.export_obj('buoy.obj')

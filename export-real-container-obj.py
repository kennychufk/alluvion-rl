import numpy as np
import alluvion as al

from util import Unit

dp = al.Depot(np.float32)
unit = Unit(real_kernel_radius=1,
            real_density0=1,
            real_gravity=-1)
container_width = unit.from_real_length(0.24)
container_dim = dp.f3(container_width, container_width, container_width)
container_mesh = al.Mesh()
container_mesh.set_box(container_dim, 8)
container_mesh.export_obj('cube24.obj')

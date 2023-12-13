import sys
from pathlib import Path
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import alluvion as al

model_path = Path(sys.argv[1])
mesh = al.Mesh()
mesh.set_obj(str(model_path))
mass, com, inertia_diag, inertia_off_diag = mesh.calculate_mass_properties()

print("=== Original mass properties ===")
print(f"Mass = {mass}")
print(f"Center of mass = {com}")
print(f"Inertia (diagonal) = {inertia_diag}")
print(f"Inertia (off-diagonal) = {inertia_off_diag}")

inertia = np.array([[inertia_diag.x, inertia_off_diag.x, inertia_off_diag.y],
                    [inertia_off_diag.x, inertia_diag.y, inertia_off_diag.z],
                    [inertia_off_diag.y, inertia_off_diag.z, inertia_diag.z]])
eigenvalues, eigenvectors = LA.eig(inertia)

print("=== Eigendecomposition ===")
print(f"eigenvalues = {eigenvalues}")
print("eigenvectors:")
print(eigenvectors)

q = R.from_matrix(np.transpose(eigenvectors / LA.det(eigenvectors))).as_quat()

mesh.translate(-com)
mesh.rotate(al.float4(q[0], q[1], q[2], q[3]))
mass, com, inertia_diag, inertia_off_diag = mesh.calculate_mass_properties()

print("=== Transformed mass properties ===")
print(f"Mass = {mass}")
print(f"Center of mass = {com}")
print(f"Inertia (diagonal) = {inertia_diag}")
print(f"Inertia (off-diagonal) = {inertia_off_diag}")

max_dist2 = 0.0
for vertex in mesh.vertices:
    dist2 = vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z
    if dist2 > max_dist2:
        max_dist2 = dist2
max_dist = np.sqrt(max_dist2)

print("=== Geometry ===")
print(f"Furthest distance from center: {max_dist}")

mesh.scale(1 / max_dist)

output_path = model_path.parent.joinpath(f'{model_path.stem}-pa.obj')
mesh.export_obj(str(output_path))

mesh_pa = al.Mesh()
mesh_pa.set_obj(str(output_path))
mass, com, inertia_diag, inertia_off_diag = mesh_pa.calculate_mass_properties()

print("=== Scaled mass properties ===")
print(f"Mass = {mass}")
print(f"Center of mass = {com}")
print(f"Inertia (diagonal) = {inertia_diag}")
print(f"Inertia (off-diagonal) = {inertia_off_diag}")

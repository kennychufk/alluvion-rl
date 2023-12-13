import numpy as np
import alluvion as al
from util import BuoyInterpolator, RigidInterpolator, write_pile, read_pile

dp = al.Depot(np.float32)

piv_folder_name = '20210415_162749-laser-too-high'
truth_dir = f'/media/kennychufk/vol1bk0/{piv_folder_name}/'
shape_dir = '/home/kennychufk/workspace/pythonWs/shape-al'
buoy_filter_postfix = '-f18'

used_buoy_ids = np.load(f'{truth_dir}/rec/marker_ids.npy')
buoy_trajectories = [
    np.load(f'{truth_dir}/rec/marker-{used_buoy_id}{buoy_filter_postfix}.npy')
    for used_buoy_id in used_buoy_ids
]
buoy_interpolators = [
    BuoyInterpolator(dp, sample_interval=0.01, trajectory=trajectory)
    for trajectory in buoy_trajectories
]

agitator_interpolator = RigidInterpolator(dp, f'{truth_dir}/Trace.csv')
agitator_shift = np.array([0.52, -(0.14294 + 0.015 + 0.005) - 0.12, 0])

num_frames = 2000
frame_interval = 0.01
num_buoys = len(buoy_interpolators)
num_rigids = num_buoys + 2

for frame_id in range(num_frames):
    t = frame_id * frame_interval
    xs = np.zeros((num_rigids, 3), dtype=np.float32)
    vs = np.zeros((num_rigids, 3), dtype=np.float32)
    qs = np.zeros((num_rigids, 4), dtype=np.float32)
    omegas = np.zeros((num_rigids, 3), dtype=np.float32)

    xs[0, 1] = 0.12
    qs[0, -1] = 1
    for buoy_id in range(num_buoys):
        xs[buoy_id + 1] = buoy_interpolators[buoy_id].get_x(t)
        vs[buoy_id + 1] = buoy_interpolators[buoy_id].get_v(t)
        qs[buoy_id + 1] = buoy_interpolators[buoy_id].get_q(t)
        omegas[buoy_id +
               1] = buoy_interpolators[buoy_id].get_angular_velocity(t)

    xs[-1] = agitator_interpolator.get_x(t) + dp.f3(agitator_shift)
    vs[-1] = agitator_interpolator.get_v(t)
    qs[-1, -1] = 1
    write_pile(f'{truth_dir}/{frame_id}.pile', xs, vs, qs, omegas)

np.save(f'{truth_dir}/agitator_option.npy', 'stirrer/stirrer')
identity_q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]),
                     len(agitator_interpolator.x)).reshape(-1, 4)
agitator_trajectory = np.concatenate(
    (agitator_interpolator.x + agitator_shift, identity_q,
     agitator_interpolator.t[:, np.newaxis]),
    axis=1)
np.save(f'{shape_dir}/manipulator/{piv_folder_name}.npy', agitator_trajectory)
np.save(
    f'{shape_dir}/manipulator/offsets/{piv_folder_name}.stirrer-stirrer.npy',
    np.zeros(3))

import numpy as np
from .quaternion import get_quat3, rotate_using_quaternion


def get_obs_dim():
    return 32


def get_act_dim():
    return 21


def get_coil_x_from_com(dp, unit, buoy_spec, truth_buoy_pile_real):
    buoy_x_real = dp.coat(truth_buoy_pile_real.x).get()
    buoy_q = dp.coat(truth_buoy_pile_real.q).get()
    shift_to_coil_center = rotate_using_quaternion(
        np.array([
            0,
            unit.to_real_length(buoy_spec.coil_center - buoy_spec.comy), 0
        ]), buoy_q)
    coil_x_real = buoy_x_real + shift_to_coil_center
    return coil_x_real


# using real unit except density: which is relative to density0
def make_obs(dp, unit, kinematic_viscosity_real, truth_buoy_pile_real,
             coil_x_real, usher_sampling, num_buoys):
    obs_aggregated = np.zeros([num_buoys, get_obs_dim()], dp.default_dtype)
    buoy_v_real = dp.coat(truth_buoy_pile_real.v).get()
    buoy_q = dp.coat(truth_buoy_pile_real.q).get()
    buoy_q3 = get_quat3(buoy_q)

    sample_v_real = unit.to_real_velocity(usher_sampling.sample_data3.get())
    sample_density_relative = usher_sampling.sample_data1.get()
    sample_vort_real = unit.to_real_angular_velocity(
        usher_sampling.sample_vort.get())
    sample_container_kernel_sim = usher_sampling.sample_boundary_kernel.get(
    )[0]
    sample_container_kernel_vol_grad_real = unit.to_real_per_length(
        sample_container_kernel_sim[:, :3])
    sample_container_kernel_vol = sample_container_kernel_sim[:,
                                                              3]  # dimensionless

    for buoy_id in range(num_buoys):
        xi = coil_x_real[buoy_id]
        vi = buoy_v_real[buoy_id]
        xij = xi - coil_x_real
        d2 = np.sum(xij * xij, axis=1)
        dist_sort_index = np.argsort(d2)[1:]

        obs_aggregated[buoy_id] = np.concatenate(
            (buoy_v_real[buoy_id], buoy_q3[buoy_id], xij[dist_sort_index[0]],
             vi - buoy_v_real[dist_sort_index[0]], xij[dist_sort_index[1]],
             vi - buoy_v_real[dist_sort_index[1]],
             sample_v_real[buoy_id].flatten(),
             sample_density_relative[buoy_id],
             sample_vort_real[buoy_id].flatten(),
             sample_container_kernel_vol_grad_real[buoy_id].flatten(),
             sample_container_kernel_vol[buoy_id].flatten(), unit.rdensity0 /
             1000, kinematic_viscosity_real * 1e6, num_buoys / 9),
            axis=None)
    return obs_aggregated


def set_usher_param(usher, dp, unit, truth_buoy_pile_real, coil_x_real,
                    act_aggregated, num_buoys):
    # [0:3] [3:6] [6:9] displacement from buoy x
    # [9:12] [12:15] [15:18] velocity offset from buoy v
    # [18] focal dist
    # [19] usher kernel radius
    # [20] strength
    xoffset_real = act_aggregated[:, 0:9].reshape(3, num_buoys, 3)
    voffset_real = act_aggregated[:, 9:18].reshape(3, num_buoys, 3)
    buoy_v_real = dp.coat(truth_buoy_pile_real.v).get()
    focal_x_real = np.zeros_like(xoffset_real)
    focal_v_real = np.zeros_like(voffset_real)
    for buoy_id in range(num_buoys):
        focal_x_real[:,
                     buoy_id] = xoffset_real[:, buoy_id] + coil_x_real[buoy_id]
        focal_v_real[:,
                     buoy_id] = voffset_real[:, buoy_id] + buoy_v_real[buoy_id]

    dp.coat(usher.focal_x).set(unit.from_real_length(focal_x_real))
    dp.coat(usher.focal_v).set(unit.from_real_velocity(focal_v_real))

    dp.coat(usher.focal_dist).set(unit.from_real_length(act_aggregated[:, 18]))

    dp.coat(usher.usher_kernel_radius).set(
        unit.from_real_length(act_aggregated[:, 19]))

    dp.coat(usher.drive_strength).set(
        unit.from_real_angular_velocity(act_aggregated[:, 20]))

import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.spatial.transform import Slerp, RotationSpline
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splrep, splev
from numpy import linalg as LA


# TODO: rename? only used for robot-controlled object
class RigidInterpolator:

    def __init__(self, dp, unit, trace_filename):
        self.dp = dp
        self.unit = unit
        robot_frames = pd.read_csv(trace_filename,
                                   comment='"',
                                   names=[
                                       'tid', 'j0', 'j1', 'j2', 'j3', 'j4',
                                       'j5', 'x', 'y', 'z', 'v'
                                   ])
        x = np.array([
            -robot_frames['x'].to_numpy(), robot_frames['z'].to_numpy(),
            robot_frames['y'].to_numpy()
        ]).transpose() / 1000.0
        print('shape', x.shape)
        self.cubic_spline = CubicSpline(np.arange(4000) / 200, x)

    def get_x_real_from_real_t(self, t):
        x_real = self.cubic_spline(t)
        return self.dp.f3(x_real[0], x_real[1], x_real[2])

    def get_x_real(self, t):
        x_real = self.cubic_spline(self.unit.to_real_time(t))
        return self.dp.f3(x_real[0], x_real[1], x_real[2])

    def get_x(self, t):
        return self.unit.from_real_length(self.get_x_real(t))

    def get_v_real_from_real_t(self, t):
        v_real = self.cubic_spline(t, 1)
        return self.dp.f3(v_real[0], v_real[1], v_real[2])

    def get_v_real(self, t):
        v_real = self.cubic_spline(self.unit.to_real_time(t), 1)
        return self.dp.f3(v_real[0], v_real[1], v_real[2])

    def get_v(self, t):
        return self.unit.from_real_velocity(self.get_v_real(t))


class LeapInterpolator:

    def __init__(self, dp, filename, offset=None):
        self.dp = dp
        self.snapshots = np.load(filename)
        self.t = self.snapshots[:, -1]

        self.filtered_snapshots = np.empty_like(self.snapshots)
        self.filtered_snapshots[:] = self.snapshots
        n_interior_knots = 100
        qs = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
        knots = np.quantile(self.t, qs)
        for i in [0, 1, 2, 3, 4, 6]:
            tck = splrep(self.t, self.snapshots[:, i], t=knots, k=3)
            self.filtered_snapshots[:, i] = splev(self.t, tck)
        self.filtered_snapshots[:, 3:7] /= np.repeat(
            LA.norm(self.filtered_snapshots[:, 3:7], axis=1),
            4).reshape(-1, 4)

        self.x = self.filtered_snapshots[:, 0:3]
        if offset is not None:
            self.x += offset
        self.q = self.filtered_snapshots[:, 3:7]
        self.rot = R.from_quat(self.q)
        self.cubic_spline = CubicSpline(self.t, self.x)
        self.rot_spline = RotationSpline(self.t, self.rot)

    def get_x(self, t):
        x_real = self.cubic_spline(t)
        return self.dp.f3(x_real)

    def get_v(self, t):
        v_real = self.cubic_spline(t, 1)
        return self.dp.f3(v_real)

    def get_q(self, t):
        return self.dp.f4(self.rot_spline(t).as_quat())

    def get_angular_velocity(self, t):
        return self.dp.f3(self.rot_spline(t, 1))


# use real units
class BuoyInterpolator:

    def __init__(self, dp, sample_interval, trajectory):
        self.dp = dp
        times = np.arange(len(trajectory)) * sample_interval
        self.cubic_spline = CubicSpline(times, trajectory['x'])
        self.slerp = Slerp(
            times, R.from_quat(np.roll(trajectory['q'], -1, axis=-1))
        )  # TODO: change emstrobol output format to prevent similar bugs?
        # both scipy and alluvion(float4) uses <x, y, z, w>
        # only penrose/emstrobol uses <w, x, y, z>
        # TODO: redo offline regression due to the presence of better DNN model params

    def get_x(self, t):
        return self.cubic_spline(t)

    def get_v(self, t):
        return self.cubic_spline(t, 1)

    def get_q(self, t):
        return self.slerp(t).as_quat()

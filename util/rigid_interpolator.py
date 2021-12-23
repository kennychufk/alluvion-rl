import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd


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
        pass

    def get_x_real(self, t):
        x_real = self.cubic_spline(self.unit.to_real_time(t))
        return self.dp.f3(x_real[0], x_real[1], x_real[2])

    def get_x(self, t):
        return self.unit.from_real_length(self.get_x_real(t))

    def get_v_real(self, t):
        v_real = self.cubic_spline(self.unit.to_real_time(t), 1)
        return self.dp.f3(v_real[0], v_real[1], v_real[2])

    def get_v(self, t):
        return self.unit.from_real_velocity(self.get_v_real(t))

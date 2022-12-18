import numpy as np


class Unit:

    def __init__(self, real_kernel_radius, real_density0, real_gravity):
        self.rdensity0 = real_density0
        self.rg = real_gravity

        self.rl = real_kernel_radius
        self.rm = self.rl * self.rl * self.rl * self.rdensity0
        self.rt = np.sqrt(self.rl / -self.rg)

        self.inv_rl = 1 / self.rl
        self.inv_rm = 1 / self.rm
        self.inv_rg = 1 / self.rg
        self.inv_rt = 1 / self.rt
        self.inv_rdensity0 = 1 / self.rdensity0

    def from_real_mass(self, m):
        return m * self.inv_rm

    def to_real_mass(self, m):
        return m * self.rm

    def from_real_length(self, l):
        return l * self.inv_rl

    def to_real_length(self, l):
        return l * self.rl

    def from_real_per_length(self, l):
        return l * self.rl

    def to_real_per_length(self, l):
        return l * self.inv_rl

    def from_real_volume(self, v):
        return v * (self.inv_rl * self.inv_rl * self.inv_rl)

    def to_real_volume(self, v):
        return v * (self.rl * self.rl * self.rl)

    def from_real_dynamic_viscosity(self, mu):
        return mu * (self.inv_rdensity0 * self.inv_rl * self.inv_rl * self.rt)

    def to_real_dynamic_viscosity(self, mu):
        return mu * (self.rdensity0 * self.rl * self.rl * self.inv_rt)

    def from_real_kinematic_viscosity(self, nu):
        return nu * (self.inv_rl * self.inv_rl * self.rt)

    def to_real_kinematic_viscosity(self, nu):
        return nu * (self.rl * self.rl * self.inv_rt)

    def from_real_velocity(self, v):
        return v * (self.inv_rl * self.rt)

    def to_real_velocity(self, v):
        return v * (self.rl * self.inv_rt)

    def from_real_acceleration(self, a):
        return a * (self.inv_rl * self.rt * self.rt)

    def to_real_acceleration(self, a):
        return a * (self.rl * self.inv_rt * self.inv_rt)

    def from_real_time(self, t):
        return t * self.inv_rt

    def to_real_time(self, t):
        return t * self.rt

    def from_real_frequency(self, t):
        return t * self.rt

    def to_real_frequency(self, t):
        return t * self.inv_rt

    def from_real_density(self, rho):
        return rho * self.inv_rdensity0

    def to_real_density(self, rho):
        return rho * self.rdensity0

    def from_real_moment_of_inertia(self, i):
        return i * (self.inv_rdensity0 * self.inv_rl * self.inv_rl *
                    self.inv_rl * self.inv_rl * self.inv_rl)

    def from_real_angular_velocity(self, omega):
        return omega * self.rt

    def to_real_angular_velocity(self, omega):
        return omega * self.inv_rt

    def from_real_angular_acceleration(self, aa):
        return aa * (self.rt * self.rt)

    def to_real_angular_acceleration(self, aa):
        return aa * (self.inv_rt * self.inv_rt)

    def to_real_velocity_mse(self, e):
        return e * (self.rl * self.rl * self.inv_rt * self.inv_rt)

    def from_real_velocity_mse(self, e):
        return e * (self.inv_rl * self.inv_rl * self.rt * self.rt)

    def to_real_velocity_mse_per_kinematic_viscosity(self, derivative):
        return derivative * self.inv_rt

    def from_real_pressure(self, p):
        return p * (self.inv_rdensity0 * self.inv_rl * self.inv_rl * self.rt *
                    self.rt)

    def to_real_pressure(self, p):
        return p * (self.rdensity0 * self.rl * self.rl * self.inv_rt *
                    self.inv_rt)

    def from_real_energy(self, energy):
        return energy * (self.inv_rm * self.inv_rg * self.inv_rl)

    def to_real_energy(self, energy):
        return energy * (self.rm * self.rg * self.rl)

    def from_real_force(self, f):
        return f * (self.inv_rm * self.inv_rg)

    def to_real_force(self, f):
        return f * (self.rm * self.rg)

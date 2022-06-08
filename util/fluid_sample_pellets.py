import numpy as np


class FluidSamplePellets:

    def __init__(self, dp, x_src):
        set_with_file = isinstance(x_src, str)
        self.num_samples = dp.get_alu_info(
            x_src)[0][0] if set_with_file else x_src.shape[0]

        self.sample_x = dp.create_coated((self.num_samples), 3)
        self.sample_data3 = dp.create_coated((self.num_samples), 3)
        self.sample_vort = dp.create_coated((self.num_samples), 3)
        self.sample_data1 = dp.create_coated((self.num_samples), 1)
        self.sample_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_neighbors = dp.create_coated((self.num_samples), 1,
                                                     np.uint32)
        self.sample_pellet_neighbors = dp.create_coated(
            (self.num_samples, dp.cni.max_num_neighbors_per_particle), 4)
        self.sample_num_pellet_neighbors = dp.create_coated((self.num_samples),
                                                            1, np.uint32)
        if set_with_file:
            self.sample_x.read_file(x_src)
        else:
            self.sample_x.set(x_src)
        self.dp = dp

    def destroy_variables(self):
        self.dp.remove(self.sample_x)
        self.dp.remove(self.sample_data3)
        self.dp.remove(self.sample_vort)
        self.dp.remove(self.sample_data1)
        self.dp.remove(self.sample_neighbors)
        self.dp.remove(self.sample_num_neighbors)
        self.dp.remove(self.sample_pellet_neighbors)
        self.dp.remove(self.sample_num_pellet_neighbors)

    def prepare_neighbor_and_boundary(self, runner, solver):
        solver.update_particle_neighbors()
        runner.launch_make_bead_pellet_neighbor_list(
            self.sample_x, solver.pid, solver.pid_length,
            self.sample_neighbors, self.sample_num_neighbors,
            self.sample_pellet_neighbors, self.sample_num_pellet_neighbors,
            solver.grid_anomaly, solver.max_num_particles, self.num_samples)

    def prepare_neighbor_and_boundary_wrap1(self, runner, solver):
        solver.update_particle_neighbors_wrap1()
        runner.launch_make_bead_pellet_neighbor_list_wrap1(
            self.sample_x, solver.pid, solver.pid_length,
            self.sample_neighbors, self.sample_num_neighbors,
            self.sample_pellet_neighbors, self.sample_num_pellet_neighbors,
            solver.grid_anomaly, solver.max_num_particles, self.num_samples)

    def sample_vector3(self, runner, solver, fluid_var):
        runner.launch_sample_fluid(self.sample_x, solver.particle_x,
                                   solver.particle_density, fluid_var,
                                   self.sample_neighbors,
                                   self.sample_num_neighbors,
                                   self.sample_data3, self.num_samples)
        return self.sample_data3

    def sample_density(self, runner):
        runner.launch_sample_density_with_pellets(
            self.sample_x, self.sample_neighbors, self.sample_num_neighbors,
            self.sample_pellet_neighbors, self.sample_num_pellet_neighbors,
            self.sample_data1, self.num_samples)
        return self.sample_data1

    def sample_fluid_density(self, runner):
        runner.launch_sample_fluid_density(self.sample_neighbors,
                                           self.sample_num_neighbors,
                                           self.sample_data1, self.num_samples)
        return self.sample_data1

    def sample_velocity(self, runner, solver):
        runner.launch_sample_velocity_with_pellets(
            self.sample_x, solver.particle_x, solver.particle_density,
            solver.particle_v, self.sample_neighbors,
            self.sample_num_neighbors, self.sample_data3,
            self.sample_pellet_neighbors, self.sample_num_pellet_neighbors,
            self.num_samples)
        return self.sample_data3


class OptimSamplingPellets(FluidSamplePellets):

    def __init__(self, dp, pipe_length, pipe_radius, ts, num_particles):
        self.ts = ts
        self.num_sections = 14
        self.num_rotations = 16
        self.num_rs = 16  # should be even number
        self.num_particles = num_particles

        # r contains 0 but not pipe_radius
        self.rs = pipe_radius / self.num_rs * np.arange(self.num_rs)

        num_samples = self.num_rs * self.num_sections * self.num_rotations
        sample_x_host = np.zeros((num_samples, 3), dp.default_dtype)
        section_length = pipe_length / self.num_sections
        offset_y_per_rotation = section_length / self.num_rotations
        theta_per_rotation = np.pi * 2 / self.num_rotations
        for i in range(num_samples):
            section_id = i // (self.num_rs * self.num_rotations)
            rotation_id = (i // self.num_rs) % (self.num_rotations)
            r_id = i % self.num_rs
            theta = theta_per_rotation * rotation_id
            sample_x_host[i] = np.array([
                self.rs[r_id] * np.cos(theta),
                pipe_length * -0.5 + section_length * section_id +
                offset_y_per_rotation * rotation_id,
                self.rs[r_id] * np.sin(theta)
            ], dp.default_dtype)
        super().__init__(dp, sample_x_host)
        self.reset()

    def reset(self):
        self.sampling_cursor = 0
        self.vx = np.zeros((len(self.ts), self.num_rs), self.dp.default_dtype)
        self.density = np.zeros((len(self.ts), self.num_rs),
                                self.dp.default_dtype)
        self.r_stat = np.zeros((len(self.ts), self.num_particles),
                               self.dp.default_dtype)
        self.density_stat = np.zeros((len(self.ts), self.num_particles),
                                     self.dp.default_dtype)

    def aggregate(self):
        sample_vx = self.sample_data3.get().reshape(-1, self.num_rs, 3)[..., 1]
        self.vx[self.sampling_cursor] = np.mean(sample_vx, axis=0)

        density_ungrouped = self.sample_data1.get().reshape(-1, self.num_rs)
        self.density[self.sampling_cursor] = np.mean(density_ungrouped, axis=0)
        self.sampling_cursor += 1

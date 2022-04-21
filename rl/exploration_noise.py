import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, mu=0.0, sigma=0.2, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def __call__(self, action_dim):
        x = self.x_prev + self.theta * (
            self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
                self.dt) * np.random.normal(size=action_dim)
        self.x_prev = x
        return x

    def reset(self, action_dim):
        self.x_prev = np.random.normal(self.mu, self.sigma, action_dim)


class GaussianNoise:
    def __init__(self, std_dev=0.1):
        self.std_dev = std_dev

    def __call__(self, action_dim):
        return np.random.normal(0, self.std_dev, size=action_dim)

    def reset(self, action_dim):
        pass

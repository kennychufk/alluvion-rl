import numpy as np


def evaluate_and_derive(x, f, ground_truth, loss_func, fd_size, *args):
    simulated_center = f(x, *args)
    loss_center = loss_func(ground_truth, simulated_center)
    n = len(x)
    if type(fd_size) is not np.ndarray:
        fd_size = np.repeat(fd_size, n)
    gradient = np.zeros(n)
    for i in range(n):
        delta_vector = np.zeros(n)
        delta_vector[i] = fd_size[i]
        simulated_plus = f(x + delta_vector, *args)
        loss_plus = loss_func(ground_truth, simulated_plus)
        gradient[i] = (loss_plus - loss_center) / delta_vector[i]
    return loss_center, gradient, simulated_center


class AdamOptim:
    def __init__(self, x, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros_like(x)
        self.v = np.zeros_like(x)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def update(self, f, ground_truth, loss_func, x, fd_size, *args):
        loss, g, simulated = evaluate_and_derive(x, f, ground_truth, loss_func,
                                                 fd_size, *args)
        new_x = np.zeros_like(x)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
        mhat = self.m / (1.0 - self.beta1**(self.t + 1))
        vhat = self.v / (1.0 - self.beta2**(self.t + 1))
        new_x = x - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        self.t += 1
        return new_x, loss, g, simulated

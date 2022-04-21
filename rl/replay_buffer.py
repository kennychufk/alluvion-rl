import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.state0 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.state1 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def store(self, state0, action, rew, state1, done):
        self.state0[self.ptr] = state0
        self.action[self.ptr] = action
        self.rew[self.ptr] = rew
        self.state1[self.ptr] = state1
        self.term[self.ptr] = 1 - done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_buffer(self, batch_size):
        idxs = random.sample(range(self.size), batch_size)
        selected_vectors = [
            self.state0[idxs], self.action[idxs], self.rew[idxs],
            self.state1[idxs], self.term[idxs]
        ]
        return [
            torch.tensor(v, dtype=torch.float).to(torch.device('cuda'))
            for v in selected_vectors
        ]

    def load(self, containing_dir):
        self.state0 = np.load(f'{containing_dir}/state0.npy')
        self.action = np.load(f'{containing_dir}/action.npy')
        self.rew = np.load(f'{containing_dir}/rew.npy')
        self.state1 = np.load(f'{containing_dir}/state1.npy')
        self.term = np.load(f'{containing_dir}/term.npy')
        self.capacity = np.load(f'{containing_dir}/capacity.npy').item()
        self.size = np.load(f'{containing_dir}/size.npy').item()
        self.ptr = np.load(f'{containing_dir}/ptr.npy').item()

    def save(self, containing_dir):
        np.save(f'{containing_dir}/state0.npy', self.state0)
        np.save(f'{containing_dir}/action.npy', self.action)
        np.save(f'{containing_dir}/rew.npy', self.rew)
        np.save(f'{containing_dir}/state1.npy', self.state1)
        np.save(f'{containing_dir}/term.npy', self.term)
        np.save(f'{containing_dir}/capacity.npy', self.capacity)
        np.save(f'{containing_dir}/size.npy', self.size)
        np.save(f'{containing_dir}/ptr.npy', self.ptr)

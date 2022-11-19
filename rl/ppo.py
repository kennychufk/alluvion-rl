import os

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_sizes, min_action,
                 max_action):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        self.action_half_extent = (max_action - min_action) * 0.5
        self.action_mean = (max_action + min_action) * 0.5
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([state_dim] + list(hidden_sizes) + [action_dim],
                          nn.Tanh)

    def _distribution(self, state):
        mu = self.mu_net(state)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, action):
        return pi.log_prob(action).sum(axis=-1)

    def forward(self, state, action=None):
        # Produce action distributions for given state, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(state)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(pi, action)
        return pi, logp_a

    def from_normalized_action(self, action):
        return self.action_half_extent * action + self.action_mean


class PPO(nn.Module):

    def __init__(self,
                 actor_lr,
                 critic_lr,
                 state_dim,
                 action_dim,
                 min_action,
                 max_action,
                 hidden_sizes,
                 num_steps_per_epoch,
                 clip_ratio=0.2,
                 train_actor_num_iterations=80,
                 train_critic_num_iterations=80,
                 gamma=0.99,
                 lam=0.97,
                 target_kl=0.01):
        super().__init__()

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.hidden_sizes = hidden_sizes
        self.num_steps_per_epoch = num_steps_per_epoch
        self.clip_ratio = clip_ratio
        self.train_actor_num_iterations = train_actor_num_iterations
        self.train_critic_num_iterations = train_critic_num_iterations
        self.gamma = gamma
        self.lam = lam
        self.target_kl = target_kl

        self.critic = mlp([state_dim] + list(hidden_sizes) + [1], nn.Tanh)
        self.critic.to(torch.device('cuda'))
        self.actor = MLPGaussianActor(state_dim, action_dim, hidden_sizes,
                                      min_action, max_action)
        self.actor.to(torch.device('cuda'))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)
        self.memory = PPOReplayBuffer(num_steps_per_epoch, state_dim,
                                      action_dim)

    def step(self, state):
        with torch.no_grad():
            state = torch.tensor(state,
                                 dtype=torch.float).to(torch.device('cuda'))
            pi = self.actor._distribution(state)
            action = pi.sample()
            logp_a = self.actor._log_prob_from_distribution(pi, action)
            value = self.critic(state)
        return action.cpu().detach().numpy(), value.cpu().detach().numpy(
        ), logp_a.cpu().detach().numpy()

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state,
                                 dtype=torch.float).to(torch.device('cuda'))
            pi = self.actor._distribution(state)
            action = pi.sample()
        return action.numpy()

    def compute_loss_pi(self, state, action, adv, logp):
        pi, logp_new = self.actor(state, action)
        ratio = torch.exp(logp_new - logp)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                               1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp - logp_new).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, state, ret):
        diff = self.critic(state) - ret
        return (diff * diff).mean()

    def update(self):
        state, action, ret, adv, logp = self.memory.get()
        pi_l_old, pi_info_old = self.compute_loss_pi(state, action, adv, logp)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(state, ret).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_num_iterations):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(state, action, adv, logp)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.actor_optimizer.step()

        # Value function learning
        for i in range(self.train_critic_num_iterations):
            self.critic_optimizer.zero_grad()
            loss_v = self.compute_loss_v(state, ret)
            loss_v.backward()
            self.critic_optimizer.step()

        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

    def save_models(self, parent_dir):
        torch.save(self.actor.state_dict(),
                   os.path.join(parent_dir, 'actor.pt'))
        torch.save(self.critic.state_dict(),
                   os.path.join(parent_dir, 'critic.pt'))

    def load_models(self, parent_dir):
        self.actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'actor.pt')))
        self.critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'critic.pt')))


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]


class PPOReplayBuffer:

    def __init__(self, capacity, state_dim, action_dim):
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.adv = np.zeros(capacity, dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.val = np.zeros(capacity, dtype=np.float32)
        self.logp = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.path_start_idx = 0
        self.ptr = 0

    def store(self, state, action, rew, val, logp):
        if self.ptr == self.capacity:
            raise Exception("PPOReplayBuffer overflow")
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val, gamma, lam):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew[path_slice], last_val)
        vals = np.append(self.val[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, gamma * lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret[path_slice] = discount_cumsum(rews, gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        if self.ptr != self.capacity:
            raise Exception("Buffer has to be full before get()")
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv.mean(), self.adv.std()
        self.adv_buf = (self.adv - adv_mean) / adv_std

        result = [self.state, self.action, self.ret, self.adv, self.logp]
        return [
            torch.tensor(v, dtype=torch.float).to(torch.device('cuda'))
            for v in result
        ]

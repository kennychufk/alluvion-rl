import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .replay_buffer import ReplayBuffer


class MLPTwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, final_layer_scale):
        super(MLPTwinCritic, self).__init__()
        net0_elements = []
        net1_elements = []
        layer_sizes = [state_dim + action_dim] + list(hidden_sizes) + [1]
        for i in range(len(layer_sizes) - 1):
            net0_elements.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            net1_elements.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                net0_elements.append(nn.ReLU())
                net1_elements.append(nn.ReLU())
        self.net0 = nn.Sequential(*net0_elements)
        self.net1 = nn.Sequential(*net1_elements)

        final_layer0 = [
            m for m in self.net0.modules() if not isinstance(m, nn.Sequential)
        ][-1]
        final_layer0.weight.data.mul_(final_layer_scale)
        final_layer0.bias.data.mul_(final_layer_scale)

        final_layer1 = [
            m for m in self.net1.modules() if not isinstance(m, nn.Sequential)
        ][-1]
        final_layer1.weight.data.mul_(final_layer_scale)
        final_layer1.bias.data.mul_(final_layer_scale)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net0(sa), self.net1(sa)

    def q0(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.net0(sa)


class MLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, min_action,
                 max_action, final_layer_scale):
        super(MLPActor, self).__init__()
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.action_half_extent = (max_action - min_action) * 0.5
        self.action_mean = (max_action + min_action) * 0.5
        elements = []
        layer_sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        for i in range(len(layer_sizes) - 1):
            elements.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                elements.append(nn.ReLU())
            else:
                elements.append(nn.Tanh())
        self.net = nn.Sequential(*elements)

        final_layer = [
            m for m in self.net.modules() if not isinstance(m, nn.Sequential)
        ][-2]
        final_layer.weight.data.mul_(final_layer_scale)
        final_layer.bias.data.mul_(final_layer_scale)

    def forward(self, state):
        return self.net(state)

    def from_normalized_action(self, action):
        return self.action_half_extent * action + self.action_mean


class TD3:
    def __init__(self,
                 actor_lr,
                 critic_lr,
                 critic_weight_decay,
                 state_dim,
                 action_dim,
                 hidden_sizes,
                 expl_noise_func,
                 soft_update_rate,
                 min_action,
                 max_action,
                 gamma=0.99,
                 replay_size=1000000,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_update_freq=2,
                 actor_final_scale=1,
                 critic_final_scale=1,
                 learn_after=25000,
                 batch_size=64):
        self.gamma = gamma
        self.soft_update_rate = soft_update_rate
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.learn_after = learn_after
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.critic_weight_decay = critic_weight_decay
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.actor_final_scale = actor_final_scale
        self.critic_final_scale = critic_final_scale
        self.memory = ReplayBuffer(replay_size, state_dim, action_dim)
        self.train_step = 0

        self.actor = MLPActor(state_dim, action_dim, hidden_sizes, min_action,
                              max_action, actor_final_scale)
        self.actor.to(torch.device('cuda'))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = MLPTwinCritic(state_dim, action_dim, hidden_sizes,
                                    critic_final_scale)
        self.critic.to(torch.device('cuda'))
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        self.target_actor = MLPActor(state_dim, action_dim, hidden_sizes,
                                     min_action, max_action, actor_final_scale)
        self.target_actor.to(torch.device('cuda'))
        self.target_critic = MLPTwinCritic(state_dim, action_dim, hidden_sizes,
                                           critic_final_scale)
        self.target_critic.to(torch.device('cuda'))

        self.expl_noise_func = expl_noise_func
        self.expl_noise_func.reset(action_dim)

        TD3.hard_update(self.target_actor, self.actor)
        TD3.hard_update(self.target_critic, self.critic)

    def uniform_random_action(self):
        return np.random.uniform(-1, 1, self.action_dim)

    def get_action(self, state, enable_noise=True):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).to(torch.device('cuda'))
        mu = self.actor.forward(state).cpu().detach().numpy()
        if enable_noise:
            mu += self.expl_noise_func(self.action_dim)
            np.clip(mu, -1, 1, out=mu)
        self.actor.train()
        return mu

    def get_value(self, state, action):
        return self.critic.forward(
            torch.tensor(state, dtype=torch.float).to(torch.device('cuda')),
            torch.tensor(action, dtype=torch.float).to(torch.device('cuda')))

    def remember(self, state0, action, rew, state1, done):
        self.memory.store(state0, action, rew, state1, done)

    def learn(self):
        self.train_step += 1
        state, action, reward, new_state, term = self.memory.sample_buffer(
            self.batch_size)

        # Optimize critic
        self.target_actor.eval()
        self.target_critic.eval()
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            new_action = (self.target_actor(new_state) + noise).clamp(-1, 1)

            target_q0, target_q1 = self.target_critic(new_state, new_action)
            target_q = torch.min(target_q0, target_q1)
            target_q = reward.view(
                self.batch_size,
                1) + self.gamma * term.view(self.batch_size, 1) * target_q

        current_q0, current_q1 = self.critic.forward(state, action)
        critic_loss = F.mse_loss(current_q0, target_q) + F.mse_loss(
            current_q1, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor
        if self.train_step % self.policy_update_freq == 0:
            self.critic.eval()
            for p in self.critic.parameters():
                p.requires_grad = False
            actor_loss = -self.critic.q0(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic.train()
            for p in self.critic.parameters():
                p.requires_grad = True

            TD3.soft_update(self.target_actor, self.actor,
                            self.soft_update_rate)
            TD3.soft_update(self.target_critic, self.critic,
                            self.soft_update_rate)

    @staticmethod
    def hard_update(dst_net, src_net):
        src_state_dict = src_net.state_dict()
        dst_state_dict = dst_net.state_dict()
        with torch.no_grad():
            for name, dst_param in dst_state_dict.items():
                dst_param.copy_(src_state_dict[name])

    @staticmethod
    def soft_update(dst_net, src_net, tau):
        src_state_dict = src_net.state_dict()
        dst_state_dict = dst_net.state_dict()
        with torch.no_grad():
            for name, dst_param in dst_state_dict.items():
                dst_param.mul_(1.0 - tau)
                dst_param.add_(src_state_dict[name] * tau)

    def save_models(self, parent_dir):
        torch.save(self.actor.state_dict(),
                   os.path.join(parent_dir, 'actor.pt'))
        torch.save(self.target_actor.state_dict(),
                   os.path.join(parent_dir, 'target_actor.pt'))
        torch.save(self.critic.state_dict(),
                   os.path.join(parent_dir, 'critic.pt'))
        torch.save(self.target_critic.state_dict(),
                   os.path.join(parent_dir, 'target_critic.pt'))

    def load_models(self, parent_dir):
        self.actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'actor.pt')))
        self.target_actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'target_actor.pt')))
        self.critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'critic.pt')))
        self.target_critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'target_critic.pt')))
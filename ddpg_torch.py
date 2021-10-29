import os

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, mu=0.0, sigma=0.2, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def __call__(self, act_dim):
        x = self.x_prev + self.theta * (
            self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
                self.dt) * np.random.normal(size=act_dim)
        self.x_prev = x
        return x

    def reset(self, act_dim):
        self.x_prev = np.random.normal(self.mu, self.sigma, act_dim)


class GaussianNoise:
    def __init__(self, std_dev=0.1):
        self.std_dev = std_dev

    def __call__(self, act_dim):
        return np.random.normal(0, self.std_dev, size=act_dim)

    def reset(self, act_dim):
        pass


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.obs0 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.obs1 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def store(self, obs0, act, rew, obs1, done):
        self.obs0[self.ptr] = obs0
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.obs1[self.ptr] = obs1
        self.term[self.ptr] = 1 - done
        if (self.ptr % 10000 == 0):
            print('obs', obs0, obs1)
            print('act rew term', act, rew, self.term[self.ptr])
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_buffer(self, batch_size):
        idxs = random.sample(range(self.size), batch_size)
        selected_vectors = [
            self.obs0[idxs], self.act[idxs], self.rew[idxs], self.obs1[idxs],
            self.term[idxs]
        ]
        return [
            torch.tensor(v, dtype=torch.float).to(torch.device('cuda'))
            for v in selected_vectors
        ]


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, final_layer_scale):
        super(MLPCritic, self).__init__()
        elements = []
        layer_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        for i in range(len(layer_sizes) - 1):
            elements.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                elements.append(nn.LayerNorm(layer_sizes[i + 1]))
                elements.append(nn.ReLU())
        self.net = nn.Sequential(*elements)

        final_layer = [
            m for m in self.net.modules() if not isinstance(m, nn.Sequential)
        ][-1]
        final_layer.weight.data.mul_(final_layer_scale)
        final_layer.bias.data.mul_(final_layer_scale)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], 1))


class MLPTwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, final_layer_scale):
        super(MLPTwinCritic, self).__init__()
        elements = []
        layer_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        for i in range(len(layer_sizes) - 1):
            elements.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                elements.append(nn.LayerNorm(layer_sizes[i + 1]))
                elements.append(nn.ReLU())
        self.net0 = nn.Sequential(*elements)
        self.net1 = nn.Sequential(*elements)

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

    def forward(self, obs, act):
        sa = torch.cat([obs, act], 1)
        return self.net0(sa), self.net1(sa)

    def q0(self, obs, act):
        sa = torch.cat([obs, act], 1)
        return self.net0(sa)


class MLPCriticMixAction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, final_layer_scale):
        super(MLPCriticMixAction, self).__init__()

        obs_net_elements = []
        obs_net_sizes = [obs_dim] + list(hidden_sizes)[:-1]
        for i in range(len(obs_net_sizes) - 1):
            obs_net_elements.append(
                nn.Linear(obs_net_sizes[i], obs_net_sizes[i + 1]))
            obs_net_elements.append(nn.LayerNorm(obs_net_sizes[i + 1]))
            obs_net_elements.append(nn.ReLU())
        self.obs_net = nn.Sequential(*obs_net_elements)

        mix_net_elements = []
        mix_net_sizes = list(hidden_sizes)[-1 - 1:] + [1]
        mix_net_sizes[0] += act_dim
        for i in range(len(mix_net_sizes) - 1):
            mix_net_elements.append(
                nn.Linear(mix_net_sizes[i], mix_net_sizes[i + 1]))
            if (i < len(mix_net_sizes) - 2):
                mix_net_elements.append(nn.ReLU())

        self.mix_net = nn.Sequential(*mix_net_elements)

        final_layer = [
            m for m in self.mix_net.modules()
            if not isinstance(m, nn.Sequential)
        ][-1]
        final_layer.weight.data.mul_(final_layer_scale)
        final_layer.bias.data.mul_(final_layer_scale)

    def forward(self, obs, act):
        obs_net_out = self.obs_net(obs)
        return self.mix_net(torch.cat([obs_net_out, act], 1))


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, min_action, max_action,
                 final_layer_scale):
        super(MLPActor, self).__init__()
        self.act_dim = act_dim
        self.min_action = min_action
        self.max_action = max_action
        self.action_half_extent = (max_action - min_action) * 0.5
        self.action_mean = (max_action + min_action) * 0.5
        elements = []
        layer_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        for i in range(len(layer_sizes) - 1):
            elements.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                elements.append(nn.LayerNorm(layer_sizes[i + 1]))
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

    def from_normalized_action(self, act):
        return self.action_half_extent * act + self.action_mean


class TD3:
    def __init__(self,
                 actor_lr,
                 critic_lr,
                 critic_weight_decay,
                 obs_dim,
                 act_dim,
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
                 actor_final_scale=0.051961524,
                 critic_final_scale=0.005196152,
                 learn_after=0,
                 batch_size=64):
        self.gamma = gamma
        self.soft_update_rate = soft_update_rate
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.learn_after = learn_after
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.critic_weight_decay = critic_weight_decay
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.actor_final_scale = actor_final_scale
        self.critic_final_scale = critic_final_scale
        self.memory = ReplayBuffer(replay_size, obs_dim, act_dim)
        self.train_step = 0

        self.actor = MLPActor(obs_dim, act_dim, hidden_sizes, min_action,
                              max_action, actor_final_scale)
        self.actor.to(torch.device('cuda'))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = MLPTwinCritic(obs_dim, act_dim, hidden_sizes,
                                    critic_final_scale)
        self.critic.to(torch.device('cuda'))
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        self.target_actor = MLPActor(obs_dim, act_dim, hidden_sizes,
                                     min_action, max_action, actor_final_scale)
        self.target_actor.to(torch.device('cuda'))
        self.target_critic = MLPTwinCritic(obs_dim, act_dim, hidden_sizes,
                                           critic_final_scale)
        self.target_critic.to(torch.device('cuda'))

        self.expl_noise_func = expl_noise_func
        self.expl_noise_func.reset(act_dim)

        DDPGAgent.hard_update(self.target_actor, self.actor)
        DDPGAgent.hard_update(self.target_critic, self.critic)

    def uniform_random_action(self):
        return np.random.uniform(-1, 1, self.act_dim)

    def get_action(self, observation, enable_noise=True):
        self.actor.eval()
        observation = torch.tensor(observation,
                                   dtype=torch.float).to(torch.device('cuda'))
        mu = self.actor.forward(observation).cpu().detach().numpy()
        if enable_noise:
            mu += self.expl_noise_func(self.act_dim)
            np.clip(mu, -1, 1, out=mu)
        self.actor.train()
        return mu

    def remember(self, obs0, act, rew, obs1, done):
        self.memory.store(obs0, act, rew, obs1, done)

    def learn(self):
        self.train_step += 1
        if len(self.memory) < max(self.batch_size, self.learn_after):
            return
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
            # TODO: is view() redundant?
            target_q = reward.view(
                self.batch_size,
                1) + self.gamma * term.view(self.batch_size, 1) * target_q

        self.critic_optimizer.zero_grad()
        current_q0, current_q1 = self.critic.forward(state, action)
        critic_loss = F.mse_loss(target_q, current_q0) + F.mse_loss(
            target_q, current_q1)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor
        if self.train_step % self.policy_update_freq == 0:
            self.critic.eval()
            for p in self.critic.parameters():
                p.requires_grad = False

            self.actor_optimizer.zero_grad()
            mu = self.actor.forward(state)
            # actor_loss = -self.critic.forward(state, mu)
            actor_loss = -self.critic.q0(state, mu)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic.train()
            for p in self.critic.parameters():
                p.requires_grad = True

            DDPGAgent.soft_update(self.target_actor, self.actor,
                                  self.soft_update_rate)
            DDPGAgent.soft_update(self.target_critic, self.critic,
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
                   os.path.join(parent_dir, 'Actor_ddpg'))
        torch.save(self.target_actor.state_dict(),
                   os.path.join(parent_dir, 'TargetActor_ddpg'))
        torch.save(self.critic.state_dict(),
                   os.path.join(parent_dir, 'Critic_ddpg'))
        torch.save(self.target_critic.state_dict(),
                   os.path.join(parent_dir, 'TargetCritic_ddpg'))

    def load_models(self, parent_dir):
        self.actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'Actor_ddpg')))
        self.target_actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'TargetActor_ddpg')))
        self.critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'Critic_ddpg')))
        self.target_critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'TargetCritic_ddpg')))


class DDPGAgent:
    def __init__(self,
                 actor_lr,
                 critic_lr,
                 critic_weight_decay,
                 obs_dim,
                 act_dim,
                 hidden_sizes,
                 expl_noise_func,
                 soft_update_rate,
                 min_action,
                 max_action,
                 gamma=0.99,
                 replay_size=1000000,
                 actor_final_scale=0.051961524,
                 critic_final_scale=0.005196152,
                 learn_after=0,
                 batch_size=64):
        self.gamma = gamma
        self.soft_update_rate = soft_update_rate
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.learn_after = learn_after
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.critic_weight_decay = critic_weight_decay
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.actor_final_scale = actor_final_scale
        self.critic_final_scale = critic_final_scale
        self.memory = ReplayBuffer(replay_size, obs_dim, act_dim)

        self.actor = MLPActor(obs_dim, act_dim, hidden_sizes, min_action,
                              max_action, actor_final_scale)
        self.actor.to(torch.device('cuda'))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = MLPCriticMixAction(obs_dim, act_dim, hidden_sizes,
                                         critic_final_scale)
        self.critic.to(torch.device('cuda'))
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        self.target_actor = MLPActor(obs_dim, act_dim, hidden_sizes,
                                     min_action, max_action, actor_final_scale)
        self.target_actor.to(torch.device('cuda'))
        self.target_critic = MLPCriticMixAction(obs_dim, act_dim, hidden_sizes,
                                                critic_final_scale)
        self.target_critic.to(torch.device('cuda'))

        self.expl_noise_func = expl_noise_func
        self.expl_noise_func.reset(act_dim)

        DDPGAgent.hard_update(self.target_actor, self.actor)
        DDPGAgent.hard_update(self.target_critic, self.critic)

    def uniform_random_action(self):
        return np.random.uniform(-1, 1, self.act_dim)

    def get_action(self, observation, enable_noise=True):
        self.actor.eval()
        observation = torch.tensor(observation,
                                   dtype=torch.float).to(torch.device('cuda'))
        mu = self.actor.forward(observation).cpu().detach().numpy()
        if enable_noise:
            mu += self.expl_noise_func(self.act_dim)
            np.clip(mu, -1, 1, out=mu)
        self.actor.train()
        return mu

    def remember(self, obs0, act, rew, obs1, done):
        self.memory.store(obs0, act, rew, obs1, done)

    def learn(self):
        if len(self.memory) < max(self.batch_size, self.learn_after):
            return
        state, action, reward, new_state, term = self.memory.sample_buffer(
            self.batch_size)

        # Optimize critic
        self.target_actor.eval()
        self.target_critic.eval()
        with torch.no_grad():
            target_actions = self.target_actor.forward(new_state)
            critic_value_ = self.target_critic.forward(new_state,
                                                       target_actions)

            target = reward.view(self.batch_size, 1) + self.gamma * term.view(
                self.batch_size, 1) * critic_value_

        self.critic_optimizer.zero_grad()
        critic_value = self.critic.forward(state, action)
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor
        self.critic.eval()
        for p in self.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        mu = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic.train()
        for p in self.critic.parameters():
            p.requires_grad = True

        DDPGAgent.soft_update(self.target_actor, self.actor,
                              self.soft_update_rate)
        DDPGAgent.soft_update(self.target_critic, self.critic,
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
                   os.path.join(parent_dir, 'Actor_ddpg'))
        torch.save(self.target_actor.state_dict(),
                   os.path.join(parent_dir, 'TargetActor_ddpg'))
        torch.save(self.critic.state_dict(),
                   os.path.join(parent_dir, 'Critic_ddpg'))
        torch.save(self.target_critic.state_dict(),
                   os.path.join(parent_dir, 'TargetCritic_ddpg'))

    def load_models(self, parent_dir):
        self.actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'Actor_ddpg')))
        self.target_actor.load_state_dict(
            torch.load(os.path.join(parent_dir, 'TargetActor_ddpg')))
        self.critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'Critic_ddpg')))
        self.target_critic.load_state_dict(
            torch.load(os.path.join(parent_dir, 'TargetCritic_ddpg')))

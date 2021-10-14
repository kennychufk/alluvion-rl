import argparse
import random
from collections import deque

import numpy as np
import torch
import gym
import wandb

from ddpg_torch import DDPGAgent, OrnsteinUhlenbeckProcess, GaussianNoise

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2021)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

env = gym.make('LunarLanderContinuous-v2')
env.seed(args.seed)

agent = DDPGAgent(actor_lr=1e-4,
                  critic_lr=1e-3,
                  critic_weight_decay=1e-2,
                  obs_dim=8,
                  act_dim=2,
                  max_action=env.action_space.high[0],
                  expl_noise_func=OrnsteinUhlenbeckProcess(),
                  hidden_sizes=[400, 300],
                  soft_update_rate=0.001,
                  batch_size=64)

wandb.init(project='rl-continuous')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.critic_weight_decay = agent.critic_weight_decay
config.obs_dim = agent.obs_dim
config.act_dim = agent.act_dim
config.hidden_sizes = agent.hidden_sizes
config.max_action = agent.target_actor.max_action
config.soft_update_rate = agent.soft_update_rate
config.gamma = agent.gamma
config.replay_size = agent.replay_size
config.actor_final_scale = agent.actor_final_scale
config.critic_final_scale = agent.critic_final_scale
config.sigma = agent.expl_noise_func.sigma
config.theta = agent.expl_noise_func.theta
config.learn_after = agent.learn_after
config.seed = args.seed

wandb.watch(agent.critic)

score_history = deque(maxlen=100)
with open('switch', 'w') as f:
    f.write('1')
i = 0
sample_step = 0
while True:
    with open('switch', 'r') as f:
        if f.read(1) == '0':
            break
    obs = env.reset()
    done = False
    score = 0
    while not done:
        if sample_step < agent.learn_after:
            act = agent.uniform_random_action()
        else:
            act = agent.get_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        # env.render()
    score_history.append(score)

    sample_step += 1
    if i % 50 == 0:
        agent.save_models(wandb.run.dir)
    wandb.log({'score': score, 'score100': np.mean(list(score_history))})
    i += 1

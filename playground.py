import sys
import argparse
import math
import random
import os
from collections import deque

import alluvion as al
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
import wandb
import torch
import time

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise
from util import Environment, EnvironmentPIV, get_state_dim, get_action_dim, eval_agent

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--block-scan', metavar='s', type=bool, default=False)
args = parser.parse_args()

train_dirs = [
    f"{args.truth_dir}/rltruth-be268318-0526.07.32.30/",
    f"{args.truth_dir}/rltruth-5caefe43-0526.14.46.12/",
    f"{args.truth_dir}/rltruth-e8edf09d-0526.18.34.19/",
    f"{args.truth_dir}/rltruth-6de1d91b-0526.09.31.47/",
    f"{args.truth_dir}/rltruth-3b860b54-0526.23.12.15/",
    f"{args.truth_dir}/rltruth-eb3494c1-0527.00.32.34/",
    f"{args.truth_dir}/rltruth-e9ba71d8-0527.01.52.31/"
]

# # 100 markers
# train_dirs = [
#     f"{args.truth_dir}/rltruth-70eb294d-0528.12.35.00/",
#     f"{args.truth_dir}/rltruth-9b57dabc-0528.17.09.24/",
#     f"{args.truth_dir}/rltruth-1b06a408-0528.18.46.50/",
#     f"{args.truth_dir}/rltruth-ffc2674d-0528.20.27.56/",
#     f"{args.truth_dir}/rltruth-f218e02c-0528.22.05.13/",
#     f"{args.truth_dir}/rltruth-a4f6c703-0528.23.50.15/",
#     f"{args.truth_dir}/rltruth-63caecfb-0529.01.28.09/"
# ]
#
# # shape reward
# train_dirs = [
#     f"{args.truth_dir}/rltruth-287fff2f-0606.02.28.24/",
#     f"{args.truth_dir}/rltruth-749f1ce5-0606.04.12.08/",
#     f"{args.truth_dir}/rltruth-e54c4fb9-0606.05.55.43/",
#     f"{args.truth_dir}/rltruth-433ce26e-0606.07.41.42/"
# ]
#
# # density reward
# train_dirs = [
#     f"{args.truth_dir}/rltruth-365f1a1d-0606.18.57.29/",
#     f"{args.truth_dir}/rltruth-578d33eb-0606.19.16.45/",
#     f"{args.truth_dir}/rltruth-09b70888-0606.19.35.41/",
#     f"{args.truth_dir}/rltruth-5f6f0042-0606.19.57.18/"
# ]
dp = al.Depot(np.float32)
ma_alphas = [0.0625, 0.125, 0.25, 0.4]
env = Environment(dp,
                  truth_dirs=train_dirs,
                  cache_dir=args.cache_dir,
                  ma_alphas=ma_alphas,
                  display=args.display)
env.seed(args.seed)

max_xoffset = 0.1
max_voffset = 0.1
max_focal_dist = 0.20
min_usher_kernel_radius = 0.01
max_usher_kernel_radius = 0.12
max_strength = 1000

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            state_dim=get_state_dim(),
            action_dim=get_action_dim(),
            expl_noise_func=GaussianNoise(),
            gamma=0.95,
            min_action=np.array([
                -max_xoffset, -max_xoffset, -max_xoffset, -max_voffset,
                -max_voffset, -max_voffset, min_usher_kernel_radius, 0
            ]),
            max_action=np.array([
                +max_xoffset, +max_xoffset, +max_xoffset, +max_voffset,
                +max_voffset, +max_voffset, max_usher_kernel_radius,
                max_strength
            ]),
            learn_after=10000,
            replay_size=36000000,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=0.05 / np.sqrt(1000),
            critic_final_scale=1,
            soft_update_rate=0.005,
            batch_size=256)
max_timesteps = 10000000
if args.block_scan:
    max_timesteps = 1000

wandb.init(project='alluvion-rl')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.critic_weight_decay = agent.critic_weight_decay
config.state_dim = agent.state_dim
config.action_dim = agent.action_dim
config.hidden_sizes = agent.hidden_sizes
config.max_action = agent.target_actor.max_action
config.min_action = agent.target_actor.min_action
config.soft_update_rate = agent.soft_update_rate
config.gamma = agent.gamma
config.replay_size = agent.replay_size
config.actor_final_scale = agent.actor_final_scale
config.critic_final_scale = agent.critic_final_scale
# config.sigma = agent.expl_noise_func.sigma
# config.theta = agent.expl_noise_func.theta
config.learn_after = agent.learn_after
config.batch_size = agent.batch_size
config.seed = args.seed
config.ma_alphas = env.ma_alphas

wandb.watch(agent.critic)

score_history = deque(maxlen=100)
episode_id = 0
episode_t = 0
episode_reward = 0
episode_info = {}
state_aggregated = env.reset()
done = False

piv_truth_dirs = [
    '/media/kennychufk/vol1bk0/20210415_162749-laser-too-high/',
    '/media/kennychufk/vol1bk0/20210415_164304/',
    '/media/kennychufk/vol1bk0/20210416_101435/',
    '/media/kennychufk/vol1bk0/20210416_102548/',
    '/media/kennychufk/vol1bk0/20210416_103739/',
    '/media/kennychufk/vol1bk0/20210416_104936/',
    '/media/kennychufk/vol1bk0/20210416_120534/'
]
env_piv = EnvironmentPIV(dp,
                         truth_dirs=piv_truth_dirs,
                         cache_dir=args.cache_dir,
                         ma_alphas=config['ma_alphas'],
                         display=args.display,
                         volume_method=al.VolumeMethod.pellets)

for t in range(max_timesteps):
    episode_t += 1
    if t < agent.learn_after:
        action_aggregated = np.zeros((env.num_buoys, agent.action_dim))
        for buoy_id in range(env.num_buoys):
            action_aggregated[buoy_id] = agent.uniform_random_action()
    else:
        action_aggregated = agent.get_action(state_aggregated)
    new_state_aggregated, reward, done, info = env.step(
        agent.actor.from_normalized_action(action_aggregated))
    # done_int = int(done) if episode_t < env._max_episode_steps else 0
    done_int = int(done)

    for buoy_id in range(env.num_buoys):
        agent.remember(state_aggregated[buoy_id], action_aggregated[buoy_id],
                       reward, new_state_aggregated[buoy_id], done_int)
    episode_reward += reward
    state_aggregated = new_state_aggregated
    for key in info:
        if key not in episode_info:
            episode_info[key] = 0
        episode_info[key] += info[key]

    if t >= agent.learn_after:  # as memory size is env.num_buoys * episode_t
        agent.learn()

    if done:
        score_history.append(episode_reward)
        log_object = {'score': episode_reward}
        if len(score_history) == score_history.maxlen:
            log_object['score100'] = np.mean(list(score_history))

        for key in episode_info:
            if (key != 'truth_sqr') and (key != 'num_masked'):
                log_object[key] = episode_info[key]

        episode_id += 1
        episode_t = 0
        episode_reward = 0
        episode_info = {}
        state_aggregated = env.reset()
        done = False

        if episode_id % 50 == 0:
            result_dict = {}
            log_object['val-piv'] = eval_agent(env_piv,
                                               agent,
                                               result_dict,
                                               report_state_action=False)
            for result_key in result_dict:
                log_object[result_key] = result_dict[result_key]
            print('result_dict', result_dict)
            print('log_object', log_object)
            save_dir = f"artifacts/{wandb.run.id}/models/{episode_id}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            agent.save_models(save_dir)

        wandb.log(log_object)

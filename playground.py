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
    f"{args.truth_dir}/diagonal-train3/rltruth-02e8c63b-0908.22.23.33",
    f"{args.truth_dir}/diagonal-train3/rltruth-03104431-0909.00.37.02",
    f"{args.truth_dir}/diagonal-train3/rltruth-0642071e-0909.01.44.23",
    f"{args.truth_dir}/diagonal-train3/rltruth-0974a500-0909.06.37.46",
    f"{args.truth_dir}/diagonal-train3/rltruth-0a42f27e-0909.01.23.13",
    f"{args.truth_dir}/diagonal-train3/rltruth-0af9f49e-0908.23.43.02",
    f"{args.truth_dir}/diagonal-train3/rltruth-0b743a4c-0909.04.42.26",
    f"{args.truth_dir}/diagonal-train3/rltruth-164ecf83-0908.21.14.19",
    f"{args.truth_dir}/diagonal-train3/rltruth-18bf36fe-0908.19.17.41",
    f"{args.truth_dir}/diagonal-train3/rltruth-1d9cb091-0908.22.50.30",
    f"{args.truth_dir}/diagonal-train3/rltruth-20e50ebe-0909.02.38.34",
    f"{args.truth_dir}/diagonal-train3/rltruth-26873fec-0908.18.27.56",
    f"{args.truth_dir}/diagonal-train3/rltruth-355c0455-0909.07.06.07",
    f"{args.truth_dir}/diagonal-train3/rltruth-3b1342a2-0909.05.03.28",
    f"{args.truth_dir}/diagonal-train3/rltruth-41a67e4b-0908.12.10.51",
    f"{args.truth_dir}/diagonal-train3/rltruth-450ca6ba-0908.21.56.47",
    f"{args.truth_dir}/diagonal-train3/rltruth-4638c729-0908.19.42.03",
    f"{args.truth_dir}/diagonal-train3/rltruth-55f957a7-0909.03.27.12",
    f"{args.truth_dir}/diagonal-train3/rltruth-60a7de51-0908.16.18.13",
    f"{args.truth_dir}/diagonal-train3/rltruth-64e8ce75-0908.17.07.31",
    f"{args.truth_dir}/diagonal-train3/rltruth-65daf30b-0908.11.43.28",
    f"{args.truth_dir}/diagonal-train3/rltruth-67bffda4-0908.21.35.23",
    f"{args.truth_dir}/diagonal-train3/rltruth-69390159-0909.05.54.42",
    f"{args.truth_dir}/diagonal-train3/rltruth-74ec6b40-0909.03.52.11",
    f"{args.truth_dir}/diagonal-train3/rltruth-7b571e20-0908.14.22.06",
    f"{args.truth_dir}/diagonal-train3/rltruth-8ba8447e-0909.06.15.58",
    f"{args.truth_dir}/diagonal-train3/rltruth-92cc4964-0908.23.19.45",
    f"{args.truth_dir}/diagonal-train3/rltruth-9af872fc-0908.13.25.24",
    f"{args.truth_dir}/diagonal-train3/rltruth-9e164c09-0908.16.47.12",
    f"{args.truth_dir}/diagonal-train3/rltruth-a0879469-0908.13.53.36",
    f"{args.truth_dir}/diagonal-train3/rltruth-a129c81a-0908.12.36.59",
    f"{args.truth_dir}/diagonal-train3/rltruth-a3f24348-0909.02.58.46",
    f"{args.truth_dir}/diagonal-train3/rltruth-a84d73b5-0908.20.25.22",
    f"{args.truth_dir}/diagonal-train3/rltruth-ab794f5a-0908.17.28.19",
    f"{args.truth_dir}/diagonal-train3/rltruth-aec7b749-0909.04.12.44",
    f"{args.truth_dir}/diagonal-train3/rltruth-aef45da2-0909.05.28.48",
    f"{args.truth_dir}/diagonal-train3/rltruth-be19ef90-0908.17.58.10",
    f"{args.truth_dir}/diagonal-train3/rltruth-d6ae892e-0908.15.36.07",
    f"{args.truth_dir}/diagonal-train3/rltruth-d9caf75f-0908.15.15.52",
    f"{args.truth_dir}/diagonal-train3/rltruth-e123014d-0909.00.04.21",
    f"{args.truth_dir}/diagonal-train3/rltruth-e52dfc81-0908.12.58.00",
    f"{args.truth_dir}/diagonal-train3/rltruth-ead00588-0908.14.49.19",
    f"{args.truth_dir}/diagonal-train3/rltruth-f2eedd09-0909.02.11.24",
    f"{args.truth_dir}/diagonal-train3/rltruth-f4aa23b1-0908.20.04.33",
    f"{args.truth_dir}/diagonal-train3/rltruth-f807cdaa-0908.18.56.23",
    f"{args.truth_dir}/diagonal-train3/rltruth-fdf6749b-0908.15.55.43",
    f"{args.truth_dir}/diagonal-train3/rltruth-fdf98b0c-0908.20.46.09",
    f"{args.truth_dir}/diagonal-train3/rltruth-fe144a96-0909.00.58.36",
]

dp = al.Depot(np.float32)
ma_alphas = [0.0625, 0.125, 0.25, 0.4]
env = Environment(dp,
                  truth_dirs=train_dirs,
                  cache_dir=args.cache_dir,
                  ma_alphas=ma_alphas,
                  display=args.display)
env.seed(args.seed)

min_xoffset_y = -0.02
max_xoffset_y = 0.1
max_xoffset = 0.1
max_voffset = 0.1
max_focal_dist = 0.20
min_usher_kernel_radius = 0.01
max_usher_kernel_radius = 0.12
max_strength = 25

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            state_dim=get_state_dim(),
            action_dim=get_action_dim(),
            expl_noise_func=GaussianNoise(),
            gamma=0.95,
            min_action=np.array([
                -max_xoffset, min_xoffset_y, -max_xoffset, -max_voffset,
                -max_voffset, -max_voffset, min_usher_kernel_radius, 0
            ]),
            max_action=np.array([
                +max_xoffset, max_xoffset_y, +max_xoffset, +max_voffset,
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

# piv_truth_dirs = [
#     '/media/kennychufk/vol1bk0/20210415_162749-laser-too-high/',
#     '/media/kennychufk/vol1bk0/20210415_164304/',
#     '/media/kennychufk/vol1bk0/20210416_101435/',
#     '/media/kennychufk/vol1bk0/20210416_102548/',
#     '/media/kennychufk/vol1bk0/20210416_103739/',
#     '/media/kennychufk/vol1bk0/20210416_104936/',
#     '/media/kennychufk/vol1bk0/20210416_120534/'
# ]
# env_piv = EnvironmentPIV(dp,
#                          truth_dirs=piv_truth_dirs,
#                          cache_dir=args.cache_dir,
#                          ma_alphas=config['ma_alphas'],
#                          display=args.display,
#                          volume_method=al.VolumeMethod.pellets)

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
            # result_dict = {}
            # log_object['val-sim'] = eval_agent(env_val_sim,
            #                                    agent,
            #                                    result_dict,
            #                                    report_state_action=False)
            # for result_key in result_dict:
            #     log_object[result_key] = result_dict[result_key]
            # print('result_dict', result_dict)
            print('log_object', log_object)
            save_dir = f"artifacts/{wandb.run.id}/models/{episode_id}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            agent.save_models(save_dir)

        wandb.log(log_object)

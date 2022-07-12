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
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-056411d7-0705.06.09.11",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-073e1145-0708.00.04.08",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-12dedbfc-0705.05.08.10",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-185a23ff-0705.03.07.02",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-1d0da5d2-0708.02.09.42",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-246e1ee1-0708.03.51.39",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-2f9a4640-0708.05.09.44",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-3451c452-0705.02.26.54",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-40651d5b-0705.08.11.08",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-509f0cf0-0708.00.26.26",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-5238f2eb-0708.01.07.14",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-54b4ec71-0705.03.50.08",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-5aac6599-0708.04.30.28",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-6111aeb5-0708.02.29.27",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-64de5e29-0705.05.27.52",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-80345e00-0705.07.51.42",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-81b8e237-0705.02.47.21",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-870c772f-0708.03.11.23",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-8e691ad8-0705.06.29.24",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-979ff2b9-0705.04.08.50",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-9eb4691e-0708.04.10.33",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-a0d2056d-0705.07.28.34",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-a18cd9d2-0708.05.29.20",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-a31542b5-0708.01.50.00",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-b703432b-0705.06.48.40",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-c0a86c7b-0708.01.27.48",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-c1cbf6bc-0708.00.46.24",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-c1ef461b-0705.04.28.09",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-ce036240-0705.05.48.26",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-ce715c42-0705.03.27.07",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-ceee4f09-0708.03.31.10",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-d6dbb185-0708.02.51.13",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-d8e212e6-0705.07.08.04",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-db4cb3e1-0708.04.49.54",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-dd42d0a4-0708.05.48.53",
    f"{args.truth_dir}/ou-9-buoyancy/rltruth-f02e86e5-0705.04.48.37",
]

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
max_strength = 25

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

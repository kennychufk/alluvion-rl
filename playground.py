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
    f"{args.truth_dir}/diagonal-train/rltruth-05b9002e-0626.23.55.32",
    f"{args.truth_dir}/diagonal-train/rltruth-0639c442-0627.06.28.56",
    f"{args.truth_dir}/diagonal-train/rltruth-073f81a0-0627.14.12.07",
    f"{args.truth_dir}/diagonal-train/rltruth-0935e031-0627.05.37.45",
    f"{args.truth_dir}/diagonal-train/rltruth-1809a415-0627.12.10.08",
    f"{args.truth_dir}/diagonal-train/rltruth-1a9e0c7d-0627.18.53.14",
    f"{args.truth_dir}/diagonal-train/rltruth-1bba526e-0627.03.45.42",
    f"{args.truth_dir}/diagonal-train/rltruth-221428bf-0626.23.14.17",
    f"{args.truth_dir}/diagonal-train/rltruth-25b5e42c-0627.03.01.29",
    f"{args.truth_dir}/diagonal-train/rltruth-351bf7c9-0627.01.38.42",
    f"{args.truth_dir}/diagonal-train/rltruth-428a04c9-0627.04.30.52",
    f"{args.truth_dir}/diagonal-train/rltruth-5190e0d0-0627.13.08.38",
    f"{args.truth_dir}/diagonal-train/rltruth-527bc842-0627.08.04.54",
    f"{args.truth_dir}/diagonal-train/rltruth-5a5828b5-0627.01.18.01",
    f"{args.truth_dir}/diagonal-train/rltruth-5d6b2c73-0627.03.23.18",
    f"{args.truth_dir}/diagonal-train/rltruth-64276db3-0627.02.19.41",
    f"{args.truth_dir}/diagonal-train/rltruth-6480b8ec-0627.18.21.33",
    f"{args.truth_dir}/diagonal-train/rltruth-69eeec6e-0627.09.24.14",
    f"{args.truth_dir}/diagonal-train/rltruth-6a7d0a1f-0627.00.36.16",
    f"{args.truth_dir}/diagonal-train/rltruth-6ceeafcc-0627.15.31.46",
    f"{args.truth_dir}/diagonal-train/rltruth-6e715a0e-0627.17.21.44",
    f"{args.truth_dir}/diagonal-train/rltruth-72871e9b-0627.09.49.50",
    f"{args.truth_dir}/diagonal-train/rltruth-7940df8d-0627.01.58.48",
    f"{args.truth_dir}/diagonal-train/rltruth-8450e44f-0627.04.08.28",
    f"{args.truth_dir}/diagonal-train/rltruth-9207bc09-0627.00.16.14",
    f"{args.truth_dir}/diagonal-train/rltruth-9258b1e2-0626.22.53.18",
    f"{args.truth_dir}/diagonal-train/rltruth-93bb87a0-0627.05.16.17",
    f"{args.truth_dir}/diagonal-train/rltruth-9613fdeb-0626.23.36.06",
    f"{args.truth_dir}/diagonal-train/rltruth-9dde1752-0627.06.53.11",
    f"{args.truth_dir}/diagonal-train/rltruth-9f57c92e-0627.12.43.29",
    f"{args.truth_dir}/diagonal-train/rltruth-9fff70e1-0627.14.43.43",
    f"{args.truth_dir}/diagonal-train/rltruth-a815ef81-0627.08.56.16",
    f"{args.truth_dir}/diagonal-train/rltruth-b3516552-0627.16.11.20",
    f"{args.truth_dir}/diagonal-train/rltruth-ba67943c-0627.04.53.27",
    f"{args.truth_dir}/diagonal-train/rltruth-c0e6df6a-0627.16.49.39",
    f"{args.truth_dir}/diagonal-train/rltruth-c12dc2aa-0627.10.47.58",
    f"{args.truth_dir}/diagonal-train/rltruth-c8c6a155-0627.10.20.44",
    f"{args.truth_dir}/diagonal-train/rltruth-cd3fe9fc-0627.15.51.02",
    f"{args.truth_dir}/diagonal-train/rltruth-cfe678e9-0627.06.04.08",
    f"{args.truth_dir}/diagonal-train/rltruth-d4052d93-0627.00.56.40",
    f"{args.truth_dir}/diagonal-train/rltruth-d43edaa5-0627.11.40.55",
    f"{args.truth_dir}/diagonal-train/rltruth-d4a1873e-0627.02.39.53",
    f"{args.truth_dir}/diagonal-train/rltruth-d5d36ed0-0627.08.30.52",
    f"{args.truth_dir}/diagonal-train/rltruth-ee2d5ad1-0627.07.42.28",
    f"{args.truth_dir}/diagonal-train/rltruth-ef5ab1b6-0627.16.30.05",
    f"{args.truth_dir}/diagonal-train/rltruth-f1945362-0627.17.48.28",
    f"{args.truth_dir}/diagonal-train/rltruth-f57d9cc1-0627.07.16.07",
    f"{args.truth_dir}/diagonal-train/rltruth-f6a7f70d-0627.11.15.02",
    f"{args.truth_dir}/diagonal-train/rltruth-fed7632f-0627.13.40.41",
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

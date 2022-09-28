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
    f"{args.truth_dir}/diagonal-train4/rltruth-12ff5012-0914.15.52.21",
    f"{args.truth_dir}/diagonal-train4/rltruth-14f08550-0915.06.44.03",
    f"{args.truth_dir}/diagonal-train4/rltruth-1654bb10-0914.23.56.12",
    f"{args.truth_dir}/diagonal-train4/rltruth-1af3c395-0915.04.02.08",
    f"{args.truth_dir}/diagonal-train4/rltruth-1dc784ae-0914.06.08.59",
    f"{args.truth_dir}/diagonal-train4/rltruth-227ecffa-0915.05.25.49",
    f"{args.truth_dir}/diagonal-train4/rltruth-22ba32a5-0914.05.33.45",
    f"{args.truth_dir}/diagonal-train4/rltruth-2a7d4859-0914.17.10.05",
    f"{args.truth_dir}/diagonal-train4/rltruth-2a98f2fb-0914.14.42.39",
    f"{args.truth_dir}/diagonal-train4/rltruth-2d482973-0915.01.58.26",
    f"{args.truth_dir}/diagonal-train4/rltruth-2d9afd64-0914.13.35.00",
    f"{args.truth_dir}/diagonal-train4/rltruth-312c0cc8-0914.11.53.13",
    f"{args.truth_dir}/diagonal-train4/rltruth-33cbd54f-0914.03.59.31",
    f"{args.truth_dir}/diagonal-train4/rltruth-350aefb5-0914.05.02.10",
    f"{args.truth_dir}/diagonal-train4/rltruth-3c127f13-0914.08.55.27",
    f"{args.truth_dir}/diagonal-train4/rltruth-448949d6-0914.03.23.47",
    f"{args.truth_dir}/diagonal-train4/rltruth-44eba036-0915.04.44.08",
    f"{args.truth_dir}/diagonal-train4/rltruth-5640dc4e-0914.20.17.18",
    f"{args.truth_dir}/diagonal-train4/rltruth-5d9f1cd4-0914.17.47.12",
    f"{args.truth_dir}/diagonal-train4/rltruth-65288f7b-0915.00.38.11",
    f"{args.truth_dir}/diagonal-train4/rltruth-73469bdd-0914.06.43.30",
    f"{args.truth_dir}/diagonal-train4/rltruth-8150d9a2-0914.18.58.36",
    f"{args.truth_dir}/diagonal-train4/rltruth-87b12884-0914.10.41.05",
    f"{args.truth_dir}/diagonal-train4/rltruth-882bd11e-0914.18.20.53",
    f"{args.truth_dir}/diagonal-train4/rltruth-8834d9d4-0914.19.38.36",
    f"{args.truth_dir}/diagonal-train4/rltruth-8cecae4f-0914.01.50.28",
    f"{args.truth_dir}/diagonal-train4/rltruth-8dfe9db8-0914.07.17.17",
    f"{args.truth_dir}/diagonal-train4/rltruth-9196b61d-0914.22.41.50",
    f"{args.truth_dir}/diagonal-train4/rltruth-9c9a292c-0914.23.16.52",
    f"{args.truth_dir}/diagonal-train4/rltruth-9d4aaf45-0914.02.52.46",
    f"{args.truth_dir}/diagonal-train4/rltruth-a35ca720-0914.07.49.48",
    f"{args.truth_dir}/diagonal-train4/rltruth-a93a68aa-0914.12.26.32",
    f"{args.truth_dir}/diagonal-train4/rltruth-aff784dd-0914.22.04.00",
    f"{args.truth_dir}/diagonal-train4/rltruth-b5351c5b-0914.21.28.35",
    f"{args.truth_dir}/diagonal-train4/rltruth-b5a1fc48-0914.14.09.07",
    f"{args.truth_dir}/diagonal-train4/rltruth-b79afb3c-0914.16.27.42",
    f"{args.truth_dir}/diagonal-train4/rltruth-b9b86c25-0914.12.59.04",
    f"{args.truth_dir}/diagonal-train4/rltruth-b9d5ee38-0914.20.51.11",
    f"{args.truth_dir}/diagonal-train4/rltruth-cada1ed3-0914.10.04.19",
    f"{args.truth_dir}/diagonal-train4/rltruth-cf367ec6-0914.15.16.06",
    f"{args.truth_dir}/diagonal-train4/rltruth-d24a33dc-0914.09.29.59",
    f"{args.truth_dir}/diagonal-train4/rltruth-d4537532-0914.04.30.48",
    f"{args.truth_dir}/diagonal-train4/rltruth-d949a937-0914.02.21.44",
    f"{args.truth_dir}/diagonal-train4/rltruth-d94bc10d-0915.02.33.42",
    f"{args.truth_dir}/diagonal-train4/rltruth-dbdf3f05-0914.01.20.58",
    f"{args.truth_dir}/diagonal-train4/rltruth-e2f978b2-0915.03.10.03",
    f"{args.truth_dir}/diagonal-train4/rltruth-ebc119cb-0914.08.22.41",
    f"{args.truth_dir}/diagonal-train4/rltruth-ee7af950-0915.06.02.41",
    f"{args.truth_dir}/diagonal-train4/rltruth-f01c96fb-0915.01.15.43",
]

dp = al.Depot(np.float32)
ma_alphas = [0.0625, 0.125, 0.25, 0.4]
env = Environment(dp,
                  truth_dirs=train_dirs,
                  cache_dir=args.cache_dir,
                  ma_alphas=ma_alphas,
                  display=args.display,
                  reward_option=1)
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
config.reward_option = env.reward_option

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

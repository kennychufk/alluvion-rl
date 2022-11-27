import sys
import argparse
from collections import deque

import alluvion as al
import numpy as np
from pathlib import Path
import wandb
import torch
import time

from rl import TD3, GaussianNoise
from util import Environment, get_state_dim, get_action_dim

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--block-scan', metavar='s', type=bool, default=False)
args = parser.parse_args()

train_dirs = [
    f"{args.truth_dir}/parametric-star/rltruth-fc64ec80-1025.14.38.30",  # 4
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-b127216c-1026.14.24.40",  # 4
    f"{args.truth_dir}/parametric-nephroid/rltruth-34970c41-1102.01.21.46",  # 4
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-dda86d58-1103.13.15.01",  # 4
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-963b7147-1015.09.48.23",  # 5
    f"{args.truth_dir}/parametric-star/rltruth-206a1878-1025.15.35.07",  # 6
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-ca7539a9-1026.15.14.43",  # 6
    f"{args.truth_dir}/parametric-nephroid/rltruth-f0fef677-1102.02.03.35",  # 6
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-529a44ef-1103.13.56.55",  # 6
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-61b17546-1015.04.09.38",  # 10
    f"{args.truth_dir}/parametric-star/rltruth-5a06fb70-1025.16.03.37",  # 8
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-3bce54ea-1026.16.14.39",  # 8
    f"{args.truth_dir}/parametric-nephroid/rltruth-7633a710-1102.02.42.52",  # 8
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-c4dcb856-1103.14.45.27",  # 8
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-c0a38d05-1015.03.20.54",  # 11
    f"{args.truth_dir}/parametric-star/rltruth-a59c4482-1025.16.32.10",  # 10
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-2c3e65d1-1026.17.06.50",  # 10
    f"{args.truth_dir}/parametric-nephroid/rltruth-77a3b794-1102.03.02.55",  # 10
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-b26a5d76-1103.15.05.56",  # 10
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-cec5603e-1014.23.54.16",  # 12
    f"{args.truth_dir}/parametric-star/rltruth-ac940701-1025.17.11.51",  # 12
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-e49e783b-1026.18.01.59",  # 12
    f"{args.truth_dir}/parametric-nephroid/rltruth-68eb3a7b-1102.03.22.04",  # 12
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-35ea7d2a-1103.15.25.00",  # 12
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-370823da-1015.02.02.27",  # 13
    f"{args.truth_dir}/parametric-star/rltruth-82e1bac6-1025.18.21.28",  # 16
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-42402e06-1026.19.46.41",  # 16
    f"{args.truth_dir}/parametric-nephroid/rltruth-4e1f0fe8-1102.03.41.23",  # 16
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-3c6bfdb7-1103.15.47.41",  # 16
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-f39e190c-1015.06.35.08",  # 14
    f"{args.truth_dir}/parametric-star/rltruth-a37105f3-1025.20.02.24",  # 22
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-acebfc56-1026.22.31.20",  # 22
    f"{args.truth_dir}/parametric-nephroid/rltruth-6099faf5-1102.04.02.53",  # 22
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-13efbe11-1103.16.08.26",  # 22
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-94ac4e05-1014.21.07.35",  # 21
    f"{args.truth_dir}/parametric-star/rltruth-10f9e2f9-1025.22.30.48",  # 30
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-c9547f0f-1027.02.26.35",  # 30
    f"{args.truth_dir}/parametric-nephroid/rltruth-c85e79a2-1102.04.23.40",  # 30
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-0952ce06-1103.16.35.52",  # 30
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-75124ebf-1015.08.08.44",  # 27
    f"{args.truth_dir}/parametric-star/rltruth-2304995d-1026.01.27.09",  # 40
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-a8fc3f9f-1027.07.12.58",  # 40
    f"{args.truth_dir}/parametric-nephroid/rltruth-764fe695-1102.04.52.44",  # 40
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-b386c0be-1103.16.59.50",  # 40
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-e0a74210-1015.02.57.18",  # 39
    f"{args.truth_dir}/parametric-star/rltruth-231b76ba-1026.04.52.43",  # 52
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-354803e4-1027.11.40.25",  # 52
    f"{args.truth_dir}/parametric-nephroid/rltruth-7b60a53f-1102.05.16.22",  # 52
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-12eb20c5-1103.17.26.29",  # 52
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-e81cdc75-1015.08.30.40",  # 54
    f"{args.truth_dir}/parametric-star/rltruth-de256931-1026.11.54.55",  # 66
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-be3012ff-1027.14.45.16",  # 66
    f"{args.truth_dir}/parametric-nephroid/rltruth-ab55bae7-1102.05.41.34",  # 66
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-17e39f8a-1103.17.51.14",  # 66
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-61380666-1015.05.21.56",  # 62
    f"{args.truth_dir}/parametric-star/rltruth-988675e6-1026.16.30.21",  # 82
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-250e3e81-1028.18.15.43",  # 82
    f"{args.truth_dir}/parametric-nephroid/rltruth-099b5858-1102.06.10.45",  # 82
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-d0a085dc-1103.18.18.39",  # 82
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-ebb22bdd-1015.03.42.01",  # 87
    f"{args.truth_dir}/parametric-star/rltruth-3fc9006b-1026.22.58.05",  # 100
    f"{args.truth_dir}/parametric-bidir-circles/rltruth-bb7b2551-1028.22.54.12",  # 100
    f"{args.truth_dir}/parametric-nephroid/rltruth-639de5d5-1102.06.37.34",  # 100
    f"{args.truth_dir}/parametric-epicycloid2/rltruth-939cb2e9-1103.18.47.37",  # 100
    f"{args.truth_dir}/parametric-interesting-loop/rltruth-9c477345-1015.09.16.03",  # 95
]

dp = al.Depot(np.float32)
ma_alphas = [0.0625, 0.125, 0.25, 0.4]
local_reward_ratio = 0.1
env = Environment(dp,
                  truth_dirs=train_dirs,
                  cache_dir=args.cache_dir,
                  ma_alphas=ma_alphas,
                  display=args.display,
                  reward_metric='eulerian')
env.seed(args.seed)

min_xoffset_y = -0.02
max_xoffset_y = 0.1
max_xoffset = 0.1
max_voffset = 0.1
max_focal_dist = 0.20
min_usher_kernel_radius = 0.01
max_usher_kernel_radius = 0.12
max_strength = 25
switch_min = -1
switch_max = 1

agent = TD3(actor_lr=3e-4,
            critic_lr=3e-4,
            critic_weight_decay=0,
            state_dim=get_state_dim(),
            action_dim=get_action_dim(),
            expl_noise_func=GaussianNoise(),
            gamma=0.95,
            min_action=np.array([
                -max_xoffset, min_xoffset_y, -max_xoffset, -max_voffset,
                -max_voffset, -max_voffset, min_usher_kernel_radius, 0,
                switch_min
            ]),
            max_action=np.array([
                +max_xoffset, max_xoffset_y, +max_xoffset, +max_voffset,
                +max_voffset, +max_voffset, max_usher_kernel_radius,
                max_strength, switch_max
            ]),
            learn_after=10000,
            replay_size=7200000,
            hidden_sizes=[400, 300],
            actor_final_scale=0.05 / np.sqrt(1000),
            critic_final_scale=1,
            soft_update_rate=0.005,
            policy_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            batch_size=100)
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
config.learn_after = agent.learn_after
config.batch_size = agent.batch_size
config.seed = args.seed
config.ma_alphas = env.ma_alphas
config.reward_metric = env.reward_metric
config.policy_update_freq = agent.policy_update_freq
config.policy_noise = agent.policy_noise
config.noise_clip = agent.noise_clip
config.local_reward_ratio = local_reward_ratio

wandb.watch(agent.critic[0])

score_history = deque(maxlen=100)
episode_id = 0
episode_t = 0
episode_reward = 0
episode_info = {}
# within_t_accumulator = {}
# for metric in env.metrics:
#     within_t_accumulator[metric] = {'error': 0, 'baseline': 0}
state = env.reset()
done = False

for t in range(max_timesteps):
    episode_t += 1
    if t < agent.learn_after:
        action = np.zeros((env.num_buoys, agent.action_dim))
        for buoy_id in range(env.num_buoys):
            action[buoy_id] = agent.uniform_random_action()
    else:
        action = agent.get_action(state)
    new_state, reward, local_rewards, done, step_info = env.step(
        agent.actor.from_normalized_action(action), compute_local_rewards=True)
    done_int = 0

    for buoy_id in range(env.num_buoys):
        agent.remember(env.dir_id % 5, state[buoy_id], action[buoy_id],
                       reward + local_rewards[buoy_id] * local_reward_ratio,
                       new_state[buoy_id], done_int)
    episode_reward += reward
    state = new_state
    for key in step_info:
        if key not in episode_info:
            episode_info[key] = 0
        episode_info[key] += step_info[key]

    if t >= agent.learn_after:
        agent.learn(env.dir_id % 5)

    if done:
        score_history.append(episode_reward)
        log_object = {'score': episode_reward}
        if len(score_history) == score_history.maxlen:
            log_object['score100'] = np.mean(list(score_history))
        # for metric in env.metrics:
        #     episode_info[f'{metric}-m%'] = within_t_accumulator[
        #             metric]['error'] / within_t_accumulator[metric]['baseline']
        #     within_t_accumulator[metric]['error']= 0
        #     within_t_accumulator[metric]['baseline'] = 0

        for key in episode_info:
            if (not key.endswith("_baseline")) and (
                    not key.endswith("_num_samples")):
                log_object[key] = episode_info[key]

        episode_id += 1
        episode_t = 0
        episode_reward = 0
        episode_info = {}
        state = env.reset()
        done = False

        if episode_id % 50 == 0:
            print('log_object', log_object)
            save_dir = f"artifacts/{wandb.run.id}/models/{episode_id}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            agent.save_models(save_dir)

        wandb.log(log_object)

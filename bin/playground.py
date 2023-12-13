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
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-46ac6fb3-1105.00.11.03",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-b82c6c5c-1105.01.38.31",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-44711c9c-1105.02.23.58",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-55f7364a-1105.03.02.33",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-d7715a69-1105.03.37.41",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-43c1e488-1105.04.25.45",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-c189ffa1-1105.05.01.15",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-95dd78bd-1105.05.44.33",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-838dedf7-1105.06.23.24",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-1d69bc18-1105.07.12.34",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-51135384-1105.07.53.40",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-043fc517-1105.08.42.42",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-41b01e73-1105.09.33.00",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-440435b4-1105.11.27.28",
    f"{args.truth_dir}/parametric-star-epicycloid/rltruth-9a48ebdd-1105.12.13.30",
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
            replay_size=36000000,
            hidden_sizes=[2048, 2048, 1024],
            actor_final_scale=0.05 / np.sqrt(1000),
            critic_final_scale=1,
            soft_update_rate=0.005,
            policy_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
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
        agent.actor.from_normalized_action(action),
        compute_local_rewards=False)
    done_int = 0

    for buoy_id in range(env.num_buoys):
        agent.remember(0, state[buoy_id], action[buoy_id], reward,
                       new_state[buoy_id], done_int)
    episode_reward += reward
    state = new_state
    for key in step_info:
        if key not in episode_info:
            episode_info[key] = 0
        episode_info[key] += step_info[key]

    if t >= agent.learn_after:
        agent.learn(0)

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

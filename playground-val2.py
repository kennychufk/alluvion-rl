import sys
import argparse

import alluvion as al
import numpy as np
from pathlib import Path
import time

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise
from util import get_state_dim, get_action_dim, eval_agent

parser = argparse.ArgumentParser(description='RL playground')
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', metavar='d', type=bool, default=False)
parser.add_argument('--model-dir', type=str, required=True)
args = parser.parse_args()

dp = al.Depot(np.float32)

max_xoffset = 0.05
max_voffset = 0.04
max_focal_dist = 0.20
min_usher_kernel_radius = 0.01
max_usher_kernel_radius = 0.08
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

val_dirs = [
    f"{args.truth_dir}/rltruth-be268318-0526.07.32.30/",
    f"{args.truth_dir}/rltruth-5caefe43-0526.14.46.12/",
    f"{args.truth_dir}/rltruth-e8edf09d-0526.18.34.19/",
    f"{args.truth_dir}/rltruth-6de1d91b-0526.09.31.47/",
    f"{args.truth_dir}/rltruth-3b860b54-0526.23.12.15/",
    f"{args.truth_dir}/rltruth-eb3494c1-0527.00.32.34/",
    f"{args.truth_dir}/rltruth-e9ba71d8-0527.01.52.31/"
]
agent.load_models(args.model_dir)
eval_agent(dp, agent, val_dirs, args.cache_dir, report_state_action=True)

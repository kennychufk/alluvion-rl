import argparse

import alluvion as al
import numpy as np
import wandb

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise
from util import Environment, EnvironmentPIV, eval_agent

parser = argparse.ArgumentParser(description='RL evaluation')
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--display', type=bool, default=False)
parser.add_argument('--run-id', type=str, required=True)
parser.add_argument('--model-iteration', type=int, default=-1)
args = parser.parse_args()

dp = al.Depot(np.float32)

api = wandb.Api()
run = api.run(f'kennychufk/alluvion-rl/{args.run_id}')
config = run.config

agent = TD3(actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            critic_weight_decay=config['critic_weight_decay'],
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            expl_noise_func=GaussianNoise(),
            gamma=config['gamma'],
            min_action=np.array(config['min_action']),
            max_action=np.array(config['max_action']),
            learn_after=config['learn_after'],
            replay_size=config['replay_size'],
            hidden_sizes=config['hidden_sizes'],
            actor_final_scale=config['actor_final_scale'],
            critic_final_scale=config['critic_final_scale'],
            soft_update_rate=config['soft_update_rate'],
            batch_size=config['batch_size'])

val_dirs = [
    f"{args.truth_dir}/rltruth-be268318-0526.07.32.30/",
    f"{args.truth_dir}/rltruth-5caefe43-0526.14.46.12/",
    f"{args.truth_dir}/rltruth-e8edf09d-0526.18.34.19/",
    f"{args.truth_dir}/rltruth-6de1d91b-0526.09.31.47/",
    f"{args.truth_dir}/rltruth-3b860b54-0526.23.12.15/",
    f"{args.truth_dir}/rltruth-eb3494c1-0527.00.32.34/",
    f"{args.truth_dir}/rltruth-e9ba71d8-0527.01.52.31/"
]

# eval_env = Environment(dp,
#                        truth_dirs=val_dirs,
#                        cache_dir=args.cache_dir,
#                        ma_alphas=config['ma_alphas'],
#                        display=args.display)
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
                         display=args.display)


if args.model_iteration < 0:
    history = run.scan_history(keys=None,
                               page_size=1000,
                               min_step=None,
                               max_step=None)
    wandb.init(project='alluvion-rl', id=f'{args.run_id}AugSim')
    for key in config:
        wandb.config[key] = config[key]
    for row_id, row in enumerate(history):
        log_object = {}
        episode_id = row_id + 1
        for key in row:
            if (row_id != row['_step']):
                print('step id mismatch')
            if (not key.startswith('gradients/') and not key.startswith('_')):
                log_object[key] = row[key]
        if episode_id % 50 == 0:
            agent.load_models(f'artifacts/{args.run_id}/models/{episode_id}/')
            result_dict = {}
            log_object['val-again'] = eval_agent(eval_env, agent, result_dict)
            # log_object['val-piv'] = eval_agent(env_piv, agent, result_dict)
            for result_key in result_dict:
                log_object[result_key] = result_dict[result_key]
        wandb.log(log_object)
else:
    agent.load_models(f'artifacts/{args.run_id}/models/{args.model_iteration}/')
    result_dict = {}
    eval_agent(env_piv, agent, result_dict)

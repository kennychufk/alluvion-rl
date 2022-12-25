import argparse
import pickle

import alluvion as al
import numpy as np
import wandb

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise
from util import Environment, EnvironmentPIV, eval_agent, get_timestamp_and_hash

parser = argparse.ArgumentParser(description='RL evaluation')
parser.add_argument('--cache-dir', type=str, default='.')
parser.add_argument('--truth-dir', type=str, required=True)
parser.add_argument('--shape-dir', type=str, default=None)
parser.add_argument('--display', type=bool, default=False)
parser.add_argument('--run-id', type=str, required=True)
parser.add_argument('--model-iteration', type=int, default=-1)
parser.add_argument('--quick-mode', type=int, default=1)
parser.add_argument('--evaluation-metrics',
                    nargs='+',
                    default=['eulerian_masked'])
parser.add_argument('--piv', type=int, default=0)
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
            learn_after=1000,
            replay_size=1000,
            hidden_sizes=config['hidden_sizes'],
            actor_final_scale=config['actor_final_scale'],
            critic_final_scale=config['critic_final_scale'],
            soft_update_rate=config['soft_update_rate'],
            batch_size=config['batch_size'])

piv_mode = (args.piv == 1)
eval_learning_curve = (args.model_iteration < 0)
quick_mode = (args.quick_mode == 1)
save_visual = not eval_learning_curve
report_state_action = not eval_learning_curve
# save_visual = True
# report_state_action = True

if piv_mode:
    val_dirs = [
        '/media/kennychufk/vol1bk0/20210415_162749-laser-too-high/',  # 4
        '/media/kennychufk/vol1bk0/20210415_164304/',  # 4
        '/media/kennychufk/vol1bk0/20210416_101435/',  # 5
        '/media/kennychufk/vol1bk0/20210416_102548/',  # 6
        '/media/kennychufk/vol1bk0/20210416_103739/',  # 7
        '/media/kennychufk/vol1bk0/20210416_104936/',  # 8
        '/media/kennychufk/vol1bk0/20210416_120534/',  # 9
        '/media/kennychufk/vol1bk0/20210416_114327',  # 8
        '/media/kennychufk/vol1bk0/20210416_115523'  # 9
    ]
    env = EnvironmentPIV(dp,
                         truth_dirs=val_dirs,
                         cache_dir=args.cache_dir,
                         ma_alphas=config['ma_alphas'],
                         display=args.display,
                         volume_method=al.VolumeMethod.pellets,
                         save_visual=save_visual)
else:
    val_dirs = [
        f"{args.truth_dir}/parametric-nephroid/rltruth-099b5858-1102.06.10.45",
        f"{args.truth_dir}/parametric-nephroid/rltruth-34970c41-1102.01.21.46",
        f"{args.truth_dir}/parametric-nephroid/rltruth-4e1f0fe8-1102.03.41.23",
        f"{args.truth_dir}/parametric-nephroid/rltruth-6099faf5-1102.04.02.53",
        f"{args.truth_dir}/parametric-nephroid/rltruth-639de5d5-1102.06.37.34",
        f"{args.truth_dir}/parametric-nephroid/rltruth-68eb3a7b-1102.03.22.04",
        f"{args.truth_dir}/parametric-nephroid/rltruth-7633a710-1102.02.42.52",
        f"{args.truth_dir}/parametric-nephroid/rltruth-764fe695-1102.04.52.44",
        f"{args.truth_dir}/parametric-nephroid/rltruth-77a3b794-1102.03.02.55",
        f"{args.truth_dir}/parametric-nephroid/rltruth-7b60a53f-1102.05.16.22",
        f"{args.truth_dir}/parametric-nephroid/rltruth-7fd0ea22-1102.02.22.12",
        f"{args.truth_dir}/parametric-nephroid/rltruth-ab55bae7-1102.05.41.34",
        f"{args.truth_dir}/parametric-nephroid/rltruth-c85e79a2-1102.04.23.40",
        f"{args.truth_dir}/parametric-nephroid/rltruth-f0fef677-1102.02.03.35",
        f"{args.truth_dir}/parametric-nephroid/rltruth-fe0a163a-1102.01.44.11",
    ]

    env = Environment(dp,
                      truth_dirs=val_dirs,
                      cache_dir=args.cache_dir,
                      ma_alphas=config['ma_alphas'],
                      display=args.display,
                      save_visual=save_visual,
                      evaluation_metrics=args.evaluation_metrics,
                      shape_dir=args.shape_dir,
                      quick_mode=quick_mode)

if eval_learning_curve:
    history = run.scan_history(keys=None,
                               page_size=1000,
                               min_step=None,
                               max_step=None)
    wandb.init(project='alluvion-rl', id=f'{args.run_id}Aug', tags=['eval'])
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
        # NOTE: to resume evaluteion. The following number should be the last step number seen in wandb console for 'score' + 2 (should end with 99)
        if episode_id < 0:
            wandb.log(log_object)
            continue
        if episode_id % 50 == 0:
            agent.load_models(f'artifacts/{args.run_id}/models/{episode_id}/')
            episode_info = {}
            eval_agent(env,
                       agent,
                       episode_info,
                       report_state_action=report_state_action,
                       run_id=args.run_id,
                       model_iteration=episode_id)
            for result_key in episode_info:
                log_object[result_key] = episode_info[result_key]
        wandb.log(log_object)
else:
    agent.load_models(
        f'artifacts/{args.run_id}/models/{args.model_iteration}/')
    episode_info = {}
    eval_agent(env,
               agent,
               episode_info,
               report_state_action=report_state_action,
               run_id=args.run_id,
               model_iteration=args.model_iteration)
    print(episode_info)
    timestamp_str, timestamp_hash = get_timestamp_and_hash()
    with open(f'{args.run_id}-{args.model_iteration}-{timestamp_str}.pickle',
              'wb') as f:
        pickle.dump(episode_info, f, pickle.HIGHEST_PROTOCOL)

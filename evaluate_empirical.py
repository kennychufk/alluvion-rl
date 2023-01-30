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
parser.add_argument('--unmasked-buoy-ids', nargs='+', type=int)
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
    # val_dirs = [
    #     '/media/kennychufk/vol1bk0/20210416_154851', # water
    #     # '/media/kennychufk/vol1bk0/20210416_155948', # water
    #     # '/media/kennychufk/vol1bk0/20210416_161703', # water
    #     # '/media/kennychufk/vol1bk0/20210416_162740', # water
    #     # '/media/kennychufk/vol1bk0/20210416_164544-slightly-wrong-focus', # gwm
    #     # '/media/kennychufk/vol1bk0/20210416_165720',  # gwm
    # ]
    env = EnvironmentPIV(dp,
                         truth_dirs=val_dirs,
                         cache_dir=args.cache_dir,
                         ma_alphas=config['ma_alphas'],
                         display=args.display,
                         unmasked_buoy_ids=args.unmasked_buoy_ids,
                         volume_method=al.VolumeMethod.pellets,
                         save_visual=save_visual)
else:
    val_dirs = [
        # f"{args.truth_dir}/diagonal-val/rltruth-484d9c63-1215.15.19.36",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-2e6e574f-1208.22.30.56",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-eb78fbb5-1209.20.00.05",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-8c97aceb-1208.13.16.15",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-8c133a54-1214.05.23.03",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-8fa33fc3-1209.12.31.42",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-c9bdd947-1214.12.19.24",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-1625e73c-1214.01.18.49",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-72331b50-1215.01.25.48",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-b32afdb1-1214.21.15.44",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-4d86d321-1208.06.18.31",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-8001db83-1208.14.51.33",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-b1377fd6-1207.16.03.22",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-f8be77e5-1212.16.43.11",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-0657cf24-1209.05.34.19",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-de581515-1215.05.29.34",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-b9c59de4-1207.22.59.15",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-97286492-1212.20.50.48",  # 4
        # f"{args.truth_dir}/diagonal-train4/rltruth-dbdf3f05-0914.01.20.58",  # 4
        # f"{args.truth_dir}/diagonal-val/rltruth-4df9af8c-1215.05.59.15",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-012be00a-1214.05.51.19",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-18f8baf5-1214.12.48.36",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-9662051c-1208.06.47.45",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-3380ef7a-1207.23.28.20",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-0b9775ce-1215.01.55.21",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-c011fe29-1208.23.00.11",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-2c4fad8e-1208.15.24.23",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-9f026940-1209.20.29.58",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-78366837-1208.14.00.25",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-e277a2c4-1214.21.45.02",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-2ed76551-1214.01.47.42",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-a8383074-1212.17.11.53",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-6bfc3765-1209.13.00.40",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-7d6068cd-1212.21.20.03",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-01e920c5-1215.15.50.46",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-b9a38b8b-1209.06.03.11",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-bca04a6f-1207.16.33.45",  # 6
        # f"{args.truth_dir}/diagonal-train4/rltruth-8cecae4f-0914.01.50.28",  # 6
        # f"{args.truth_dir}/diagonal-val/rltruth-5196bac5-1209.06.32.18",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-c854bd22-1208.23.29.11",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-30937e46-1214.22.14.39",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-6ff7c819-1214.18.04.50",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-45b0bb8a-1208.15.54.25",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-6dca60b2-1207.17.03.03",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-cecbf17a-1215.02.25.04",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-0d32575b-1215.16.20.22",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-e8482f31-1215.06.29.00",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-c39e38e7-1207.23.58.15",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-b5990810-1214.02.16.56",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-20cfd180-1212.21.49.30",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-21c71f42-1209.13.30.01",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-4a59f234-1214.06.20.04",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-ff0c3814-1212.17.40.55",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-663cc3b8-1210.06.25.18",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-0a9306a4-1208.07.17.02",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-b341d9cd-1209.20.59.00",  # 8
        # f"{args.truth_dir}/diagonal-train4/rltruth-d949a937-0914.02.21.44",  # 8
        # f"{args.truth_dir}/diagonal-val/rltruth-f251bcf3-1214.06.49.00",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-30fb6bd3-1215.16.52.04",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-9c0b0471-1208.00.27.40",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-0f311d3b-1214.18.35.10",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-00705caa-1214.02.48.20",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-43705e93-1208.07.46.22",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-5e6b433a-1214.22.44.28",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-c8909352-1208.16.26.22",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-72e03685-1215.02.55.00",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-dd6a1c0b-1213.16.15.12",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-17d76032-1207.17.31.48",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-380bb053-1209.07.01.35",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-117e328d-1212.18.10.00",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-894e2d9d-1210.06.58.26",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-5a7c118a-1209.21.29.49",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-90ffe775-1215.06.59.00",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-ec63c7b6-1209.14.00.42",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-6f162528-1208.23.58.45",  # 10
        # f"{args.truth_dir}/diagonal-train4/rltruth-9d4aaf45-0914.02.52.46",  # 10
        # f"{args.truth_dir}/diagonal-val/rltruth-91516f50-1209.14.30.21",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-9b9e0402-1209.00.29.02",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-b991d064-1215.17.24.47",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-59fe7a79-1214.03.17.50",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-c48c41b7-1209.21.59.32",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-d9c5170d-1215.03.24.52",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-92f28a8c-1210.07.30.02",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-18f4f6c8-1214.07.18.16",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-dee6047f-1208.17.01.10",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-b7c03b11-1207.18.00.47",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-0b89482b-1209.07.31.35",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-0e493572-1208.00.57.18",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-e9050519-1215.07.28.50",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-f0b0d688-1212.18.40.45",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-0db8df83-1213.16.45.37",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-c193af10-1214.23.14.58",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-af142ad3-1214.19.05.01",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-ff70f510-1208.08.15.55",  # 12
        # f"{args.truth_dir}/diagonal-train4/rltruth-448949d6-0914.03.23.47",  # 12
        # f"{args.truth_dir}/diagonal-val/rltruth-39296608-1214.23.44.52",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-6a6da182-1212.19.11.42",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-b5dbbaee-1209.22.31.27",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-f3749aa4-1214.19.34.41",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-55148607-1208.17.33.39",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-4ee7f062-1215.17.56.42",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-d07bdeb0-1208.01.26.57",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-8ca36713-1208.08.51.01",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-540b3c5b-1214.07.47.53",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-4b9d632d-1210.08.01.58",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-3ec112d0-1214.03.47.28",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-94051663-1209.08.01.19",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-31714d94-1215.07.58.37",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-8b0c490c-1213.17.18.48",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-e4ba3f8e-1207.18.29.48",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-a7600040-1215.03.54.50",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-9549a44d-1209.00.58.25",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-eed01995-1209.14.59.55",  # 16
        # f"{args.truth_dir}/diagonal-train4/rltruth-d4537532-0914.04.30.48",  # 16
        # f"{args.truth_dir}/diagonal-val/rltruth-2f606db2-1212.19.42.08",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-ec27fdba-1208.02.02.21",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-3b32a554-1214.08.17.13",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-0284f323-1207.18.59.09",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-c3515310-1213.18.42.02",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-0e1adb64-1209.15.30.38",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-22615d8c-1208.18.04.00",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-cff8c68b-1209.08.31.02",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-eea190ad-1209.01.28.08",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-90a40772-1208.09.21.31",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-8d8b5645-1214.04.17.59",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-4b808c29-1210.08.34.34",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-c2463aef-1215.18.27.33",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-2268d54d-1215.04.25.35",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-ae6cc826-1215.00.15.06",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-92008135-1209.23.00.59",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-32221b15-1215.08.29.14",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-a56384fe-1214.20.06.13",  # 22
        # f"{args.truth_dir}/diagonal-train4/rltruth-1dc784ae-0914.06.08.59",  # 22
        # f"{args.truth_dir}/diagonal-val/rltruth-76004b81-1214.20.43.30",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-a5f01b7d-1210.09.07.47",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-c0afd48b-1215.04.57.01",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-efc9f566-1215.00.52.21",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-706920c1-1215.19.05.18",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-fcfdf114-1209.09.01.54",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-ec96af99-1209.23.36.50",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-7fb777b3-1208.09.51.49",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-818de0a0-1209.01.58.03",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-ae40159d-1208.02.34.06",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-3cc52e5b-1215.09.00.22",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-f6d859fd-1209.16.01.13",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-cd042f17-1208.18.36.59",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-4109c4a0-1207.19.29.46",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-195b34da-1214.08.50.55",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-e6dc1abd-1214.04.48.21",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-002c0d94-1212.20.17.01",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-01d4c881-1213.19.12.48",  # 30
        # f"{args.truth_dir}/diagonal-train4/rltruth-ebc119cb-0914.08.22.41",  # 30
        # f"{args.truth_dir}/diagonal-val/rltruth-0b067cdc-1209.02.32.15",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-a1512685-1207.20.03.35",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-2b47f9bf-1210.09.40.46",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-d365c25d-1208.19.08.01",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-c33dd86a-1209.16.32.10",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-0b87b7e1-1208.03.06.30",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-30c7e129-1209.09.33.02",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-d9d2f33e-1210.00.11.28",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-6db0f17c-1208.10.24.19",  # 40
        # f"{args.truth_dir}/diagonal-train4/rltruth-312c0cc8-0914.11.53.13",  # 40
        # f"{args.truth_dir}/diagonal-val/rltruth-644f3d1f-1210.00.42.59",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-58645489-1208.10.56.57",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-026f3114-1210.10.12.40",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-faa0d87c-1209.10.05.51",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-5aa646b3-1209.17.14.35",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-c35d9a59-1208.19.43.47",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-506f86a3-1207.20.35.09",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-05c0f61d-1209.03.04.49",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-d0753e27-1208.03.41.46",  # 52
        # f"{args.truth_dir}/diagonal-train4/rltruth-cf367ec6-0914.15.16.06",  # 52
        # f"{args.truth_dir}/diagonal-val/rltruth-4524c271-1209.17.53.56",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-d20cbc0c-1209.10.40.40",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-0baa72e7-1208.20.22.25",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-dcf30ebc-1207.21.08.10",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-ce8c43ef-1210.10.51.19",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-4a1c3678-1208.11.37.39",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-98dd5f34-1208.04.13.58",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-5f2b59d1-1209.03.41.23",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-d07e7cd9-1210.01.18.08",  # 66
        # f"{args.truth_dir}/diagonal-train4/rltruth-8834d9d4-0914.19.38.36",  # 66
        # f"{args.truth_dir}/diagonal-val/rltruth-1ab47e43-1209.11.16.18",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-abe0e8a5-1210.11.26.28",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-e2b51761-1210.01.53.28",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-d9b9567c-1208.12.15.52",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-701a39d6-1209.04.15.58",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-21f44f1a-1208.04.53.31",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-97c1805f-1209.18.33.00",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-2c4055b4-1208.20.57.17",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-c81f6974-1207.21.43.16",  # 82
        # f"{args.truth_dir}/diagonal-train4/rltruth-65288f7b-0915.00.38.11",  # 82
        # f"{args.truth_dir}/diagonal-val/rltruth-a59b0830-1210.02.35.29",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-8acba45f-1207.22.21.31",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-549c163d-1208.21.36.39",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-69a7cdae-1210.12.04.35",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-6cb0e35e-1208.12.53.47",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-55550198-1209.19.18.24",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-f1df620b-1209.11.55.03",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-c5382717-1208.05.35.38",  # 100
        # f"{args.truth_dir}/diagonal-val/rltruth-886e0dba-1209.04.53.49",  # 100
        # f"{args.truth_dir}/diagonal-train4/rltruth-14f08550-0915.06.44.03",  # 100

        # f"{args.truth_dir}/parametric-lissajous/rltruth-e093dc5d-1219.20.32.01",
        # f"{args.truth_dir}/teaser-diagonal/rltruth-e9aaf4ff-0101.03.03.36",
        # f"{args.truth_dir}/teaser-star/rltruth-5ee15e5f-0104.23.59.19",

        # f"{args.truth_dir}/custom-randomized2/rltruth-442e8c49-0919.18.24.08",
        # f"{args.truth_dir}/custom-randomized2/rltruth-cbcb2b90-0921.22.04.29",
        # f"{args.truth_dir}/custom-randomized2/rltruth-60418e3b-0918.15.21.40",
        # f"{args.truth_dir}/custom-randomized2/rltruth-6db45e5f-0918.22.34.28",
        # f"{args.truth_dir}/custom-randomized2/rltruth-6a6532b5-0921.06.51.31",

        # f"{args.truth_dir}/val-loop2/rltruth-c6da9fc1-1221.21.47.49", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-9980b528-1222.09.12.29", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-58ac61a3-1222.06.21.49", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-4cff3e53-1222.00.43.10", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-deab9bd2-1222.11.55.44", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-63d44016-1222.14.39.27", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-c2ea582c-1222.17.24.58", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-778d8ed8-1222.03.31.54", # 4
        # f"{args.truth_dir}/val-loop2/rltruth-b030ece4-1222.15.08.05", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-e31075c4-1222.01.11.58", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-311f0cd8-1221.22.17.25", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-a134cc0f-1222.06.50.19", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-e82e6d76-1222.04.00.19",  # 8
        # f"{args.truth_dir}/val-loop2/rltruth-2447ba6b-1222.17.53.38", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-353c7ac8-1222.09.42.18", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-f92a7f02-1222.12.24.11", # 8
        # f"{args.truth_dir}/val-loop2/rltruth-a64130f9-1222.07.19.04", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-670324de-1222.04.29.01", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-460b91b6-1222.10.11.14", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-7df556d0-1221.22.46.56", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-56193181-1222.18.22.36", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-b136f9b6-1222.12.53.14", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-1e11929c-1222.01.41.02", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-073526d1-1222.15.37.04", # 22
        # f"{args.truth_dir}/val-loop2/rltruth-9b6b6e06-1222.04.59.55", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-d4510fe6-1222.16.07.08", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-e1eaad29-1222.13.23.50",  # 66
        # f"{args.truth_dir}/val-loop2/rltruth-5f1b7f96-1221.23.25.26", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-ef21a02c-1222.10.41.00", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-24ad4fbe-1222.18.53.55", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-f7555a9e-1222.07.49.32", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-33385c4f-1222.02.11.37", # 66
        # f"{args.truth_dir}/val-loop2/rltruth-946f5ea4-1222.11.19.50", # 100
        # f"{args.truth_dir}/val-loop2/rltruth-7c3a4154-1222.14.01.09", # 100
        # f"{args.truth_dir}/val-loop2/rltruth-36fb42b6-1222.16.46.27", # 100
        # f"{args.truth_dir}/val-loop2/rltruth-77f9cc3a-1222.08.27.12",  # 100
        # f"{args.truth_dir}/val-loop2/rltruth-5064b7d8-1222.19.28.14", # 100
        # f"{args.truth_dir}/val-loop2/rltruth-d0f5b0ff-1222.00.00.24", # 100
        # f"{args.truth_dir}/val-loop2/rltruth-1e7a01c5-1222.02.50.50", # 100
        # f"{args.truth_dir}/val-loop2/rltruth-c8179586-1222.05.33.31", # 100


        # f"{args.truth_dir}/val-bidir-circles2/rltruth-07683cc1-1223.04.32.34",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-0885a1ea-1223.01.32.13",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-10bb14d1-1223.11.03.52",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-111b1df1-1223.07.38.18",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-13bd57f5-1223.00.28.59",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-1b46cddb-1223.05.05.48",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-230d2af9-1223.07.30.38",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-276c84f7-1223.04.33.23",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-2d2db035-1223.02.46.22",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-37fffc66-1223.10.02.17",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-38d4d7a4-1223.06.23.17",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-47c184a4-1222.21.27.25",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-526a1740-1223.09.23.01",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-531b5fa6-1223.02.48.21",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-56c7ee87-1223.10.40.19",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-5c2bfbb0-1223.06.57.40",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-5cced53a-1223.09.28.06",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-5ce9cc22-1223.08.03.34",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-613cb1de-1223.09.55.57",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-6890716c-1223.05.17.01",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-6c55312c-1223.06.59.03",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-70f90d85-1223.02.04.56",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-7cbf3aa9-1223.04.01.46",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-7eaf83c9-1223.05.43.11",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-85e30c18-1223.10.27.57",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-8f38a8a1-1223.03.19.33",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-978b68f8-1223.11.41.46",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-9d1a99e1-1222.22.30.22",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-bc3a7a28-1223.03.30.35",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-c38d8d6f-1223.08.54.54",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-ca5e7d03-1223.03.54.19",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-d039752b-1222.23.03.26",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-d0f21711-1223.05.50.01",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-d2dffb8c-1223.08.39.31",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-d45047d7-1223.01.00.24",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-ddc9621f-1223.06.27.58",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-df89f596-1223.02.13.38",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-f1e60a1e-1223.08.21.59",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-f448906d-1222.23.44.48",
        # f"{args.truth_dir}/val-bidir-circles2/rltruth-f7e239fc-1222.21.58.46",

        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-963b7147-1015.09.48.23",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-61b17546-1015.04.09.38",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-c0a38d05-1015.03.20.54",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-cec5603e-1014.23.54.16",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-370823da-1015.02.02.27",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-f39e190c-1015.06.35.08",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-94ac4e05-1014.21.07.35",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-75124ebf-1015.08.08.44",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-e0a74210-1015.02.57.18",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-e81cdc75-1015.08.30.40",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-61380666-1015.05.21.56",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-ebb22bdd-1015.03.42.01",
        # f"{args.truth_dir}/parametric-interesting-loop/rltruth-9c477345-1015.09.16.03",

        # f"{args.truth_dir}/parametric-nephroid/rltruth-34970c41-1102.01.21.46", # 4
        # f"{args.truth_dir}/parametric-nephroid/rltruth-fe0a163a-1102.01.44.11", # 5
        # f"{args.truth_dir}/parametric-nephroid/rltruth-f0fef677-1102.02.03.35", # 6
        # f"{args.truth_dir}/parametric-nephroid/rltruth-7fd0ea22-1102.02.22.12", # 7
        # f"{args.truth_dir}/parametric-nephroid/rltruth-7633a710-1102.02.42.52", # 8
        # f"{args.truth_dir}/parametric-nephroid/rltruth-77a3b794-1102.03.02.55", # 10
        # f"{args.truth_dir}/parametric-nephroid/rltruth-68eb3a7b-1102.03.22.04", # 12
        # f"{args.truth_dir}/parametric-nephroid/rltruth-4e1f0fe8-1102.03.41.23", # 16
        # f"{args.truth_dir}/parametric-nephroid/rltruth-6099faf5-1102.04.02.53", # 22
        # f"{args.truth_dir}/parametric-nephroid/rltruth-c85e79a2-1102.04.23.40", # 30
        # f"{args.truth_dir}/parametric-nephroid/rltruth-764fe695-1102.04.52.44", # 40
        # f"{args.truth_dir}/parametric-nephroid/rltruth-7b60a53f-1102.05.16.22", # 52
        # f"{args.truth_dir}/parametric-nephroid/rltruth-ab55bae7-1102.05.41.34", # 66
        # f"{args.truth_dir}/parametric-nephroid/rltruth-099b5858-1102.06.10.45", # 82
        # f"{args.truth_dir}/parametric-nephroid/rltruth-639de5d5-1102.06.37.34", # 100
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

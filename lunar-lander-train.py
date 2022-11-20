import argparse
import random
from collections import deque

import numpy as np
import torch
import gym
import wandb

from rl import TD3, OrnsteinUhlenbeckProcess, GaussianNoise

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2021)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env = gym.make('LunarLander-v2', continuous=True, enable_wind=False)
env.action_space.seed(args.seed)

agent = TD3(actor_lr=1e-4,
            critic_lr=1e-3,
            critic_weight_decay=0,
            state_dim=8,
            action_dim=2,
            min_action=env.action_space.low,
            max_action=env.action_space.high,
            replay_size=1000000,
            expl_noise_func=GaussianNoise(std_dev=0.1),
            hidden_sizes=[400, 300],
            soft_update_rate=0.001,
            batch_size=64,
            actor_final_scale=0.051961524,
            critic_final_scale=0.005196152)
max_timesteps = 1000000

wandb.init(project='rl-continuous')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.critic_weight_decay = agent.critic_weight_decay
config.state_dim = agent.state_dim
config.action_dim = agent.action_dim
config.hidden_sizes = agent.hidden_sizes
config.min_action = agent.target_actor.min_action
config.max_action = agent.target_actor.max_action
config.soft_update_rate = agent.soft_update_rate
config.gamma = agent.gamma
config.replay_size = agent.replay_size
config.actor_final_scale = agent.actor_final_scale
config.critic_final_scale = agent.critic_final_scale
config.learn_after = agent.learn_after
config.seed = args.seed
config.max_timesteps = max_timesteps

wandb.watch(agent.critic)

score_history = deque(maxlen=100)
episode_id = 0
episode_t = 0
episode_reward = 0
state = env.reset(seed=args.seed)
done = False

for t in range(max_timesteps):
    episode_t += 1
    if t < agent.learn_after:
        action = env.action_space.sample()
    else:
        action = agent.get_action(state)
    new_state, reward, done, info = env.step(
        agent.actor.from_normalized_action(action))
    done_int = int(done) if episode_t < env._max_episode_steps else 0
    agent.remember(state, action, reward, new_state, done_int)
    episode_reward += reward
    state = new_state
    # env.render()
    if t >= agent.learn_after:
        agent.learn(symmetrize=False)

    if done:
        score_history.append(episode_reward)
        log_object = {'score': episode_reward}
        if len(score_history) == score_history.maxlen:
            log_object['score100'] = np.mean(list(score_history))
        wandb.log(log_object)

        episode_id += 1
        episode_t = 0
        episode_reward = 0
        state = env.reset()
        done = False

        if episode_id % 50 == 0:
            agent.save_models(wandb.run.dir)

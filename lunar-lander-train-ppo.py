import argparse
import random
from collections import deque

import numpy as np
import torch
import gym
import wandb

from rl import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2021)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

env = gym.make('LunarLander-v2', continuous=True, enable_wind=False)
env.action_space.seed(args.seed)

agent = PPO(actor_lr=3e-4,
            critic_lr=1e-3,
            state_dim=8,
            action_dim=2,
            min_action=env.action_space.low,
            max_action=env.action_space.high,
            hidden_sizes=[400, 300],
            num_steps_per_epoch=4000)
num_epochs = 10000

wandb.init(project='rl-continuous')
config = wandb.config
config.actor_lr = agent.actor_lr
config.critic_lr = agent.critic_lr
config.state_dim = agent.state_dim
config.action_dim = agent.action_dim
config.hidden_sizes = agent.hidden_sizes
config.min_action = agent.actor.min_action
config.max_action = agent.actor.max_action
config.gamma = agent.gamma
config.lam = agent.lam
config.seed = args.seed
config.num_epochs = num_epochs

wandb.watch(agent.critic)

score_history = deque(maxlen=100)
episode_id = 0
episode_t = 0
episode_reward = 0
state = env.reset(seed=args.seed)
done = False

for epoch_id in range(num_epochs):
    for t in range(agent.num_steps_per_epoch):
        action, value, logp = agent.step(state)

        new_state, reward, done, info = env.step(
            agent.actor.from_normalized_action(action))
        episode_reward += reward
        episode_t += 1

        agent.memory.store(state, action, reward, value, logp)
        state = new_state

        timeout = episode_t == env._max_episode_steps
        terminal = done or timeout
        epoch_ended = t == (agent.num_steps_per_epoch - 1)

        if terminal or epoch_ended:
            if epoch_ended and not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' %
                      episode_t,
                      flush=True)
            if timeout or epoch_ended:
                _, value, _ = agent.step(state)
            else:
                value = 0
            agent.memory.finish_path(value, agent.gamma, agent.lam)
            if terminal:
                score_history.append(episode_reward)
                log_object = {'score': episode_reward}
                if len(score_history) == score_history.maxlen:
                    log_object['score100'] = np.mean(list(score_history))
                wandb.log(log_object)
            state = env.reset()
            episode_id += 1
            episode_reward = 0
            episode_t = 0

    # update
    agent.update()
    if epoch_id % 50 == 0:
        agent.save_models(wandb.run.dir)

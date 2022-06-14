import numpy as np
from pathlib import Path

from .environment import Environment
from .timestamp import get_timestamp_and_hash
from .policy_codec import get_action_dim, get_state_dim


def eval_agent(eval_env, agent, result_dict, report_state_action=False):
    timestamp_str, timestamp_hash = get_timestamp_and_hash()

    all_episodes_error_accm = 0.
    all_episodes_mask_accm = 0
    episode_reward = 0
    episode_mask_accm = 0
    episode_error_accm = 0
    episode_truth_accm = 0
    for _ in range(len(eval_env.truth_dirs)):
        state, done = eval_env.reset(), False
        if (report_state_action):
            state_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys,
                 get_state_dim()))
            action_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys,
                 get_action_dim()))
            cursor = 0
        dirname_short = Path(eval_env.truth_dir).name[4:15]
        while not done:
            action = agent.get_action(state, enable_noise=False)
            real_action = agent.actor.from_normalized_action(action)
            if (report_state_action):
                state_history[cursor] = state
                action_history[cursor] = real_action
                cursor += 1
            state, reward, done, info = eval_env.step(real_action)
            all_episodes_error_accm += info['v_error']
            episode_reward += info['v_error']
            episode_error_accm += info['v_error']
            episode_truth_accm += info['truth_sqr']
            episode_mask_accm += info['num_masked']
            all_episodes_mask_accm += info['num_masked']
        print(episode_reward)
        result_dict[dirname_short] = episode_reward / episode_mask_accm
        result_dict[
            f'{dirname_short}%'] = episode_error_accm / episode_truth_accm
        episode_reward = 0
        episode_mask_accm = 0
        episode_error_accm = 0
        episode_truth_accm = 0
        if (report_state_action):
            report_save_dir = Path(f'val-{dirname_short}-{timestamp_hash}')
            report_save_dir.mkdir(parents=True)
            np.save(f'{str(report_save_dir)}/state.npy', state_history)
            np.save(f'{str(report_save_dir)}/action.npy', action_history)

    return all_episodes_error_accm / all_episodes_mask_accm

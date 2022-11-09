import numpy as np
from pathlib import Path

from .environment import Environment
from .timestamp import get_timestamp_and_hash
from .policy_codec import get_action_dim, get_state_dim


def eval_agent(eval_env,
               agent,
               episode_info,
               report_state_action=False,
               run_id=None,
               model_iteration=None):
    timestamp_str, timestamp_hash = get_timestamp_and_hash()

    all_episodes_error_accm = 0.
    all_episodes_mask_accm = 0
    episode_error_accm = 0
    episode_truth_accm = 0
    volumetric_error_accm = 0
    volumetric_baseline_accm = 0
    within_t_mean_error = 0
    within_t_mean_truth = 0
    for _ in range(len(eval_env.truth_dirs)):
        state, done = eval_env.reset(), False
        if (report_state_action):
            state_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys,
                 get_state_dim()))
            action_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys,
                 get_action_dim()))
            sim_v_history = np.zeros(
                (eval_env._max_episode_steps,
                 eval_env.simulation_sampling.num_samples, 3))
            cursor = 0
        dirname_short = Path(eval_env.truth_dir).name[4:15]
        if report_state_action or eval_env.save_visual:
            report_save_dir = Path(
                f'val-{dirname_short}-{run_id}-{model_iteration}-{timestamp_hash}'
            )
            report_save_dir.mkdir(parents=True)
            if eval_env.save_visual:
                eval_env.save_dir_visual = report_save_dir
        while not done:
            action = agent.get_action_symmetrized(state)
            real_action = agent.actor.from_normalized_action(action)
            if (report_state_action):
                state_history[cursor] = state
                action_history[cursor] = real_action
                sim_v_history[
                    cursor] = eval_env.simulation_sampling.sample_data3.get()
                cursor += 1
            state, reward, done, step_info = eval_env.step(real_action)
            all_episodes_error_accm += step_info['v_error']
            episode_error_accm += step_info['v_error']
            episode_truth_accm += step_info['truth_sqr']
            all_episodes_mask_accm += step_info['num_masked']
            if step_info['num_masked'] > 0:
                within_t_mean_error += step_info['v_error'] / step_info[
                    'num_masked']
                within_t_mean_truth += step_info['truth_sqr'] / step_info[
                    'num_masked']
            if 'volumetric_error' in step_info:
                volumetric_error_accm += step_info['volumetric_error']
                volumetric_baseline_accm += step_info['volumetric_baseline']

        episode_info[
            f'{eval_env.num_buoys}-{dirname_short}%'] = episode_error_accm / episode_truth_accm
        episode_info[
            f'{eval_env.num_buoys}-{dirname_short}m%'] = within_t_mean_error / within_t_mean_truth
        if volumetric_baseline_accm > 0:
            episode_info[
                f'{eval_env.num_buoys}-vol%'] = volumetric_error_accm / volumetric_baseline_accm
            episode_info[f'{eval_env.num_buoys}-vol'] = volumetric_error_accm
        episode_error_accm = 0
        episode_truth_accm = 0
        volumetric_error_accm = 0
        volumetric_baseline_accm = 0
        if (report_state_action):
            np.save(f'{str(report_save_dir)}/state.npy', state_history)
            np.save(f'{str(report_save_dir)}/action.npy', action_history)
            np.save(f'{str(report_save_dir)}/sim_v_real.npy', sim_v_history)
            # np.save(f'{str(report_save_dir)}/truth_v_real.npy',
            #         eval_env.truth_v_collection)

    return all_episodes_error_accm / all_episodes_mask_accm

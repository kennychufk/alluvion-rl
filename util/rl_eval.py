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

    all_episodes_accumulator = {}

    for metric in eval_env.metrics:
        all_episodes_accumulator[metric] = {'error': 0, 'num_samples': 0}

    for _ in range(len(eval_env.truth_dirs)):
        state, done = eval_env.reset(), False
        if (report_state_action):
            state_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys,
                 get_state_dim()))
            action_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys,
                 get_action_dim()))
            value_history = np.zeros(
                (eval_env._max_episode_steps, eval_env.num_buoys, 2))
            sim_v_history = np.zeros(
                (eval_env._max_episode_steps,
                 eval_env.simulation_sampling.num_samples, 3))
            error_histories = {}
            baseline_histories = {}
            error_num_sample_histories = {}
            for metric in eval_env.metrics:
                error_histories[metric] = np.zeros(eval_env._max_episode_steps)
                baseline_histories[metric] = np.zeros(
                    eval_env._max_episode_steps)
                error_num_sample_histories[metric] = np.zeros(
                    eval_env._max_episode_steps)
            cursor = 0
        dirname_short = Path(eval_env.truth_dir).name[4:15]
        if report_state_action or eval_env.save_visual:
            report_save_dir = Path(
                f'val-{dirname_short}-{run_id}-{model_iteration}-{timestamp_hash}'
            )
            report_save_dir.mkdir(parents=True)
            if eval_env.save_visual:
                eval_env.save_dir_visual = report_save_dir

        accumulator = {}
        within_t_accumulator = {}
        for metric in eval_env.metrics:
            accumulator[metric] = {'error': 0, 'baseline': 0}
            within_t_accumulator[metric] = {'error': 0, 'baseline': 0}
        while not done:
            action = agent.get_action_symmetrized(state)
            value = agent.get_value(state, action)
            action_converted = agent.actor.from_normalized_action(action)

            if (report_state_action):
                state_history[cursor] = state
                action_history[cursor] = action_converted
                value_history[cursor] = value
                sim_v_history[
                    cursor] = eval_env.simulation_sampling.sample_data3.get()
                cursor += 1
            state, reward, local_rewards, done, step_info = eval_env.step(
                action_converted, compute_local_rewards=False)
            for metric in eval_env.metrics:
                error = step_info[metric + '_error']
                baseline = step_info[metric + '_baseline']
                num_samples = step_info[metric + '_num_samples']
                if report_state_action:
                    error_histories[metric][cursor] = error
                    baseline_histories[metric][cursor] = baseline
                    error_num_sample_histories[metric][cursor] = num_samples
                all_episodes_accumulator[metric]['error'] += error
                accumulator[metric]['error'] += error
                accumulator[metric]['baseline'] += baseline
                all_episodes_accumulator[metric]['num_samples'] += num_samples
                if num_samples > 0:
                    within_t_accumulator[metric][
                        'error'] += error / num_samples
                    within_t_accumulator[metric][
                        'baseline'] += baseline / num_samples

        for metric in eval_env.metrics:
            episode_info[
                f'{eval_env.num_buoys}-{dirname_short}-{metric}-a%'] = accumulator[
                    metric]['error'] / accumulator[metric]['baseline']
            episode_info[
                f'{eval_env.num_buoys}-{dirname_short}-{metric}-m%'] = within_t_accumulator[
                    metric]['error'] / within_t_accumulator[metric]['baseline']
        if (report_state_action):
            np.save(f'{str(report_save_dir)}/state.npy', state_history)
            np.save(f'{str(report_save_dir)}/action.npy', action_history)
            np.save(f'{str(report_save_dir)}/value.npy', value_history)
            np.save(f'{str(report_save_dir)}/sim_v_real.npy', sim_v_history)
            np.save(f'{str(report_save_dir)}/kernel_radius_recon.npy',
                    eval_env.real_kernel_radius)
            for metric in eval_env.metrics:
                np.save(f'{str(report_save_dir)}/{metric}_error.npy',
                        error_histories[metric])
                np.save(f'{str(report_save_dir)}/{metric}_baseline.npy',
                        baseline_histories[metric])
                np.save(f'{str(report_save_dir)}/{metric}_num_samples.npy',
                        error_num_sample_histories[metric])
    for metric in eval_env.metrics:
        episode_info[f'overall-{metric}'] = all_episodes_accumulator[metric][
            'error'] / all_episodes_accumulator[metric]['num_samples']

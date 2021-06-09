import os
import csv
import pandas
import pickle
from datetime import datetime


def create_dir_at(path, verbose=True):
    try:
        os.mkdir(path)
    except OSError:
        if verbose:
            print("Creation of the directory %s failed" % path)
    else:
        if verbose:
            print("Successfully created the directory %s " % path)


def csv_ext(filename):
    return filename if filename[-4:] == '.csv' else filename + '.csv'


def pkl_ext(filename):
    return filename if filename[-4:] == '.pkl' else filename + '.pkl'


def save_obj(obj, filename):
    with open(pkl_ext(filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(pkl_ext(filename), 'rb') as f:
        return pickle.load(f)


def datetime_directory(suffix=None, verbose=True):
    now = datetime.now()
    if suffix is None:
        path = 'results/' + now.strftime("%Y%m%d-%H%M%S") + '/'
    else:
        path = 'results/' + now.strftime("%Y%m%d-%H%M%S") + suffix + '/'
    create_dir_at(path, verbose)
    return path


def dict_to_csv(dictionary, filename, indexing='col'):
    filename = csv_ext(filename)
    w = csv.writer(open(filename, 'w'))
    if indexing == 'col':
        w.writerow(dictionary.keys())
        w.writerow(dictionary.values())
    else:
        for key, val in dictionary.items():
            w.writerow([key, val])


def df_to_csv(df, filename):
    filename = csv_ext(filename)
    df.to_csv(filename, index=False, header=True)


def ag_env_directory(exp_dir, agent, env):
    return exp_dir + agent.name + '_' + env.name


def net_directory(env_name):
    return 'results/pretrained/' + env_name


def get_filename(key, exp_dir=None, ag_env_dir=None, net_dir=None, env_name=None, instance_number=None,
                 hidden_layers=None, cst=None):
    """
    Work with unique file names.
    :param key: (str)
    :param exp_dir: (str)
    :param ag_env_dir: (str)
    :param net_dir: (str)
    :param env_name: (str)
    :param instance_number: (int)
    :param hidden_layers: (list)
    :param cst: (float)
    :return: requested filename
    """
    if key == 'global_config':
        return exp_dir + 'global_config'
    elif key == 'env_config':
        return exp_dir + env_name + '_config'
    elif key == 'global_results':
        return ag_env_dir + '/global_results_instance_' + str(instance_number) + '.csv'
    elif key == 'reward_trajectory':
        return ag_env_dir + '/rewards_trajectory_instance_' + str(instance_number) + '.csv'
    elif key == 'q_map':
        return ag_env_dir + '/q_map_instance_' + str(instance_number) + '.csv'
    elif key == 'loss':
        return ag_env_dir + '/loss_instance_' + str(instance_number)
    elif key == 'replay_buffer':
        return ag_env_dir + '/replay_buffer_instance_' + str(instance_number) + '.csv'
    elif key == 'constant_net':
        return net_dir + '/constant_net_' + str(hidden_layers) + '_' + str(round(cst, 1))
    else:
        raise ValueError('Key [' + key + '] not found in get_filename.')


def agents_config_to_md(exp_dir, parameters_list=None, env_name=None):
    filename = exp_dir + 'agents_config.md'
    f = open(filename, 'a+')

    # Environment name before agents config
    if env_name is not None:
        f.write((len(env_name) + 6) * '#' + '\n')
        f.write(3 * ' ' + env_name + '\n')
        f.write((len(env_name) + 6) * '#' + '\n\n')

    # Agents config
    if parameters_list is not None:
        for p in parameters_list:
            max_key_length = max([len(key) for key in p.keys()])
            for key, val in p.items():
                shift = (max_key_length - len(key)) * ' ' + ' : '
                f.write(key + shift + str(val) + '\n')
            f.write('\n')

    f.close()


def benchmark_save(ag_env_dir, agent, env, n_episodes, n_steps, instance_number, trajectory, steps_per_episode,
                   return_per_episode, discounted_return_per_episode, time_per_episode, ag_backup):
    # 1) Global results save
    global_results_df = pandas.DataFrame(columns=['agent', 'environment', 'gamma', 'instance_number', 'episode_number',
                                                  'max_steps', 'performed_steps', 'return', 'discounted_return',
                                                  'time'])
    for i in range(n_episodes):
        global_results_df = global_results_df.append(
            {'agent': agent.name, 'environment': env.name, 'gamma': env.gamma, 'instance_number': instance_number,
             'episode_number': i + 1, 'max_steps': n_steps, 'performed_steps': steps_per_episode[i],
             'return': return_per_episode[i], 'discounted_return': discounted_return_per_episode[i],
             'time': time_per_episode[i]}, ignore_index=True
        )
    global_results_fname = get_filename(key='global_results', ag_env_dir=ag_env_dir, instance_number=instance_number)
    df_to_csv(df=global_results_df, filename=global_results_fname)

    # 2) Reward trajectory save
    trajectory_fname = get_filename(key='reward_trajectory', ag_env_dir=ag_env_dir, instance_number=instance_number)
    df_to_csv(df=trajectory, filename=trajectory_fname)

    # 3) Agent-specific backup
    if ag_backup is not None:
        keys = list(ag_backup.keys())

        # 3.1) Q-table
        if 'q_map' in keys:
            columns = ['x', 'y', 'a', 'q']
            q_map_df = pandas.DataFrame(columns=columns)
            q_map = ag_backup['q_map']
            n_x, n_y, n_a = q_map.shape
            for x in range(1, n_x + 1):
                for y in range(1, n_y + 1):
                    for a in env.actions:
                        q_map_df = q_map_df.append(
                            pandas.DataFrame([[x, y, a, q_map[x - 1][y - 1][a]]], columns=columns))
            q_map_fname = get_filename(key='q_map', ag_env_dir=ag_env_dir, instance_number=instance_number)
            df_to_csv(df=q_map_df, filename=q_map_fname)

        # 3.2) Loss function
        if 'loss' in keys:
            loss_fname = get_filename(key='loss', ag_env_dir=ag_env_dir, instance_number=instance_number)
            save_obj(ag_backup['loss'], filename=loss_fname)

        # 3.2) Replay buffer
        if 'replay_buffer' in keys:
            columns = ['x', 'y', 'frequency']
            replay_buffer_df = pandas.DataFrame(columns=columns)
            for s, freq in ag_backup['replay_buffer'].items():
                replay_buffer_df = replay_buffer_df.append(pandas.DataFrame([[s[0], s[1], freq]], columns=columns))
            replay_buffer_fname = get_filename(key='replay_buffer', ag_env_dir=ag_env_dir,
                                               instance_number=instance_number)
            df_to_csv(df=replay_buffer_df, filename=replay_buffer_fname)

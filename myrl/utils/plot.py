import timeit
import pandas
import itertools
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import drmax.utils.save as save
from drmax.utils.configreader import expand
from drmax.environments.gridworld.gridworld import action_int_to_str


COLORS = [
    'orange',
    'lightblue',
    'peru',
    'tomato',
    'grey',
    'khaki',
    'mediumpurple',
    'pink',
    'yellowgreen',
    'aquamarine'
]
MARKERS = ('o', '^', 's', 'D', 'P', '*')


def rows_to_confidence_intervals_dict(v, confidence=0.9, remove_nan=True):
    n_row = v.shape[0]
    mci = {'mean': [], 'lower_bound': [], 'upper_bound': []}
    for i in range(n_row):
        v_i = v[i][~np.isnan(v[i])] if remove_nan else v[i]
        m, m_lo, m_up = mean_confidence_interval(v_i, confidence=confidence)
        mci['mean'].append(m)
        mci['lower_bound'].append(m_lo)
        mci['upper_bound'].append(m_up)
    return mci


def moving_average(v, window_size):
    v_series = pandas.Series(v)
    windows = v_series.rolling(window_size)
    moving_averages = windows.mean().to_list()
    return moving_averages[window_size - 1:]


def mean_confidence_interval(data, confidence=0.9):
    """
    Compute the mean and confidence interval of the the input data array-like.
    :param data: (array-like)
    :param confidence: probability of the mean to lie in the interval
    :return: (tuple) mean, interval upper-endpoint, interval lower-endpoint
    """
    a = 1.0 * np.array(data)
    if len(a) == 1:
        return a[0], a[0], a[0]
    m, se = np.mean(a), scipy.stats.sem(a)
    if confidence is None:
        return m, None, None
    else:
        h = se * scipy.stats.t.ppf((1.0 + confidence) / 2.0, len(a) - 1)
        return m, m-h, m+h


def plot_trajectories(dictionary, filename, x_key, x_label, y_label, markers_freq, window_size, do_show=False):
    """
    Plot results for all agents
    :param dictionary: (dict) dictionary with the following structure (Example) :
    dictionary = {
        'RMax': {
            x_key: [5, 10, 15, 20, 25, 30]
            'mean': [0, 0, 1, 1, 0, 1]
            'lower_bound': [0, 0, 0, 0, 0, 1]
            'upper_bound': [0, 0, 2, 2, 0, 1]
        },
        'Random': {
            x_key: [5, 10, 15, 20]
            'mean': [0, 0, 1, 0]
            'lower_bound': [0, 0, -2, 0]
            'upper_bound': [0, 0, 3, 0]
        }
    }
    :param filename: (str)
    :param x_label: (str)
    :param y_label: (str)
    :param markers_freq: (int) markers frequency on graphs
    :param window_size: (int) size of the moving average window
    :param do_show: (bool)
    :return:
    """
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=COLORS)
    m = itertools.cycle(MARKERS)
    for key, val in dictionary.items():
        x, y, y_lo, y_up = val[x_key], val['mean'], val['lower_bound'], val['upper_bound']
        # Compute moving average if necessary
        if window_size > 1:
            if window_size > len(x):
                print('WARNING: required window size in moving average exceeds data length (window_size = ' +
                      str(window_size) + ' data length = ' + str(len(x)) + '). Graph could not be generated.')
                return None
            x, y, y_lo, y_up = (moving_average(v, window_size=window_size) for v in (x, y, y_lo, y_up))
        markers_freq = max(int(len(x) / 10), 1) if markers_freq is None else markers_freq
        ax.plot(x, y, label=key, marker=next(m), markevery=markers_freq)
        ax.fill_between(x, y_lo, y_up, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    # ax.set_title(title, loc='center')
    ax.grid()
    if do_show:
        plt.show()
    plt.savefig(filename)
    plt.close('all')


def densities_from_df(df, n_instances, n_x, n_y):
    densities = {}
    for i in range(1, n_instances + 1):
        m = np.zeros(shape=(n_x, n_y))
        dfi = df.loc[df['instance_number'] == i]
        for _, row in dfi.iterrows():
            m[row['x'] - 1, row['y'] - 1] = row['frequency']
        densities[i] = np.rot90(m)
    v_min = df["frequency"].min()
    v_max = df["frequency"].max()
    return densities, v_min, v_max


def plot_losses(loss_per_instance_dict, n_instances, filename, env_name, ag_name):
    fig, axes = plt.subplots(n_instances, 1, figsize=(8, 2 * n_instances))
    if n_instances == 1:
        axes = [axes]
    for i in range(n_instances):
        n_losses = len(loss_per_instance_dict[i + 1])
        axes[i].plot(loss_per_instance_dict[i + 1], c='lightsteelblue')
        if n_losses > 10:
            w = int(n_losses / 10)
            x_ma = np.arange(w / 2, n_losses - w / 2 + 1)
            y_ma = moving_average(loss_per_instance_dict[i + 1], window_size=w)
            axes[i].plot(x_ma, y_ma, c='mediumorchid')
        axes[i].set_title('Instance ' + str(i + 1), loc='center')
    fig.suptitle('Losses: ' + ag_name + ' in ' + env_name)
    plt.subplots_adjust(top=0.85, hspace=0.4)
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def plot_densities(densities, v_min, v_max, n_instances, n_x, n_y, filename, env_name, ag_name, cmap='binary'):  # YlGn
    fig, axes = plt.subplots(n_instances, 2, figsize=(2 * n_x, 3 * n_x * n_instances / 4))
    if n_instances == 1:
        axes = [axes]
    for i in range(n_instances):
        im = axes[i][0].imshow(densities[i + 1], cmap=cmap, vmin=0, vmax=v_max)
        plt.colorbar(im, ax=axes[i][0])
        im = axes[i][1].imshow(densities[i + 1], cmap=cmap, norm=LogNorm(vmin=v_min, vmax=v_max))
        plt.colorbar(im, ax=axes[i][1])
        # Set axis for both standard and log-normalized graphs
        for j in [0, 1]:
            axes[i][j].set_title('Instance ' + str(i + 1), loc='center')
            axes[i][j].set_xticks(np.arange(n_x))
            axes[i][j].set_yticks(np.arange(n_y))
            axes[i][j].set_xticklabels(np.arange(1, n_x + 1))
            axes[i][j].set_yticklabels(np.arange(1, n_y + 1))
    fig.suptitle('Replay buffer : ' + ag_name + ' in ' + env_name + '\n(left = raw and right = log-normalized)')
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def plot_q_table(q, n_instances, n_x, n_y, n_a, filename, env_name, ag_name, compute_mean=True, confidence=0.9):
    n_rows = n_instances + 1 if compute_mean else n_instances
    fig, axes = plt.subplots(n_rows, 4, figsize=(2.5 * n_x, 3 * n_x * n_instances / 4))

    # Per-instance Q-table
    for i in range(1, n_instances + 1):
        qi = q.loc[q['instance_number'] == i]
        v_min, v_max = qi['q'].min(), qi['q'].max()
        for a in range(n_a):
            qia = qi.loc[qi['a'] == a][['x', 'y', 'q']].to_numpy()
            m = np.zeros(shape=(n_x, n_y))
            for x, y, q_value in qia:
                m[int(x) - 1][int(y) - 1] = q_value
            m = np.rot90(m)
            im = axes[i - 1, a].imshow(m, cmap='magma', vmin=v_min, vmax=v_max, aspect='auto')
            axes[i - 1, a].set_xticks(np.arange(n_x))
            axes[i - 1, a].set_yticks(np.arange(n_y))
            axes[i - 1, a].set_xticklabels(np.arange(1, n_x + 1))
            axes[i - 1, a].set_yticklabels(np.arange(1, n_y + 1)[::-1])
            axes[i - 1, a].set_title('Q(s, ' + action_int_to_str(a) + ')', loc='center')
        plt.colorbar(im, ax=axes[i - 1, :])

    # Compute mean Q-table
    if compute_mean:
        m, u = {}, {}
        for a in range(n_a):
            qa = q.loc[q['a'] == a]
            m[a], u[a] = np.zeros(shape=(n_x, n_y)), np.zeros(shape=(n_x, n_y))
            for x in range(1, n_x + 1):
                for y in range(1, n_y + 1):
                    q_xya = qa.loc[(qa['x'] == x) & (qa['y'] == y)]['q'].to_numpy()
                    q_mean, _, q_up = mean_confidence_interval(q_xya, confidence=confidence)
                    m[a][x - 1][y - 1], u[a][x - 1][y - 1] = q_mean, q_up - q_mean
            m[a], u[a] = np.rot90(m[a]), np.rot90(u[a])
        v_min, v_max = np.min([m[a] for a in range(n_a)]), np.max([m[a] for a in range(n_a)])
        threshold = v_min + 0.8 * (v_max - v_min)

        # Plot mean Q-table
        for a in range(n_a):
            im = axes[n_rows - 1, a].imshow(m[a], cmap='magma', vmin=v_min, vmax=v_max, aspect='auto')
            axes[n_rows - 1, a].set_xticks(np.arange(n_x))
            axes[n_rows - 1, a].set_yticks(np.arange(n_y))
            axes[n_rows - 1, a].set_xticklabels(np.arange(1, n_x + 1))
            axes[n_rows - 1, a].set_yticklabels(np.arange(1, n_y + 1)[::-1])
            axes[n_rows - 1, a].set_title('Q(s, ' + action_int_to_str(a) + ')', loc='center')
            for x in range(n_x):
                for y in range(n_y):
                    tx = str(round(m[a][x, y], 2)) + '\n±' + str(round(u[a][x, y], 1))
                    cl = ["white", "black"][int(m[a][x, y] > threshold)]
                    axes[n_rows - 1, a].text(y, x, tx, ha="center", va="center", color=cl)
        plt.colorbar(im, ax=axes[n_rows - 1, :])

    # plt.subplots_adjust(right=0.77, wspace=0.25, hspace=0.5)
    fig.suptitle('Q-tables: ' + ag_name + ' in ' + env_name)
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')


def row_pad(v, length, value=np.nan):
    w = np.full(shape=length - len(v), fill_value=value)
    return np.concatenate((v, w), axis=0)


def retrieve_trajectories(n_instances, ag_env_dir, x_key, file_key):
    trajectories = []
    x = []
    for instance_number in range(1, n_instances + 1):
        filename = save.get_filename(key=file_key, ag_env_dir=ag_env_dir, instance_number=instance_number)
        trajectory_i = pandas.read_csv(filename)
        trajectories.append(trajectory_i)
        if len(x) < len(trajectory_i[x_key]):
            x = trajectory_i[x_key]
    return x, trajectories


def benchmark_plot(exp_dir, agents, environment, markers_freq, reward_plot_ma, return_plot_ma, do_show=False):
    config_filename = save.get_filename(key='global_config', exp_dir=exp_dir)
    config = save.load_obj(config_filename)
    n_env = len(config['environments'])
    matching_env_names = [int(ep['parameters']['name'] == environment.name) for ep in config['environments']]
    n_instances_index = np.dot(matching_env_names, np.arange(n_env))
    n_instances = expand(config['n_instances'], n_env)[n_instances_index]

    # 1) Reward trajectories

    # Buffers to store trajectories
    rewards = {}
    cumulative_rewards = {}
    discounted_rewards = {}
    cumulative_discounted_rewards = {}
    return_per_episode = {}
    discounted_return_per_episode = {}

    for ag in agents:
        ag_env_dir = save.ag_env_directory(exp_dir=exp_dir, agent=ag, env=environment)

        steps, reward_trajectories = retrieve_trajectories(n_instances, ag_env_dir, x_key='step',
                                                           file_key='reward_trajectory')
        episodes, return_trajectories = retrieve_trajectories(n_instances, ag_env_dir, x_key='episode_number',
                                                              file_key='global_results')

        for col_key, x_key, x, buffer, trajectories in [
            ('r', 'step', steps, rewards, reward_trajectories),
            ('cr', 'step', steps, cumulative_rewards, reward_trajectories),
            ('dr', 'step', steps, discounted_rewards, reward_trajectories),
            ('cdr', 'step', steps, cumulative_discounted_rewards, reward_trajectories),
            ('return', 'episode', episodes, return_per_episode, return_trajectories),
            ('discounted_return', 'episode', episodes, discounted_return_per_episode, return_trajectories),
        ]:
            max_length = max([len(t[col_key]) for t in trajectories])
            v = np.transpose([row_pad(t[col_key].to_numpy(), length=max_length) for t in trajectories])
            ag_traj_dict = rows_to_confidence_intervals_dict(v)  # mean lower upper dictionary
            ag_traj_dict[x_key] = x
            buffer[ag.name] = ag_traj_dict

    for ma in reward_plot_ma:
        for dct, sfx, x_key, xlb, ylb in [
            (rewards, '_rewards_ma' + str(ma) + '.pdf', 'step', 'Decision epoch', 'Rewards'),
            (cumulative_rewards, '_cumulative_rewards_ma' + str(ma) + '.pdf', 'step', 'Decision epoch',
             'Cumulative Rewards'),
            (discounted_rewards, '_discounted_rewards_ma' + str(ma) + '.pdf', 'step', 'Decision epoch',
             'Discounted Rewards'),
            (cumulative_discounted_rewards, '_cumulative_discounted_rewards_ma' + str(ma) + '.pdf', 'step',
             'Decision epoch', 'Cumulative Discounted Rewards')
        ]:
            plot_trajectories(dct, filename=exp_dir + environment.name + sfx, x_key=x_key, x_label=xlb, y_label=ylb,
                              markers_freq=markers_freq, window_size=ma, do_show=do_show)

    for ma in return_plot_ma:
        for dct, sfx, x_key, xlb, ylb in [
            (return_per_episode, '_return_per_episode_ma' + str(ma) + '.pdf', 'episode', 'Episode',
             'Return per episode'),
            (discounted_return_per_episode, '_discounted_return_per_episode_ma' + str(ma) + '.pdf', 'episode',
             'Episode', 'Discounted return per episode')
        ]:
            plot_trajectories(dct, filename=exp_dir + environment.name + sfx, x_key=x_key, x_label=xlb, y_label=ylb,
                              markers_freq=markers_freq, window_size=ma, do_show=do_show)

    # 2) Agent-specific data (if it exists)
    for ag in agents:
        ag_env_dir = save.ag_env_directory(exp_dir=exp_dir, agent=ag, env=environment)

        # 2.1) Q-map
        try:
            q = pandas.DataFrame()
            for instance_number in range(1, n_instances + 1):
                q_map_i_fname = save.get_filename(key='q_map', ag_env_dir=ag_env_dir, instance_number=instance_number)
                q_map_i = pandas.read_csv(q_map_i_fname)
                q_map_i['instance_number'] = q_map_i.shape[0] * [instance_number]
                q = pandas.concat([q, q_map_i])
            n_x, n_y, n_a = max(q['x']), max(q['y']), 4  # max(df['a']) + 1
            # Plot Q-table for each instance + mean Q-table
            fname = exp_dir + environment.name + '_' + ag.name + '_q_table.pdf'
            plot_q_table(q, n_instances, n_x, n_y, n_a, filename=fname, env_name=environment.name, ag_name=ag.name,
                         compute_mean=True, confidence=0.9)
        except IOError:
            pass

        # 2.2) Replay-buffer
        try:
            rb = pandas.DataFrame()
            for instance_number in range(1, n_instances + 1):
                replay_buffer_i_fname = save.get_filename(key='replay_buffer', ag_env_dir=ag_env_dir,
                                                          instance_number=instance_number)
                replay_buffer_i = pandas.read_csv(replay_buffer_i_fname)
                replay_buffer_i['instance_number'] = replay_buffer_i.shape[0] * [instance_number]
                rb = pandas.concat([rb, replay_buffer_i])

            env_config_filename = save.get_filename(key='env_config', exp_dir=exp_dir, env_name=environment.name)
            env_config = save.load_obj(filename=env_config_filename)
            grid_width, grid_height = env_config['width'], env_config['height']

            densities, v_min, v_max = densities_from_df(rb, n_instances, n_x=grid_width, n_y=grid_height)
            fname = exp_dir + environment.name + '_' + ag.name + '_replay_buffer.pdf'
            plot_densities(densities, v_min, v_max, n_instances, n_x=grid_width, n_y=grid_height, filename=fname,
                           env_name=environment.name, ag_name=ag.name)
        except IOError:
            pass

        # 2.3) Loss
        try:
            loss_per_instance_dict = {}
            for instance_number in range(1, n_instances + 1):
                loss_fname = save.get_filename(key='loss', ag_env_dir=ag_env_dir, instance_number=instance_number)
                loss_i = save.load_obj(filename=loss_fname)
                loss_per_instance_dict[instance_number] = loss_i
            fname = exp_dir + environment.name + '_' + ag.name + '_losses.pdf'
            plot_losses(loss_per_instance_dict, n_instances, filename=fname, env_name=environment.name,
                        ag_name=ag.name)
        except IOError:
            pass

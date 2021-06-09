import time
import pandas
import numpy as np
import dill
import multiprocessing
import torch
from tqdm import trange

import drmax.utils.configreader as reader
import drmax.utils.save as save
import drmax.utils.plot as plot
from drmax.environments.gridworld.gridworld import GridWorld
from drmax.models import QNetwork


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def rl_experiment(agent, env, instance_number, n_instances, n_episodes, n_steps, timeout, save_ma, verbose=True):
    gamma = env.gamma

    # Save buffers
    trajectory = [[0, 0.0, 0.0, 0.0, 0.0]]  # Rewards trajectory
    trajectory_buffer = [trajectory[-1][1:]]

    steps_per_episode = np.zeros(n_episodes, dtype=int)
    return_per_episode = np.zeros(n_episodes)
    discounted_return_per_episode = np.zeros(n_episodes)
    time_per_episode = np.zeros(n_episodes)

    if verbose:
        print('Environment :', env.name)
        print('Agent       :', agent.name)
        print('Instance    :', str(instance_number), "/", str(n_instances))

    # For each episode.
    step_counter = 1
    is_timeout = False
    for episode in trange(n_episodes, desc='Episodes'):

        # Compute initial state and reward
        s = env.get_initial_state()
        r = 0.0
        is_terminal = False
        init_time = time.clock()
        reached_terminal_state = False

        for step in range(n_steps):

            # Compute the agent's policy
            a = agent.act(s, r, is_terminal)

            # Execute in MDP
            r, s_p, is_terminal = env.step(s, a)

            # Save trajectory
            if save_ma == 1:
                trajectory.append([
                    step_counter,  # step
                    r,  # reward
                    r * gamma ** float(step),  # discounted reward
                    r + trajectory[-1][3],  # cumulative reward
                    r * gamma ** float(step) + trajectory[-1][4]  # cumulative discounted reward
                ])
            else:
                trajectory_buffer.append([
                    r,  # reward
                    r * gamma ** float(step),  # discounted reward
                    r + trajectory_buffer[-1][2],  # cumulative reward
                    r * gamma ** float(step) + trajectory_buffer[-1][3]  # cumulative discounted reward
                ])
                if len(trajectory_buffer) == save_ma + 1:
                    trajectory_buffer = trajectory_buffer[1:]
                    trajectory.append([step_counter] + list(np.mean(trajectory_buffer, axis=0)))
                    trajectory_buffer = [trajectory_buffer[-1]]

            # Store return per episode
            return_per_episode[episode] += r
            discounted_return_per_episode[episode] += r * (gamma ** step)

            # Terminal / timeout check
            is_timeout = True if timeout is not None and step_counter == timeout else False
            if is_terminal or is_timeout:
                steps_per_episode[episode] = step + 1
                reached_terminal_state = True

                # Save remaining trajectory_buffer before break if timeout or end of last episode
                if save_ma > 1:
                    if is_timeout or episode + 1 == n_episodes:
                        trajectory_buffer = trajectory_buffer[1:]
                        trajectory.append([step_counter] + list(np.mean(trajectory_buffer, axis=0)))

                break

            # Update pointer
            s = s_p
            step_counter += 1

        # Record time and number of steps
        if not reached_terminal_state:
            steps_per_episode[episode] = n_steps
        time_per_episode[episode] = time.clock() - init_time

        # A final update
        agent.act(s, r, is_terminal)

        # Reset the MDP, tell the agent the episode is over
        # env.reset()
        agent.end_of_episode()

        # Timeout break
        if is_timeout:
            break

    # Re-initialization for multiple instances
    ag_backup = agent.get_backup()

    # If Gridworld and agent is Q-based, save Q-values for visual
    if type(env) == GridWorld:
        ag_backup = grid_world_visuals(env, agent, ag_backup)

    # Re-initialization for multiple instances
    agent.re_init()

    trajectory.pop(0)  # cleaning trick
    trajectory_df = pandas.DataFrame(trajectory, columns=['step', 'r', 'dr', 'cr', 'cdr'])

    if verbose:
        print('Successfully ended', env.name, agent.name, '(instance', str(instance_number), "/", str(n_instances), ')')

    return trajectory_df, steps_per_episode, return_per_episode, discounted_return_per_episode, time_per_episode, \
           ag_backup


def grid_world_visuals(env, agent, ag_backup):
    """
    Backup for visuals in gridworld environment.
    :param env:
    :param agent:
    :param ag_backup:
    :return:
    """
    if ag_backup is None:  # Replace by dict if None
        ag_backup = {}

    try:
        if agent.learns_Q_function:
            ag_backup['q_map'] = env.get_q_map(agent)
    except AttributeError:
        pass

    try:
        if agent.has_replay_buffer:
            ag_backup['replay_buffer'] = agent.get_replay_buffer()
    except AttributeError:
        pass

    return ag_backup


def run(ag, env, instance_number, n_ins, n_ep, n_st, timeout, save_ma, verbose, do_save, ag_env_dir):
    tr, steps_per_episode, return_per_episode, discounted_return_per_episode, time_per_episode, ag_backup = \
        rl_experiment(agent=ag, env=env, instance_number=instance_number, n_instances=n_ins, n_episodes=n_ep,
                      n_steps=n_st, timeout=timeout, save_ma=save_ma, verbose=verbose)

    # Save this environment-agent pair results
    if do_save:
        save.benchmark_save(ag_env_dir=ag_env_dir, agent=ag, env=env, n_episodes=n_ep, n_steps=n_st,
                            instance_number=instance_number, trajectory=tr, steps_per_episode=steps_per_episode,
                            return_per_episode=return_per_episode,
                            discounted_return_per_episode=discounted_return_per_episode,
                            time_per_episode=time_per_episode, ag_backup=ag_backup)


def benchmark(config, do_run=True, do_save=True, exp_dir=None, do_plot=True, do_show=False, verbose=True):
    # Create saving directory and save configuration
    if do_save:
        if exp_dir is None:
            exp_dir = save.datetime_directory(verbose=verbose)
        config_filename = save.get_filename(key='global_config', exp_dir=exp_dir)
        save.dict_to_csv(dictionary=config, filename=config_filename)  # CSV
        save.save_obj(obj=config, filename=config_filename)  # PKL

    # Read configuration file
    env_ag, n_env, n_episodes, n_steps, n_instances, timeouts, n_processes, is_multi_proc, save_ma, env_names =\
        reader.benchmark_settings_from_config(config=config, exp_dir=exp_dir)

    # Check values
    assert n_env == len(n_episodes), 'Please provide a number of episodes for each environment (provided ' \
                                     + str(len(n_episodes)) + ')'
    assert n_env == len(n_steps), 'Please provide a number of steps for each environment (provided ' \
                                  + str(len(n_steps)) + ')'
    assert n_env == len(n_instances), 'Please provide a number of instances for each environment (provided ' \
                                      + str(len(n_instances)) + ')'
    assert n_env == len(timeouts), 'Please provide a timeout value for each environment (provided '\
                                   + str(len(timeouts)) + ')'
    assert len(env_names) == len(set(env_names)), 'Please provide unique names for environments: ' + str(env_names)

    # Create pool of workers
    if is_multi_proc:
        pool = multiprocessing.Pool(processes=n_processes)
        jobs = []

    # Verbose
    if verbose:
        print('Running: ', n_env, 'environments with (respectfully)', [len(env_ag[env]) for env in env_ag],
              'agent(s) and', n_instances, 'instance(s).')

    if do_run:
        for env in env_ag:
            # 0) Save environment info
            if do_save:
                env_config_filename = save.get_filename(key='env_config', exp_dir=exp_dir, env_name=env.name)
                save.dict_to_csv(dictionary=env.get_info(), filename=env_config_filename)  # CSV
                save.save_obj(obj=env.get_info(), filename=env_config_filename)  # PKL

            # 1) Run all agents for this environment
            n_ep = n_episodes.pop(0)
            n_st = n_steps.pop(0)
            n_ins = n_instances.pop(0)
            timeout = timeouts.pop(0)
            for ag in env_ag[env]:
                if do_save:
                    ag_env_dir = save.ag_env_directory(exp_dir, ag, env)
                    save.create_dir_at(ag_env_dir)

                for instance_number in range(1, n_ins + 1):
                    if is_multi_proc:
                        job = apply_async(pool, run, (ag, env, instance_number, n_ins, n_ep, n_st, timeout,
                                                      save_ma, verbose, do_save, ag_env_dir))
                        jobs.append(job)
                    else:
                        run(ag, env, instance_number, n_ins, n_ep, n_st, timeout, save_ma, verbose, do_save, ag_env_dir)

    # Achieve all jobs
    if is_multi_proc:
        for job in jobs:
            job.get()

    # Verbose
    if verbose:
        print('End of run: ', n_env, 'environments with (respectfully)', [len(env_ag[env]) for env in env_ag],
              'agent(s) and', n_instances, 'instance(s).')

    # Plot all agents for this environment
    if do_plot:
        for env in env_ag:
            plot.benchmark_plot(exp_dir=exp_dir, agents=env_ag[env], environment=env,
                                markers_freq=config['markers_freq'], reward_plot_ma=config['reward_plot_ma'],
                                return_plot_ma=config['return_plot_ma'], do_show=do_show)


def generate_uniform_tensors(n_tensors, dtype, dimension, magnitude):
    """
    TODO: this function must be in utils AND should be called in ODQN.
    :param n_tensors: (int)
    :param dtype: ()
    :param dimension: (int)
    :param magnitude: (tuple)
    :return: (tensor)
    """
    assert len(set(dtype)) == 1, 'Multiple data type not implemented yet.'
    dtype = dtype[0]
    min_magnitude, max_magnitude = magnitude
    uniform_scale = torch.diag(torch.tensor(max_magnitude - min_magnitude, dtype=torch.float))
    uniform_add = torch.tensor(min_magnitude, dtype=torch.float)
    tensors = torch.rand(size=(n_tensors, dimension))
    tensors = torch.add(uniform_add, torch.matmul(tensors, uniform_scale))
    if dtype is float:
        return tensors
    elif dtype is int:
        return torch.floor(tensors)
    else:
        raise ValueError('Data type [' + dtype + '] not implemented in uniform data generation.')


def train_constant_net(config, env_config, do_save):
    # Create environment
    env = reader.environment_from_env_config(env_config)

    # Retrieve environment dimensions
    state_dimension = env.get_state_dimension()
    state_dtype = env.get_state_dtype()
    state_magnitude = env.get_state_magnitude()
    n_a = len(env.actions)

    # Initlialize Q-network
    net = QNetwork(input_size=state_dimension, output_size=n_a, hidden_layers=config['hidden_layers'])

    # Create uniform input
    batch = generate_uniform_tensors(n_tensors=config['batch_size'], dtype=state_dtype, dimension=state_dimension,
                                     magnitude=state_magnitude)

    # Train constant Q-network
    cst = 1.0 / (1.0 - env_config['parameters']['gamma'])
    norm = 0.5 / float(config['minibatch_size'])
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    for _ in trange(config['n_epochs'], desc='Epochs'):
        # Sample minibatch
        indice = torch.randint(low=0, high=len(batch), size=(config['minibatch_size'],))
        mini_batch = batch[indice]

        # Forward + compute loss
        y = net(mini_batch)
        loss = norm * torch.add(input=y, other=-cst).pow(2).sum()

        # Zero gradients, perform a backward pass (gradient update), and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save constant Q-network
    if do_save:
        net_dir = save.net_directory(env_name=env_config['parameters']['name'])
        save.create_dir_at(net_dir, verbose=False)
        filename = save.get_filename(key='constant_net', net_dir=net_dir, hidden_layers=config['hidden_layers'],
                                     cst=cst)
        net.save(filename)


def train_multiple_constant_nets(config, do_save=True, verbose=True):
    """
    Fit a Q-network to 1 / (1 - gamma).
    :param config: (dict)
    :param do_save: (bool)
    :param verbose: (bool)
    :return: None
    """

    # Create pool of workers
    is_multi_proc, n_processes = reader.multiprocessing_settings_from_config(config=config)
    if is_multi_proc:
        pool = multiprocessing.Pool(processes=n_processes)
        jobs = []

    # Run all trainings
    for env_config in config['environments']:
        if is_multi_proc:
            job = apply_async(pool, train_constant_net, (config, env_config, do_save))
            jobs.append(job)
        else:
            train_constant_net(config=config, env_config=env_config, do_save=do_save)

    # Achieve all jobs
    if is_multi_proc:
        for job in jobs:
            job.get()

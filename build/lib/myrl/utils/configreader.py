import torch
import gym
import multiprocessing

from drmax.utils.save import agents_config_to_md

from drmax.agents.random import Random
from drmax.agents.rmax import RMax
from drmax.agents.qlearning import QLearning
from drmax.agents.dqn import DQN
from drmax.agents.odqn import ODQN

from drmax.environments.gridworld.gridworld import GridWorld
from drmax.environments.cartpole import CartPole
from drmax.environments.mountain_car import MountainCar
from drmax.environments.acrobot import Acrobot
from drmax.environments.pendulum import Pendulum
import drmax.environments.gym_binding as gym_binding


def expand(v, n):
    return n * [v] if type(v) is not list else v


def filtered_config_from_config(config):
    """
    Remove items from 'environments', 'n_episodes', 'n_steps', 'n_instances', 'timeout' lists in config.
    The removed items are those whose index matches a 0 in config['filter'].
    :param config: (dict) main configuration dict
    :return: (dict) the filtered main configuration dict
    """
    if 'filter' in config.keys() and config['filter'] is not None:
        f = config['filter']
        for key in ['environments', 'n_episodes', 'n_steps', 'n_instances', 'timeout']:
            if type(config[key]) is list:
                config[key] = [config[key][i] for i in range(len(f)) if f[i]]
    return config


def multiprocessing_settings_from_config(config):
    n_processes = multiprocessing.cpu_count() if config['n_processes'] is None else config['n_processes']
    is_multi_proc = n_processes > 1
    return is_multi_proc, n_processes


def benchmark_settings_from_config(config, exp_dir):
    env_ag = environments_agents_from_config(config, exp_dir)
    n_env = len(env_ag.keys())
    n_episodes = expand(config['n_episodes'], n_env)
    n_steps = expand(config['n_steps'], n_env)
    n_instances = expand(config['n_instances'], n_env)
    timeouts = expand(config['timeout'], n_env)
    is_multi_proc, n_processes = multiprocessing_settings_from_config(config=config)
    save_ma = config['save_ma']
    env_names = [ep['parameters']['name'] for ep in config['environments']]

    return env_ag, n_env, n_episodes, n_steps, n_instances, timeouts, n_processes, is_multi_proc, save_ma, env_names


def agent_from_class(agent_class, p, env, device):
    if agent_class == 'random':
        return Random(name=p['name'], actions=env.actions, gamma=env.gamma)
    elif agent_class == 'rmax':
        return RMax(name=p['name'], actions=env.actions, gamma=env.gamma, n_known=p['n_known'],
                    epsilon_q=p['epsilon_q'])
    elif agent_class == 'qlearning':
        return QLearning(name=p['name'], actions=env.actions, gamma=env.gamma, learning_rate=p['learning_rate'],
                         epsilon=p['epsilon'])
    elif agent_class == 'dqn':
        return DQN(name=p['name'], env_name=env.name, state_dimension=env.get_state_dimension(), actions=env.actions,
                   gamma=env.gamma, learning_rate=p['learning_rate'], epsilon=p['epsilon'],
                   epsilon_decay=p['epsilon_decay'], replay_memory_size=p['replay_memory_size'],
                   minibatch_size=p['minibatch_size'], double_dqn=p['double_dqn'], hidden_layers=p['hidden_layers'],
                   target_update_period=p['target_update_period'], n_gradient_updates=p['n_gradient_updates'],
                   load_constant_net=p['load_constant_net'], device=device)
    elif agent_class == 'odqn':
        return ODQN(name=p['name'], state_dimension=env.get_state_dimension(), actions=env.actions, gamma=env.gamma,
                    learning_rate=p['learning_rate'], replay_memory_size=p['replay_memory_size'],
                    minibatch_size=p['minibatch_size'], double_dqn=p['double_dqn'], hidden_layers=p['hidden_layers'],
                    target_update_period=p['target_update_period'], n_gradient_updates=p['n_gradient_updates'],
                    data_ratio=p['data_ratio'], alpha=p['alpha'], alpha_one_time=p['alpha_one_time'],
                    state_augmentation_type=p['state_augmentation_type'],
                    state_magnitude=env.get_state_magnitude(), state_dtype=env.get_state_dtype(),
                    cov_multiplier=p['cov_multiplier'], optimistic_factor=p['optimistic_factor'],
                    include_symmetric=p['include_symmetric'], device=device)
    else:
        raise ValueError('Agent class [' + agent_class + '] not found.')


def agents_from_config(config, env, exp_dir, param_in_name=False):
    agents = []
    device = device_from_config(config)

    for a in config['agents']:
        grid_search = a['grid_search']
        generic_name = a['parameters']['name']

        if grid_search:  # Grid search: build cartesian product of all parameters sets
            # Retrieve each parameter combination
            # del a['parameters']['name']
            p_grid = [{'name': generic_name}]
            for key, val_list in a['parameters'].items():
                if key is not 'name':
                    buffer = []
                    for i in range(len(p_grid)):
                        for j in range(len(val_list)):
                            p_grid_i = p_grid[i].copy()
                            p_grid_i[key] = val_list[j]
                            if param_in_name:
                                p_grid_i['name'] += '_' + key + '=' + str(p_grid_i[key])
                            buffer.append(p_grid_i)
                    p_grid = buffer

            # Rename agents by numbering them if parameters not included in name
            if not param_in_name:
                for i in range(len(p_grid)):
                    p_grid[i]['name'] += '_' + str(i + 1)

            # Save all configurations
            agents_config_to_md(exp_dir, parameters_list=p_grid)

            # Build agents
            for p in p_grid:
                agents.append(agent_from_class(agent_class=a['class'], p=p, env=env, device=device))

        else:  # Single agent
            p = a['parameters']
            agents_config_to_md(exp_dir, parameters_list=[p])
            agents.append(agent_from_class(agent_class=a['class'], p=p, env=env, device=device))

    return agents


def environment_from_env_config(env_config):
    env_class = env_config['class']
    p = env_config['parameters']

    if env_class == 'gridworld':
        return GridWorld(name=p['name'], gamma=p['gamma'], grid_name=p['grid_name'], size=p['size'],
                         slip_probability=p['slip_probability'], is_goal_terminal=p['is_goal_terminal'])
    elif env_class == 'cartpole':
        return CartPole(name=p['name'], gamma=p['gamma'], sparse_reward=p['sparse_reward'])
    elif env_class == 'mountaincar':
        return MountainCar(name=p['name'], gamma=p['gamma'], sparse_reward=p['sparse_reward'])
    elif env_class == 'acrobot':
        return Acrobot(name=p['name'], gamma=p['gamma'])
    elif env_class == 'pendulum':
        return Pendulum(name=p['name'], gamma=p['gamma'], sparse_reward=p['sparse_reward'])
    elif env_class == 'gym':
        return gym_binding.make(name=p['name'], gamma=p['gamma'])
    elif env_class == 'atari':
        # env = gym.make(p['name'])
        # env.frameskip = p['frameskip']
        return gym.make(p['name'])
    else:
        raise ValueError('Environment class [' + env_class + '] not implemented.')


def environments_agents_from_config(config, exp_dir):
    environments_agents = {}
    for env_config in config['environments']:
        env = environment_from_env_config(env_config)
        agents_config_to_md(exp_dir, env_name=env.name)
        environments_agents[env] = agents_from_config(config, env, exp_dir)

    return environments_agents


def device_from_config(config):
    if config['gpu'] and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

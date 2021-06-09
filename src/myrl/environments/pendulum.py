"""
Pendulum system
Inspired from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

import numpy as np

from drmax.environments.environment import Environment
from drmax.environments.state import State


class PendulumState(State):
    def __init__(self, theta, theta_dot):
        State.__init__(self, data=[theta, theta_dot])
        self.theta = theta
        self.theta_dot = theta_dot

    def dim(self):
        return 2

    def raw(self):
        return np.array([self.theta, self.theta_dot])


class Pendulum(Environment):

    def __init__(self, name, gamma, sparse_reward):

        self.sparse_reward = sparse_reward
        self.max_speed = 8.0
        self.max_torque = 2.0
        step = 1.0
        self.available_torque = np.arange(-self.max_torque, self.max_torque + step, step)
        actions = np.arange(len(self.available_torque))  # Discrete actions
        self.dt = 0.05
        self.gravity = 9.81
        self.mass = 1.0
        self.length = 1.0
        self.high = np.array([np.pi, self.max_speed], dtype=np.float32)

        self.max_cost = self.cost(np.pi, self.max_speed, self.max_torque)

        Environment.__init__(self, name=name, actions=actions, gamma=gamma)

    def get_state_dimension(self):
        return 2

    def get_state_dtype(self):
        return [float, float]

    def get_state_magnitude(self):
        return -self.high, self.high

    def get_initial_state(self):
        return PendulumState(
            theta=3.14,
            theta_dot=0.0
        )

    def step(self, s, a):
        theta, theta_dot = s.theta, s.theta_dot
        torque = self.available_torque[a]

        theta_dot_p = theta_dot + (- 3.0 * self.gravity / (2.0 * self.length) * np.sin(theta + np.pi) +
                                   3.0 / (self.mass * self.length ** 2) * torque) * self.dt
        theta_dot_p = np.clip(theta_dot_p, -self.max_speed, self.max_speed)
        theta_p = wrap_angle(theta + theta_dot_p * self.dt)

        s_p = PendulumState(theta=theta_p, theta_dot=theta_dot_p)

        cost_ratio = self.cost(theta_p, theta_dot_p, torque) / self.max_cost
        if self.sparse_reward:
            r = 1.0 if cost_ratio < 0.01 else 0.0
        else:
            r = 1.0 - cost_ratio

        return r, s_p, self.is_terminal(s_p)

    def is_terminal(self, s):
        return False

    def cost(self, theta, theta_dot, torque):
        return theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * torque ** 2

    def get_info(self):
        """
        Get general information to be saved on disk.
        """
        return {
            'name': self.name,
            'actions': self.actions,
            'gamma': self.gamma,
            'max_speed': self.max_speed,
            'max_torque': self.max_torque,
            'dt': self.dt,
            'gravity': self.gravity,
            'mass': self.mass,
            'length': self.length
        }


def wrap_angle(x):
    """
    Wrap angle in [- pi, pi]
    :param x: (float) angle
    :return: (float) wrapped angle
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi

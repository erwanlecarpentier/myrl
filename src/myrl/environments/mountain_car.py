"""
Mountain car environment
"""

import math
import numpy as np

from drmax.environments.environment import Environment
from drmax.environments.state import State


class MountainCarState(State):
    def __init__(self, x, x_dot):
        State.__init__(self, data=[x, x_dot])
        self.x = x
        self.x_dot = x_dot

    def dim(self):
        return 2

    def raw(self):
        return np.array([self.x, self.x_dot])


class MountainCar(Environment):

    def __init__(self, name, gamma, sparse_reward):
        Environment.__init__(self, name=name, actions=[0, 1, 2], gamma=gamma)

        self.sparse_reward = sparse_reward
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0.5 * self.max_speed
        self.force = 0.001
        self.force *= 0.5  # Increased difficulty
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

    def get_state_dimension(self):
        return 2

    def get_state_dtype(self):
        return [float, float]

    def get_state_magnitude(self):
        return self.low, self.high

    def get_initial_state(self):
        return MountainCarState(0.0, 0.0)

    def step(self, s, a):
        """
        :param s: state
        :param a: action
        :return: r, s_p, is_terminal(s_p)
        """
        x, x_dot = s.x, s.x_dot

        x_dot += (a - 1) * self.force + math.cos(3 * x) * (-self.gravity)
        x_dot = np.clip(x_dot, -self.max_speed, self.max_speed)
        x += x_dot
        x = np.clip(x, self.min_position, self.max_position)
        if x <= self.min_position and x_dot < 0.0:
            x_dot = 0.0

        s_p = MountainCarState(x, x_dot)

        done = self.is_terminal(s_p)

        if self.sparse_reward:
            r = float(done)
        else:
            r = 0.99 * float(done) + 0.01 / (1.0 + abs(x - self.goal_position))

        return r, s_p, done

    def is_terminal(self, s):
        return bool(s.x >= self.goal_position and s.x_dot <= self.goal_velocity)

    def get_info(self):
        """
        Get general information to be saved on disk.
        """
        return {
            'name': self.name,
            'actions': self.actions,
            'gamma': self.gamma,
            'min_position': self.min_position,
            'max_position': self.max_position,
            'max_speed': self.max_speed,
            'goal_position': self.goal_position,
            'goal_velocity': self.goal_velocity,
            'force': self.force,
            'gravity': self.gravity
        }

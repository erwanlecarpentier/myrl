"""
Cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import numpy as np

from drmax.environments.environment import Environment
from drmax.environments.state import State


class CartPoleState(State):
    def __init__(self, x, x_dot, theta, theta_dot):
        State.__init__(self, data=[x, x_dot, theta, theta_dot])
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def dim(self):
        return 4

    def raw(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])


class CartPole(Environment):

    def __init__(self, name, gamma, sparse_reward):
        Environment.__init__(self, name=name, actions=[0, 1], gamma=gamma)

        self.sparse_reward = sparse_reward
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Thresholds for terminal state
        self.theta_threshold_radians = 12.0 * np.pi / 180.0
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds.
        # Empirical estimates of maximum: [4.8, 2.7, 0.21, 3]
        self.high = np.array([
            self.x_threshold,
            2.7,  # np.finfo(np.float32).max,
            self.theta_threshold_radians,
            3.0  # np.finfo(np.float32).max
        ], dtype=np.float32)

        self.steps_beyond_done = None

    def get_state_dimension(self):
        return 4

    def get_state_dtype(self):
        return [float, float, float, float]

    def get_state_magnitude(self):
        return -self.high, self.high

    def get_initial_state(self):
        return CartPoleState(0., 0., 0., 0.)

    def step(self, s, a):
        """
        :param s: state
        :param a: action
        :return: r, s_p, is_terminal(s_p)
        """
        x, x_dot, theta, theta_dot = s.raw()

        force = self.force_mag if a == 1 else -self.force_mag
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * cos_theta ** 2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_acc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * theta_acc
            theta = theta + self.tau * theta_dot

        s_p = CartPoleState(x, x_dot, theta, theta_dot)

        done = self.is_terminal(s_p)

        if self.sparse_reward:
            r = 1.0 if abs(theta) < 0.05 * self.theta_threshold_radians and abs(s.x) < 0.33 * self.x_threshold else 0.0
        else:
            # r = 1.0 - abs(theta) / self.theta_threshold_radians
            r = 0.0 if done else 1.0

        return r, s_p, done

    def is_terminal(self, s):
        return bool(
            abs(s.x) > self.x_threshold
            or abs(s.theta) > self.theta_threshold_radians
        )

    def get_info(self):
        """
        Get general information to be saved on disk.
        """
        return {
            'name': self.name,
            'actions': self.actions,
            'gamma': self.gamma,
            'gravity': self.gravity,
            'masscart': self.masscart,
            'masspole': self.masspole,
            'total_mass': self.total_mass,
            'length': self.length,
            'polemass_length': self.polemass_length,
            'force_mag': self.force_mag,
            'tau': self.tau,
            'kinematics_integrator': self.kinematics_integrator,
            'theta_threshold_radians': self.theta_threshold_radians,
            'x_threshold': self.x_threshold
        }

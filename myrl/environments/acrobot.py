"""
Acrobot system
Inspired from https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
"""

import numpy as np

from drmax.environments.environment import Environment
from drmax.environments.state import State


class AcrobotState(State):
    def __init__(self, theta_1, theta_2, theta_dot_1, theta_dot_2):
        State.__init__(self, data=[theta_1, theta_2, theta_dot_1, theta_dot_2])
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_dot_1 = theta_dot_1
        self.theta_dot_2 = theta_dot_2

    def dim(self):
        return 4

    def raw(self):
        return np.array([self.theta_1, self.theta_2, self.theta_dot_1, self.theta_dot_2])


class Acrobot(Environment):

    def __init__(self, name, gamma, torque_noise_max=0.0):
        Environment.__init__(self, name=name, actions=[0, 1, 2], gamma=gamma)

        self.dt = 0.2
        self.link_length_1 = 1.0  # [m]
        self.link_length_2 = 1.0  # [m]
        self.link_mass_1 = 1.0  # [kg] mass of link 1
        self.link_mass_2 = 1.0  # [kg] mass of link 2
        self.link_com_pos_1 = 0.5  #: [m] position of the center of mass of link 1
        self.link_com_pos_2 = 0.5  #: [m] position of the center of mass of link 2
        self.link_moi = 1.  # moments of inertia for both links
        self.max_velocity_1 = 4.0 * np.pi
        self.max_velocity_2 = 9.0 * np.pi
        self.available_torque = [-1.0, 0.0, +1.0]
        self.torque_noise_max = torque_noise_max

        self.book_or_nips = 'book'  # use dynamics equations from the nips paper or the book

        self.high = np.array([np.pi, np.pi, self.max_velocity_1, self.max_velocity_2], dtype=np.float32)

    def get_state_dimension(self):
        return 4

    def get_state_dtype(self):
        return [float, float, float, float]

    def get_state_magnitude(self):
        return -self.high, self.high

    def get_initial_state(self):
        return AcrobotState(
            theta_1=0.0,
            theta_2=0.0,
            theta_dot_1=0.0,
            theta_dot_2=0.0
        )

    def step(self, s, a):
        """
        :param s: state
        :param a: action
        :return: r, s_p, is_terminal(s_p)
        """
        s = s.raw()
        torque = self.available_torque[a]
        if self.torque_noise_max > 0.0:
            torque += np.random.uniform(-self.torque_noise_max, self.torque_noise_max)
        s_augmented = np.append(s, torque)

        s_p = rk4(self._dsdt, s_augmented, [0, self.dt])
        s_p = s_p[-1]  # only get final time-step of integration
        s_p = s_p[:4]  # omit action

        s_p[0] = wrap_angle(s_p[0])
        s_p[1] = wrap_angle(s_p[1])
        s_p[2] = bound(s_p[2], -self.max_velocity_1, self.max_velocity_1)
        s_p[3] = bound(s_p[3], -self.max_velocity_2, self.max_velocity_2)

        s_p = AcrobotState(
            theta_1=s_p[0],
            theta_2=s_p[1],
            theta_dot_1=s_p[2],
            theta_dot_2=s_p[3]
        )

        terminal = self.is_terminal(s_p)
        r = 0.0 if not terminal else 1.0

        return r, s_p, terminal

    def is_terminal(self, s):
        return bool(-np.cos(s.theta_1) - np.cos(s.theta_2 + s.theta_1) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.link_mass_1
        m2 = self.link_mass_1
        l1 = self.link_length_1
        lc1 = self.link_com_pos_1
        lc2 = self.link_com_pos_1
        i1 = self.link_moi
        i2 = self.link_moi

        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        d_theta1 = s[2]
        d_theta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + i1 + i2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * d_theta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * d_theta2 * d_theta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the paper
            dd_theta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + i2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the book
            dd_theta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * d_theta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + i2 - d2 ** 2 / d1)
        dd_theta1 = -(d2 * dd_theta2 + phi1) / d1
        return d_theta1, d_theta2, dd_theta1, dd_theta2, 0.0


def bound(x, x_min, x_max):
    """
    Truncates x between x_min and x_max
    :param x:
    :param x_min:
    :param x_max:
    :return:
    """
    return min(max(x, x_min), x_max)


def rk4(derivatives, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

     Example 1 ::
        ## 2D system
        def derivatives(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivatives, y0, t)

    Example 2::
        ## 1D system
        alpha = 2
        def derivatives(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivatives, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.

    :param derivatives: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
    :param y0: initial state vector
    :param t: sample times
    :param args: additional arguments passed to the derivative function
    :param kwargs: additional keyword arguments passed to the derivative function
    :return: Runge-Kutta approximation of the ODE
    """
    try:
        n_y = len(y0)
    except TypeError:
        y_out = np.zeros((len(t),), np.float_)
    else:
        y_out = np.zeros((len(t), n_y), np.float_)
    y_out[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = y_out[i]
        k1 = np.asarray(derivatives(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivatives(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivatives(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivatives(y0 + dt * k3, thist + dt, *args, **kwargs))
        y_out[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out


def wrap_angle(x):
    """
    Wrap angle in [- pi, pi]
    :param x: (float) angle
    :return: (float) wrapped angle
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi

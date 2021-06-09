import gym

from drmax.environments.environment import Environment


def make(name, gamma):
    if name == 'CartPole-v0':
        return GymCartPole(name=name, gamma=gamma)
    else:
        raise ValueError('Agent name [' + name + '] not found.')


class GymCartPole(Environment):

    def __init__(self, name, gamma):
        Environment.__init__(self, name=name, actions=[0, 1], gamma=gamma)

        self.env = gym.make('CartPole-v0')

    def get_state_dimension(self):
        return 4

    def get_state_dtype(self):
        return float

    def get_state_magnitude(self):
        return -self.x_threshold, self.x_threshold

    def get_initial_state(self):
        return self.env.reset()

    def step(self, s, a):
        """
        :param s: state
        :param a: action
        :return: r, s_p, is_terminal(s_p)
        """
        s_p, r, is_terminal, _ = self.env.step(a)
        return r, s_p, is_terminal

    def get_info(self):
        """
        Get general information to be saved on disk.
        """
        return {
            'name': self.name,
            'actions': self.actions,
            'gamma': self.gamma
        }

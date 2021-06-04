"""
Abstract environment class
"""


class Environment(object):

    def __init__(self, name, actions, gamma):
        self.name = name
        self.actions = actions
        self.gamma = gamma

    def get_state_dimension(self):
        return None

    def get_state_dtype(self):
        return None

    def get_state_magnitude(self):
        return None

    def get_initial_state(self):
        return None

    def step(self, s, a):
        """
        :param s: state
        :param a: actionSmall 
        :return: r, s_p, is_terminal(s_p)
        """
        return 0.0, None, False

    def get_info(self):
        """
        Get general information to be saved on disk.
        """
        return {
            'name': self.name,
            'actions': self.actions,
            'gamma': self.gamma
        }

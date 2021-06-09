"""
Abstract agent class
"""


class Agent(object):

    def __init__(self, name, actions, gamma):
        self.name = name
        self.actions = actions
        self.gamma = gamma

    def act(self, s, r, is_terminal):
        pass

    def end_of_episode(self):
        pass

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(name=self.name, actions=self.actions, gamma=self.gamma)

    def get_backup(self):
        return None

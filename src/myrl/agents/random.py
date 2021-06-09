import random

from drmax.agents.agent import Agent


class Random(Agent):

    def __init__(self, name, actions, gamma):
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

    def act(self, s, r, is_terminal):
        return random.choice(self.actions)

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(name=self.name, actions=self.actions, gamma=self.gamma)

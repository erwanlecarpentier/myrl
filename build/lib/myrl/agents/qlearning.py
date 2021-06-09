import random
from collections import defaultdict

from drmax.agents.agent import Agent


class QLearning(Agent):

    def __init__(self, name, actions, gamma, learning_rate, epsilon):
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.q_max = 1.0 / (1.0 - gamma)
        self.Q = defaultdict(lambda: defaultdict(lambda: self.q_max))
        self.prev_s = None
        self.prev_a = None

        self.learns_Q_function = True

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(name=self.name, actions=self.actions, gamma=self.gamma, learning_rate=self.learning_rate,
                      epsilon=self.epsilon)

    def reset(self):
        """
        Reset the attributes to initial state (called between instances).
        :return: None
        """
        self.Q = defaultdict(lambda: defaultdict(lambda: self.q_max))
        self.prev_s = None
        self.prev_a = None

    def act(self, s, r, is_terminal):
        """
        Acting method called online during learning.
        :param s: current state of the agent
        :param r: (float) received reward for the previous transition
        :param is_terminal: (bool) True if s is terminal
        :return: return the epsilon-greedy action wrt the current learned Q-function.
        """
        self.update(self.prev_s, self.prev_a, r, s)

        a = self.epsilon_greedy_action(s, self.Q)

        self.prev_a = a
        self.prev_s = s

        return a

    def update(self, s, a, r, s_p):
        """
        Update the current Q-function
        :param s: state
        :param a: action
        :param r: (float) reward
        :param s_p:  next state
        :return: None
        """
        if s is not None and a is not None:
            self.Q[s][a] += self.learning_rate * (r + self.gamma * self.greedy_action(s_p, self.Q)[1] - self.Q[s][a])

    def end_of_episode(self):
        """
        Reset between episodes within the same MDP.
        :return: None
        """
        self.prev_s = None
        self.prev_a = None

    def epsilon_greedy_action(self, s, f):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.greedy_action(s, f)[0]

    def greedy_action(self, s, f):
        """
        Compute the greedy action wrt the input function of (s, a).
        :param s: state at which the upper-bound is evaluated
        :param f: input function of (s, a)
        :return: return the greedy action and the achieved value by this action.
        """
        a_star = random.choice(self.actions)
        u_star = f[s][a_star]
        for a in self.actions:
            u_s_a = f[s][a]
            if u_s_a > u_star:
                u_star = u_s_a
                a_star = a
        return a_star, u_star

    def q_forward(self, s, a):
        """
        Evaluate Q function at (s, a)
        :param s: state
        :param a: action
        :return: Q-value of (s, a)
        """
        return self.Q[s][a]

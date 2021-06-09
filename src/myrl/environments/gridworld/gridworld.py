import random
import numpy as np
from collections import defaultdict

from myrl.environments.environment import Environment
from myrl.environments.state import State


def action_str_to_int(a):
    if a == 'up':
        return 0
    elif a == 'right':
        return 1
    elif a == 'down':
        return 2
    elif a == 'left':
        return 3
    else:
        raise ValueError('String action [' + a + '] illegal in GridWorld environment.')


def action_int_to_str(a):
    if a == 0:
        return 'up'
    elif a == 1:
        return 'right'
    elif a == 2:
        return 'down'
    elif a == 3:
        return 'left'
    else:
        raise ValueError('Integer action [' + a + '] illegal in GridWorld environment.')


def grid_from_name(grid_name):
    path = 'drmax/environments/gridworld/grids/' + grid_name + '.txt'
    try:
        g = np.loadtxt(path, dtype=str)
    except:
        raise ValueError('Grid name [' + grid_name + '] not found.')
    return np.rot90(np.array([list(row) for row in g]), k=-1)


class GridWorldState(State):

    def __init__(self, x, y):
        State.__init__(self, data=[x, y])
        self.x = x
        self.y = y

    def dim(self):
        return 2

    def raw(self):
        return np.array([self.x, self.y])


class GridWorld(Environment):

    def __init__(self, name, gamma, grid_name=None, size=3, slip_probability=0.0, is_goal_terminal=False):
        Environment.__init__(self, name=name, actions=[0, 1, 2, 3], gamma=gamma)

        self.grid_name = grid_name
        self.size = size
        self.slip_probability = slip_probability
        self.is_goal_terminal = is_goal_terminal

        self.y_max = self.size
        self.x_max = self.size
        self.walls = []
        self.goals = []
        self.rewards = defaultdict(lambda: 0.0)
        self.x_init = 0
        self.y_init = 0

        self.get_grid()

    def get_initial_state(self):
        return GridWorldState(self.x_init, self.y_init)

    def get_state_dimension(self):
        return 2  # len(self.get_initial_state().raw())

    def get_state_dtype(self):
        return [int, int]

    def get_state_magnitude(self):
        return np.array([1, 1], dtype=np.int), np.array([self.x_max, self.y_max], dtype=np.int)

    def get_grid(self):
        if self.grid_name is None:  # Default grid
            self.y_max = self.size
            self.x_max = self.size
            self.walls = []
            self.goals = [(self.size, self.size)]
            self.rewards[self.goals[-1]] = 1.0
            self.x_init, self.y_init = 1, 1
        else:
            g = grid_from_name(self.grid_name)
            self.parse_grid(g)

    def parse_grid(self, g):
        self.y_max, self.x_max = g.shape
        for i in range(self.y_max):
            for j in range(self.x_max):
                self.parse_grid_cell(g, i, j)

    def parse_grid_cell(self, g, i, j):
        s = (i + 1, j + 1)
        if g[i][j] == 'w':  # Wall
            self.walls.append(s)
        elif g[i][j] == 'g':  # Goal cell
            self.goals.append(s)
            self.rewards[s] = 1.0
        elif g[i][j] == 'r':  # Non-goal reward cell
            self.rewards[s] = 0.1
        elif g[i][j] == 'a':  # Initial state
            self.x_init, self.y_init = s

    def step(self, s, a):

        if self.slip_probability > random.random():  # Flip direction
            if a == 0:
                a = random.choice([3, 1])
            elif a == 2:
                a = random.choice([3, 1])
            elif a == 3:
                a = random.choice([0, 2])
            elif a == 1:
                a = random.choice([0, 2])

        if a == 0 and s.y < self.y_max and not self.is_wall(s.x, s.y + 1):
            s_p = GridWorldState(s.x, s.y + 1)
        elif a == 2 and s.y > 1 and not self.is_wall(s.x, s.y - 1):
            s_p = GridWorldState(s.x, s.y - 1)
        elif a == 1 and s.x < self.x_max and not self.is_wall(s.x + 1, s.y):
            s_p = GridWorldState(s.x + 1, s.y)
        elif a == 3 and s.x > 1 and not self.is_wall(s.x - 1, s.y):
            s_p = GridWorldState(s.x - 1, s.y)
        else:
            s_p = GridWorldState(s.x, s.y)

        # Compute reward
        r = self.rewards[(s_p.x, s_p.y)]

        return r, s_p, self.is_terminal(s_p)

    def is_wall(self, x, y):
        return (x, y) in self.walls

    def is_goal(self, x, y):
        return (x, y) in self.goals

    def is_terminal(self, s):
        if self.is_goal_terminal and self.is_goal(s.x, s.y):
            return True
        else:
            return False

    def get_q_map(self, ag):
        q_map = np.zeros(shape=(self.x_max, self.y_max, len(self.actions)))
        for x in range(1, self.x_max + 1):
            for y in range(1, self.y_max + 1):
                s = GridWorldState(x, y)
                for a in self.actions:
                    q_map[x - 1][y - 1][a] = ag.q_forward(s, a)
        return q_map

    def get_info(self):
        """
        Get general information to be saved on disk.
        """
        return {
            'name': self.name,
            'actions': self.actions,
            'gamma': self.gamma,
            'grid_name': self.grid_name,
            'size': self.size,
            'slip_probability': self.slip_probability,
            'is_goal_terminal': self.is_goal_terminal,
            'y_max': self.y_max,
            'x_max': self.x_max,
            'walls': self.walls,
            'goals': self.goals,
            'rewards': dict(self.rewards),
            'x_init': self.x_init,
            'y_init': self.y_init
        }


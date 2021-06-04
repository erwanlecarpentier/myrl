import random
from collections import deque


class ReplayMemory:

    def __init__(self, capacity):
        """
        Simple replay memory structure storing a list of transitions (s, a, r, s_p, is_terminal)
        :param capacity: (int) capacity of the replay memory. If None, unbounded capacity.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *transition):
        if self.capacity is None:
            self.memory.append(transition)
        else:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def write(self, file_name):
        data = ''.join(map(sample_to_str, self.memory))
        with open(file_name, 'w') as file:
            file.write(data)

    def __len__(self):
        return len(self.memory)


class NStepsReplayMemory(ReplayMemory):

    def __init__(self, capacity, n_step, gamma):
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_memory = deque()

    def _process_n_step_memory(self):
        s_mem, a_mem, R, si_, done = self.nstep_memory.popleft()
        if not done:
            for i in range(self.n_step-1):
                si, ai, ri, si_, done = self.nstep_memory[i]
                R += ri * self.gamma ** (i+1)
                if done:
                    break

        return [s_mem, a_mem, R, si_, done]

    def push(self, *transition):
        self.nstep_memory.append(transition)
        while len(self.nstep_memory) >= self.n_step or (self.nstep_memory and self.nstep_memory[-1][4]):
            nstep_transition = self._process_n_step_memory()
            super().push(*nstep_transition)


def sample_to_str(transition):
    s, a, r, s_, d = transition
    data = [list(s), list(a), r, list(s_), 1-int(d)]
    return ' ; '.join(map(str, data)) + '\n'

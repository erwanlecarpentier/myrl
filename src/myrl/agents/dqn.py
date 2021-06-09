import random
import torch
import collections

from drmax.agents.agent import Agent
from drmax.utils.memory import ReplayMemory
from drmax.models import QNetwork, get_optimizer
from drmax.utils.save import net_directory, get_filename


class DQN(Agent):

    def __init__(self, name, env_name, state_dimension, actions, gamma, learning_rate, epsilon, epsilon_decay,
                 replay_memory_size, minibatch_size, double_dqn, hidden_layers, target_update_period,
                 n_gradient_updates, load_constant_net, device, optimizer_name='adam',
                 momentum=0):
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        self.env_name = env_name
        self.n_a = len(actions)  # Only discrete actions are implemented
        self.state_dimension = state_dimension
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = minibatch_size
        self.double_dqn = double_dqn
        self.hidden_layers = hidden_layers
        self.target_update_period = target_update_period
        self.n_gradient_updates = n_gradient_updates
        self.load_constant_net = load_constant_net
        self.device = device
        self.optimizer_name = optimizer_name
        self.momentum = momentum

        self.loss_function = torch.nn.MSELoss(reduction='mean')  # reduction: 'mean' (default), 'sum' or 'none'
        self.learns_Q_function = True  # Called for visualization by external methods
        self.has_replay_buffer = True  # Called for visualization by external methods

        # To be reset between instances:
        self.replay_memory = ReplayMemory(self.replay_memory_size)

        # Create Q-networks
        self.Q = QNetwork(input_size=self.state_dimension, output_size=self.n_a, hidden_layers=self.hidden_layers)
        self.Q_target = QNetwork(input_size=self.state_dimension, output_size=self.n_a,
                                 hidden_layers=self.hidden_layers)

        # Load existing Q-networks if required
        if self.load_constant_net:
            net_dir = net_directory(env_name)
            q_max = 1.0 / (1.0 - self.gamma)
            filename = get_filename(key='constant_net', net_dir=net_dir, hidden_layers=self.hidden_layers, cst=q_max)
            self.Q.load(filename=filename, device=self.device)
            self.Q_target.load(filename=filename, device=self.device)

        self.optimizer = get_optimizer(optimizer_name=self.optimizer_name, parameters=self.Q.parameters(),
                                       learning_rate=self.learning_rate, momentum=self.momentum)
        self.update_counter = 0
        self.losses = []
        self.prev_s = None
        self.prev_a = None

    def re_init(self):
        """
        Re-initialization for multiple instances.
        :return: None
        """
        self.__init__(name=self.name, env_name=self.env_name, state_dimension=self.state_dimension,
                      actions=self.actions, gamma=self.gamma, learning_rate=self.learning_rate, epsilon=self.epsilon,
                      epsilon_decay=self.epsilon_decay, replay_memory_size=self.replay_memory_size,
                      minibatch_size=self.minibatch_size, double_dqn=self.double_dqn, hidden_layers=self.hidden_layers,
                      target_update_period=self.target_update_period, n_gradient_updates=self.n_gradient_updates,
                      load_constant_net=self.load_constant_net, device=self.device, optimizer_name=self.optimizer_name,
                      momentum=self.momentum)

    def act(self, s, r, is_terminal):
        """
        Acting method called online during learning.
        :param s: current state of the agent
        :param r: (float) received reward for the previous transition
        :param is_terminal: (bool) True if s is terminal
        :return: return the epsilon-greedy action wrt the current learned Q-function.
        """
        self.update(self.prev_s, self.prev_a, r, s, is_terminal)

        a = self.epsilon_greedy_action(s)

        self.prev_a = a
        self.prev_s = s

        if self.epsilon_decay is not None:
            self.epsilon *= self.epsilon_decay

        return a

    def get_batch(self):
        transitions = self.replay_memory.sample(self.minibatch_size)
        batch = list(zip(*transitions))  # Create 5 lists of states, actions, rewards, next_states, terminal

        # Convert to tensors
        states = torch.FloatTensor(batch[0]).to(self.device)
        '''
        # TODO for continuous action space
        if self.is_action_space_continuous:
            actions = torch.FloatTensor(batch[1]).to(self.device)
        else:
            actions = torch.LongTensor(batch[1]).to(self.device)
        '''
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        is_terminal = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, is_terminal

    def update_target(self):
        """
        Update the target network by copying the weights of the Q network.
        :return: None
        """
        for qt_parameters, q_parameters in zip(self.Q.parameters(), self.Q_target.parameters()):
            # Exponential decay
            # target_param.data.copy_((1-tau)*target_param.data + tau*nn_param.data)
            q_parameters.data.copy_(qt_parameters.data)

    def update(self, s, a, r, s_p, is_terminal):
        """
        Update the current Q-function
        :param s: state
        :param a: action
        :param r: (float) reward
        :param s_p: next state
        :param is_terminal: (bool) True if s_p is terminal
        :return: None
        """
        # 1) Push new sample in the replay memory
        if s is not None and a is not None:
            self.replay_memory.push(s, a, r, s_p, is_terminal)

        # 2) Update the model if enough samples in the replay memory to produce a minibatch
        if len(self.replay_memory) >= self.minibatch_size:
            self.update_counter += 1
            for _ in range(self.n_gradient_updates):
                states, actions, rewards, next_states, done = self.get_batch()

                # Compute Q(s, a): compute first Q(s), then select the columns of the taken actions
                q_s_a = self.Q(states).gather(1, actions.unsqueeze(1))

                # Compute Q(s_p)
                if self.double_dqn:
                    # Compute a_p = argmax_a Q(s_{t+1}, a)
                    next_actions = self.Q(next_states).argmax(1).unsqueeze(1)
                    # Compute Q_target(s_{t+1}, a_p) for a double DQN
                    q_sp = self.Q_target(next_states).gather(1, next_actions)
                else:
                    # Compute Q(s_p) and select the maximum
                    q_sp = self.Q_target(next_states).max(1)[0].unsqueeze(1)

                # Compute the expected Q values : y[i]= r[i] + gamma * Q(s[i+1], a[i+1])
                y_target = rewards + (1 - done) * self.gamma * q_sp
                y_target = y_target.detach()

                # Compute loss
                loss = self.loss_function(q_s_a, y_target)

                # Zero gradients, perform a backward pass (gradient update), and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Target update
                if self.update_counter == self.target_update_period:
                    self.update_target()
                    self.update_counter = 0

                # Record loss
                self.losses.append(loss.item())

    def end_of_episode(self):
        """
        Reset between episodes within the same MDP.
        :return: None
        """
        self.prev_s = None
        self.prev_a = None

    def epsilon_greedy_action(self, s):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.greedy_action(s)

    def greedy_action(self, s):
        """
        Compute the greedy action wrt the Q-network
        :param s: state at which the upper-bound is evaluated
        :param f: input function of (s, a)
        :return: return the greedy action and the achieved value by this action.
        """
        with torch.no_grad():
            # state = torch.FloatTensor(s).to(self.device)
            return self.actions[self.Q(torch.FloatTensor(s).to(self.device)).cpu().detach().argmax().item()]

    def q_forward(self, s, a):
        """
        Evaluate Q function at (s, a)
        :param s: state
        :param a: action
        :return: Q-value of (s, a)
        """
        s = torch.FloatTensor([s]).to(self.device)
        a = torch.LongTensor([a]).to(self.device)
        return self.Q(s).gather(1, a.unsqueeze(1)).item()

    def get_replay_buffer(self):
        """
        Return a list of the states in the replay memory.
        If index is provided, only save a subset of the list items
        :param index:
        :return:
        """
        states = [(s[0].x, s[0].y) for s in self.replay_memory.memory]
        return dict(collections.Counter(states))

    def get_backup(self):
        return {'loss': self.losses}

import torch


def get_optimizer(parameters, optimizer_name=None, learning_rate=0.05, momentum=0):
    if optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate)
    else:  # Default is SGD
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)


class FullyConnectedNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        """
        A fully connected network model with relu activation functions.
        :param input_size: (int) input dimensionality
        :param output_size: (int) output dimensionality
        :param hidden_layers: (list) hidden layers sizes

        Example:
            net = FullyConnectedNet(input_size=3, output_size=2, hidden_layers=[5, 7])
        """
        super().__init__()

        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_layers[0])])
        for i in range(1, len(hidden_layers)):
            self.layers.append(torch.nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.output = torch.nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))

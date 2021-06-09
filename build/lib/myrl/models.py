import torch


def get_optimizer(parameters, optimizer_name=None, learning_rate=0.05, momentum=0):
    if optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate)
    else:  # Default is SGD
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)


class QNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_layers[0])])
        for i in range(1, len(hidden_layers)):
            self.layers.append(torch.nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.output = torch.nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        for la in self.layers:
            x = torch.relu(la(x))
        return self.output(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))

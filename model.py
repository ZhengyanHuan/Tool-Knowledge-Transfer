import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 16):
        super(encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.network(x), p=2, dim=1)  # L2 norm


class classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.network(x)
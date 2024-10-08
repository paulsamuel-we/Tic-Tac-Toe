import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increase neurons from 128 to 256
        self.fc2 = nn.Linear(256, 256)  # Deeper network
        self.fc3 = nn.Linear(256, 128)  # More layers for better feature extraction
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

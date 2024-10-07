import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)  # Input is the 3x3 board flattened (9 cells)
        self.fc2 = nn.Linear(128, 128)

        # Policy head: gives probabilities for actions
        self.policy_head = nn.Linear(128, 9)

        # Value head: gives the probability of winning
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.policy_head(x), dim=-1)  # Probabilities for each action
        value = torch.tanh(self.value_head(x))  # Scalar value (between -1 and 1)

        return policy, value

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)  # Input size matches the 3x3 board (flattened to 9)
        self.fc_policy = nn.Linear(128, 9)  # Output policy over the 9 possible moves
        self.fc_value = nn.Linear(128, 1)   # Output value (win/loss/draw)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # After fc1: (batch_size, 128)
        policy = F.softmax(self.fc_policy(x), dim=-1)  # Policy output
        value = torch.tanh(self.fc_value(x))  # Value output
        return policy, value

def load_model(model_path):
    model = PolicyValueNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

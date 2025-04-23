import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 24x24 -> 24x24
        self.pool = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 12x12 -> 12x12 -> 6x6 after pool
        self.fc1 = nn.Linear(32 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 5)  # assuming 5 classes

    def forward(self, x):
        x = x.view(-1, 1, 24, 24)  # reshape from [batch_size, 576] to [batch_size, 1, 24, 24]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
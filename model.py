import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Basic DQN (For environments like CartPole)
# --------------------------------------------------

class DQN(nn.Module):
    """
    Simple Fully-Connected DQN
    Used for low-dimensional state spaces (e.g., CartPole)
    """

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        """
        Forward pass
        Input: state vector
        Output: Q-values for each action
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


# --------------------------------------------------
# Convolutional DQN (For Atari / Image Input)
# --------------------------------------------------

class ConvDQN(nn.Module):
    """
    CNN-based DQN for image inputs (Atari-like environments)
    """

    def __init__(self, input_channels, action_dim):
        super(ConvDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Output size after conv layers (for 84x84 input)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        """
        Forward pass
        Input: stacked frames (B, C, H, W)
        Output: Q-values
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)   # Flatten

        x = F.relu(self.fc1(x))

        return self.fc2(x)


# --------------------------------------------------
# Dueling DQN (Advanced Architecture)
# --------------------------------------------------

class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture
    Splits Value and Advantage streams
    """

    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()

        # Shared feature layer
        self.fc1 = nn.Linear(state_dim, 128)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        """
        Forward pass
        Combines value and advantage
        """

        x = F.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    The Deep Q-Network.

    Takes a stack of 4 preprocessed game frames (4, 84, 84)
    and outputs a Q-value for each possible action.
    """

    def __init__(self, n_actions: int):
        """
        Args:
            n_actions: Number of possible actions in the environment.
        """
        super().__init__()

        # --- Convolutional layers (the "eyes") ---
        # Input: (batch, 4, 84, 84)
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        # Output: (batch, 32, 20, 20)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # Output: (batch, 64, 9, 9)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # Output: (batch, 64, 7, 7)

        # --- Fully connected layers (the "thinking") ---
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)

        Returns:
            Q-values tensor of shape (batch_size, n_actions)
        """
        # Pass through conv layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten: (batch, 64, 7, 7) → (batch, 3136)
        x = x.reshape(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))

        # Output layer — NO activation (raw Q-values)
        return self.fc2(x)
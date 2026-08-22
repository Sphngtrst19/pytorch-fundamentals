import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.flatten(x)

        return self.network(x)
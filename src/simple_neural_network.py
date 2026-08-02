import torch
import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(
            in_features=4,
            out_features=8
        )

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(
            in_features=8,
            out_features=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x
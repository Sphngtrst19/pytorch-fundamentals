from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from simple_neural_network import SimpleNeuralNetwork


RANDOM_SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "simple_neural_network.pth"


def calculate_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor
) -> float:
    model.eval()

    with torch.no_grad():
        logits = model(features)
        predictions = torch.argmax(logits, dim=1)

    accuracy = (predictions == labels).float().mean().item()

    return accuracy


def main():
    torch.manual_seed(RANDOM_SEED)

    features = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
        dtype=torch.float32
    )

    labels = torch.tensor(
        [0, 1, 1, 0],
        dtype=torch.long
    )

    dataset = TensorDataset(features, labels)

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = SimpleNeuralNetwork()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for batch_features, batch_labels in data_loader:
            logits = model(batch_features)

            loss = criterion(
                logits,
                batch_labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)

        if epoch % 20 == 0:
            accuracy = calculate_accuracy(
                model,
                features,
                labels
            )

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {average_loss:.4f} | "
                f"Accuracy: {accuracy:.2%}"
            )

    final_accuracy = calculate_accuracy(
        model,
        features,
        labels
    )

    print(f"\nFinal accuracy: {final_accuracy:.2%}")

    model.eval()

    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=1)

    print("\nPredictions")
    print(predictions)

    print("\nActual")
    print(labels)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()